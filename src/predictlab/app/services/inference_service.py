"""Cached inference engine service wrapper for Streamlit."""

import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from predictlab.inference.engine import InferenceEngine
from predictlab.inference.schemas import (
    BatchPredictionOutput,
    InferenceError,
    ModelArtifactError,
    SinglePredictionOutput,
)
from predictlab.logging_config import get_logger

logger = get_logger(__name__)

MAX_BATCH_RECORDS = 15000


def validate_and_parse_batch_csv(
    file_bytes_or_buffer: Any,
    max_records: int = MAX_BATCH_RECORDS,
) -> tuple[pd.DataFrame | None, str | None]:
    """Validate, decode, and parse an uploaded CSV input before inference.

    Handles:
    - Bytes, io.BytesIO, io.StringIO, file-like objects, or file paths.
    - Strict UTF-8 validation (catches UnicodeDecodeError).
    - Empty file detection (catches pd.errors.EmptyDataError or 0-row results).
    - Malformed structure (catches pd.errors.ParserError).
    - Cohort size limits (enforces max_records prior to running any inference).

    Returns:
        tuple[pd.DataFrame | None, str | None]: (parsed_dataframe, error_message)
        If valid, returns (df, None).
        If invalid, returns (None, error_message).
    """
    if file_bytes_or_buffer is None:
        return None, "No file provided."

    raw_bytes: bytes | None = None
    csv_str: str | None = None

    if isinstance(file_bytes_or_buffer, bytes):
        raw_bytes = file_bytes_or_buffer
    elif hasattr(file_bytes_or_buffer, "read"):
        if hasattr(file_bytes_or_buffer, "seek"):
            file_bytes_or_buffer.seek(0)
        content = file_bytes_or_buffer.read()
        if isinstance(content, bytes):
            raw_bytes = content
        elif isinstance(content, str):
            csv_str = content
        else:
            return None, f"Unsupported file content type: {type(content).__name__}"
    elif isinstance(file_bytes_or_buffer, str):
        p = Path(file_bytes_or_buffer)
        if p.exists() and p.is_file():
            raw_bytes = p.read_bytes()
        else:
            csv_str = file_bytes_or_buffer
    else:
        return None, f"Unsupported input type: {type(file_bytes_or_buffer).__name__}"

    if raw_bytes is not None:
        try:
            csv_str = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as err:
            return None, f"Invalid file encoding: {err}. Please ensure the file is encoded in UTF-8."

    assert csv_str is not None
    if not csv_str.strip():
        return None, "The uploaded CSV file is empty. Please provide a file with at least one record."

    try:
        df = pd.read_csv(io.StringIO(csv_str))
    except pd.errors.EmptyDataError:
        return None, "The uploaded CSV file is empty. Please provide a file with at least one record."
    except pd.errors.ParserError as err:
        return None, f"Malformed CSV structure: {err}"
    except Exception as err:
        return None, f"Failed to read CSV file: {err}"

    if len(df) == 0:
        return None, "The uploaded CSV file is empty. Please provide a file with at least one record."

    if len(df) > max_records:
        return None, (
            f"Uploaded cohort contains {len(df):,} records, exceeding the "
            f"{max_records:,} record interactive limit."
        )

    return df, None


@st.cache_resource(show_spinner="Loading predictive inference engine...")
def get_inference_engine(artifact_path: Path | str | None = None) -> InferenceEngine:
    """Retrieve or initialize the cached singleton InferenceEngine instance.

    Guarantees:
    - Serialized artifact is loaded once and cached across user interactions.
    - Thread-safe, stateless inference operations.
    """
    try:
        engine = InferenceEngine(model_artifact_path=artifact_path)
        logger.info("Streamlit InferenceEngine cached successfully from: %s", engine.artifact_path)
        return engine
    except ModelArtifactError as err:
        logger.error("Failed to initialize InferenceEngine: %s", err)
        raise
    except Exception as err:
        logger.error("Unexpected error loading InferenceEngine: %s", err)
        raise ModelArtifactError(f"Unexpected error loading model artifact: {err}") from err


def get_engine_metadata(engine: InferenceEngine) -> dict[str, Any]:
    """Extract clean summary information from the active inference engine."""
    return {
        "artifact_path": str(engine.artifact_path),
        "artifact_sha256": engine.artifact_sha256,
        "model_name": engine.model_name,
        "calibration_method": engine.calibration_method,
        "default_threshold": engine.decision_threshold,
        "feature_count": engine.feature_count,
        "ensemble_size": engine.ensemble_size,
        "metadata": engine.metadata,
    }


def run_single_prediction(
    engine: InferenceEngine,
    record: dict[str, Any],
    threshold: float | None = None,
    include_explanation: bool = True,
) -> SinglePredictionOutput:
    """Execute single prediction with safe fallback for explanation errors."""
    try:
        return engine.predict_single(
            record=record,
            threshold=threshold,
            include_explanation=include_explanation,
        )
    except InferenceError:
        raise
    except Exception as err:
        # If SHAP explanation failed internally, retry without explanation
        if include_explanation:
            logger.warning("SHAP explanation failed, falling back to pure prediction: %s", err)
            try:
                res = engine.predict_single(
                    record=record,
                    threshold=threshold,
                    include_explanation=False,
                )
                return res
            except Exception as inner_err:
                raise InferenceError(f"Prediction execution failed: {inner_err}") from inner_err
        raise InferenceError(f"Prediction execution failed: {err}") from err


def run_batch_prediction(
    engine: InferenceEngine,
    records: list[dict[str, Any]],
    threshold: float | None = None,
) -> BatchPredictionOutput:
    """Execute batch predictions using per-record error handling policy."""
    return engine.predict_batch(
        records=records,
        error_handling="per_record",
        threshold=threshold,
        include_explanations=False,
    )
