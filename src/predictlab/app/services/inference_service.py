"""Cached inference engine service wrapper for Streamlit."""

from pathlib import Path
from typing import Any

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
        "model_name": engine.model_name,
        "calibration_method": engine.calibration_method,
        "default_threshold": engine.decision_threshold,
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
