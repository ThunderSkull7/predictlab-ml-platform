"""Model artifact and experiment report persistence module."""

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from predictlab.config import get_settings
from predictlab.logging_config import get_logger

logger = get_logger(__name__)


def get_environment_metadata() -> dict[str, str]:
    """Capture current runtime environment and package versions."""
    import sklearn

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
    }


def save_experiment_json(
    data: dict[str, Any] | BaseModel,
    filename: str,
    output_dir: Path | str | None = None,
) -> Path:
    """Save experiment results as a formatted JSON document under reports/experiments/.

    Parameters
    ----------
    data : dict or BaseModel
        Data to serialize.
    filename : str
        Target JSON filename (e.g. 'benchmark_results.json').
    output_dir : Path or str, optional
        Destination directory (defaults to reports/experiments).

    Returns
    -------
    Path
        Path to persisted JSON file.
    """
    settings = get_settings()
    out_dir = Path(output_dir) if output_dir is not None else settings.reports_dir / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / filename
    if isinstance(data, BaseModel):
        payload = data.model_dump()
    else:
        payload = data

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Saved experiment JSON artifact: %s", file_path.resolve())
    return file_path


def save_model_artifact(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    file_path: Path | str | None = None,
) -> Path:
    """Serialize the complete fitted Pipeline and metadata bundle using Joblib.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted scikit-learn Pipeline (preprocessing + estimator).
    metadata : dict[str, Any]
        Dictionary documenting dataset SHA-256, model family, hyperparameters,
        training timestamp, and package versions.
    file_path : Path or str, optional
        Target artifact destination (defaults to artifacts/model.joblib).

    Returns
    -------
    Path
        Path to saved artifact.
    """
    settings = get_settings()
    target_path = Path(file_path) if file_path is not None else settings.artifacts_dir / "model.joblib"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Attach timestamp and environment metadata
    enriched_metadata = {
        **metadata,
        "saved_at_utc": datetime.now(UTC).isoformat(),
        "environment": get_environment_metadata(),
    }

    bundle = {
        "pipeline": pipeline,
        "metadata": enriched_metadata,
    }

    joblib.dump(bundle, target_path)
    logger.info("Serialized model artifact bundle to: %s", target_path.resolve())
    return target_path


def load_model_artifact(file_path: Path | str | None = None) -> dict[str, Any]:
    """Deserialize fitted model artifact bundle from disk.

    Parameters
    ----------
    file_path : Path or str, optional
        Path to serialized joblib file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'pipeline' and 'metadata'.
    """
    settings = get_settings()
    target_path = Path(file_path) if file_path is not None else settings.artifacts_dir / "model.joblib"
    if not target_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {target_path.resolve()}")

    bundle = joblib.load(target_path)
    logger.info("Loaded model artifact bundle from: %s", target_path.resolve())
    return bundle

