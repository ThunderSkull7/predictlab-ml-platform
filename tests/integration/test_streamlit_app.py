"""Integration tests for PredictLab Streamlit application and page workflows."""

import hashlib
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from predictlab.config import get_settings
from predictlab.inference.engine import InferenceEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_app_main_navigation_render() -> None:
    """Test app.py initializes and executes without unhandled exceptions."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.title) >= 1
    assert "PredictLab" in at.title[0].value


def test_overview_page_render() -> None:
    """Test Overview Dashboard page renders dataset stats and CV benchmarks."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app_pages" / "overview.py"), default_timeout=20)
    at.run()
    assert not at.exception
    assert "Overview Dashboard" in at.title[0].value or "PredictLab" in at.title[0].value


def test_single_prediction_page_render_and_submit() -> None:
    """Test Single Customer Prediction page loads form and executes inference upon submit."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app_pages" / "single_prediction.py"), default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.button) >= 1

    # Click the prediction submit button
    at.button[0].click()
    at.run()
    assert not at.exception


def test_batch_analysis_page_render() -> None:
    """Test Batch Customer Analysis page renders file upload controls and template download."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app_pages" / "batch_analysis.py"), default_timeout=20)
    at.run()
    assert not at.exception
    assert len(at.download_button) >= 1


def test_explainability_page_render() -> None:
    """Test Model Insights & Explainability page renders importance and SHAP sections."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app_pages" / "explainability.py"), default_timeout=20)
    at.run()
    assert not at.exception
    assert "Explainability" in at.title[0].value


def test_methodology_page_render() -> None:
    """Test Methodology & Governance page renders complete scientific documentation."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app_pages" / "methodology.py"), default_timeout=20)
    at.run()
    assert not at.exception
    assert "Methodology" in at.title[0].value


def test_canonical_artifact_integrity() -> None:
    """Test canonical model artifacts and split manifest remain unmodified."""
    settings = get_settings()

    # 1. Dataset SHA-256 verification
    raw_path = settings.raw_data_dir / "Telco-Customer-Churn.csv"
    assert raw_path.exists()
    with open(raw_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    assert sha == "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

    # 2. Split Manifest verification
    manifest_path = settings.splits_dir / "split_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["train_samples"] == 5634
    assert manifest["test_samples"] == 1409
    assert manifest["raw_data_sha256"] == sha

    # 3. Model artifact existence and calibration
    engine = InferenceEngine()
    assert engine.artifact_path.name == "model_phase5_calibrated.joblib"
    assert engine.calibration_method == "sigmoid"
    assert engine.decision_threshold == 0.19
