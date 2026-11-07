"""Unit tests for PredictLab Streamlit application services."""

import io
from typing import Any

import pandas as pd
import pytest

from predictlab.app.services.data_service import (
    generate_sample_batch_csv,
    get_sample_customer_presets,
    load_cv_model_metrics,
    load_dataset_overview_stats,
    load_permutation_importance_data,
    load_risk_tier_summary,
    load_threshold_benchmark_data,
)
from predictlab.app.services.inference_service import (
    get_engine_metadata,
    get_inference_engine,
    run_batch_prediction,
    run_single_prediction,
)
from predictlab.inference.schemas import InferenceValidationError


@pytest.fixture
def valid_customer_dict() -> dict[str, Any]:
    """Sample valid customer record for testing."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.20,
        "TotalCharges": 662.40,
    }


def test_get_inference_engine_loads_canonical_calibrated_artifact() -> None:
    """Test get_inference_engine loads and caches the canonical Phase 5 calibrated model."""
    engine = get_inference_engine()
    assert engine is not None
    assert engine.artifact_path.name == "model_phase5_calibrated.joblib"
    assert engine.calibration_method == "sigmoid"
    assert engine.decision_threshold == 0.19
    assert engine.model_name == "Hist Gradient Boosting (Sigmoid Calibrated)"


def test_get_engine_metadata() -> None:
    """Test metadata extraction from inference engine."""
    engine = get_inference_engine()
    meta = get_engine_metadata(engine)
    assert "artifact_path" in meta
    assert "model_name" in meta
    assert meta["calibration_method"] == "sigmoid"
    assert meta["default_threshold"] == 0.19
    assert isinstance(meta["metadata"], dict)


def test_run_single_prediction_valid(valid_customer_dict: dict[str, Any]) -> None:
    """Test running single prediction through application service."""
    engine = get_inference_engine()
    res = run_single_prediction(
        engine=engine,
        record=valid_customer_dict,
        threshold=0.19,
        include_explanation=True,
    )
    assert 0.0 <= res.estimated_probability <= 1.0
    assert res.predicted_class in (0, 1)
    assert res.predicted_label in ("Churned", "Retained")
    assert res.applied_threshold == 0.19
    assert res.risk_tier in ("Low", "Medium", "High")
    assert res.explanation is not None
    assert "top_churn_drivers" in res.explanation
    assert "top_retention_drivers" in res.explanation


def test_run_single_prediction_invalid_raises(valid_customer_dict: dict[str, Any]) -> None:
    """Test validation errors are raised properly."""
    engine = get_inference_engine()
    bad_record = dict(valid_customer_dict)
    bad_record["tenure"] = -10  # Negative tenure invalid

    with pytest.raises(InferenceValidationError):
        run_single_prediction(engine, bad_record)


def test_run_batch_prediction_mixed_records() -> None:
    """Test batch prediction with mixed valid and invalid records in per-record mode."""
    engine = get_inference_engine()
    csv_str = generate_sample_batch_csv(include_invalid=True)
    df = pd.read_csv(io.StringIO(csv_str))
    records = df.to_dict(orient="records")

    res = run_batch_prediction(engine, records, threshold=0.19)
    assert res.total_records == 6
    assert res.successful_predictions == 5
    assert res.failed_predictions == 1
    assert len(res.predictions) == 6

    # Verify order preservation
    for i, r in enumerate(res.predictions):
        assert r.record_index == i

    # Last record should be invalid
    last_r = res.predictions[-1]
    assert not last_r.is_valid
    assert last_r.error_message is not None
    assert "tenure" in last_r.error_message


def test_data_service_stats_and_presets() -> None:
    """Test data service static summaries and presets."""
    stats = load_dataset_overview_stats()
    assert stats["total_records"] == 7043
    assert stats["churned_records"] == 1869
    assert stats["retained_records"] == 5174
    assert stats["input_features"] == 19

    presets = get_sample_customer_presets()
    assert len(presets) >= 4
    for _name, p in presets.items():
        assert "tenure" in p
        assert "MonthlyCharges" in p
        assert "TotalCharges" in p


def test_data_service_report_loaders() -> None:
    """Test loading benchmark reports from data service."""
    cv = load_cv_model_metrics()
    assert "mean_metrics" in cv
    assert "average_precision" in cv["mean_metrics"]
    assert cv["mean_metrics"]["average_precision"] > 0.60

    df_cost = load_threshold_benchmark_data()
    assert not df_cost.empty
    assert "threshold" in df_cost.columns
    assert "true_positives" in df_cost.columns

    df_risk = load_risk_tier_summary()
    assert not df_risk.empty
    assert len(df_risk) == 3
    assert list(df_risk["risk_tier"]) == ["Low", "Medium", "High"]

    df_perm = load_permutation_importance_data()
    assert not df_perm.empty
    assert "importance_mean" in df_perm.columns
    assert len(df_perm) == 47
