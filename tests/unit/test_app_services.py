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
    validate_and_parse_batch_csv,
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


def test_validate_and_parse_batch_csv_empty() -> None:
    """Test boundary case: empty CSV string, buffer, or header-only file."""
    # Empty string
    df, err = validate_and_parse_batch_csv("")
    assert df is None
    assert err is not None
    assert "empty" in err.lower()

    # Whitespace only
    df, err = validate_and_parse_batch_csv("   \n\t  ")
    assert df is None
    assert err is not None
    assert "empty" in err.lower()

    # Empty bytes buffer
    df, err = validate_and_parse_batch_csv(io.BytesIO(b""))
    assert df is None
    assert err is not None
    assert "empty" in err.lower()

    # Header only, 0 data rows
    header_only = "gender,SeniorCitizen,Partner,Dependents\n"
    df, err = validate_and_parse_batch_csv(io.StringIO(header_only))
    assert df is None
    assert err is not None
    assert "empty" in err.lower()


def test_validate_and_parse_batch_csv_invalid_encoding() -> None:
    """Test boundary case: non-UTF8 binary encoding."""
    # Invalid UTF-8 bytes (e.g. invalid sequence)
    invalid_bytes = b"\xff\xfe\x00\x00\x99\x88\x77"
    df, err = validate_and_parse_batch_csv(io.BytesIO(invalid_bytes))
    assert df is None
    assert err is not None
    assert "Invalid file encoding" in err


def test_validate_and_parse_batch_csv_malformed() -> None:
    """Test boundary case: malformed CSV structure."""
    # Inconsistent fields / corrupted CSV structure
    malformed_csv = "col1,col2\nval1,val2\nval1,val2,val3,extra_unquoted\n"
    df, err = validate_and_parse_batch_csv(malformed_csv)
    assert df is None
    assert err is not None
    assert "Malformed CSV structure" in err or "Failed to read CSV" in err


def test_validate_and_parse_batch_csv_limits_15000_and_15001() -> None:
    """Test boundary cases: exactly 15,000 valid records vs 15,001 records limit."""
    # Build single valid line
    header = "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
    row = "Female,0,Yes,No,1,Yes,No,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85\n"

    # Exactly 15,000 rows
    content_15000 = header + (row * 15000)
    df_15000, err_15000 = validate_and_parse_batch_csv(io.StringIO(content_15000), max_records=15000)
    assert err_15000 is None
    assert df_15000 is not None
    assert len(df_15000) == 15000

    # 15,001 rows: must be rejected before inference
    content_15001 = header + (row * 15001)
    df_15001, err_15001 = validate_and_parse_batch_csv(io.StringIO(content_15001), max_records=15000)
    assert df_15001 is None
    assert err_15001 is not None
    assert "15,001 records" in err_15001
    assert "exceeding the 15,000 record interactive limit" in err_15001


def test_run_batch_prediction_all_records_valid(valid_customer_dict: dict[str, Any]) -> None:
    """Test batch prediction when 100% of records are valid."""
    engine = get_inference_engine()
    records = [dict(valid_customer_dict) for _ in range(5)]
    res = run_batch_prediction(engine, records, threshold=0.19)

    assert res.total_records == 5
    assert res.successful_predictions == 5
    assert res.failed_predictions == 0
    assert len(res.predictions) == 5
    for i, pred in enumerate(res.predictions):
        assert pred.record_index == i
        assert pred.is_valid is True
        assert pred.prediction is not None
        assert 0.0 <= pred.prediction.estimated_probability <= 1.0
        assert pred.prediction.predicted_class in (0, 1)
        assert pred.prediction.predicted_label in ("Churned", "Retained")


def test_run_batch_prediction_all_records_invalid(valid_customer_dict: dict[str, Any]) -> None:
    """Test batch prediction when 100% of records are invalid."""
    engine = get_inference_engine()
    bad_record = dict(valid_customer_dict)
    bad_record["MonthlyCharges"] = -999.0  # Invalid monthly charges

    records = [dict(bad_record) for _ in range(4)]
    res = run_batch_prediction(engine, records, threshold=0.19)

    assert res.total_records == 4
    assert res.successful_predictions == 0
    assert res.failed_predictions == 4
    assert len(res.predictions) == 4
    for i, pred in enumerate(res.predictions):
        assert pred.record_index == i
        assert pred.is_valid is False
        assert pred.prediction is None
        assert pred.error_message is not None
        assert "MonthlyCharges" in pred.error_message


def test_run_batch_prediction_order_preservation(valid_customer_dict: dict[str, Any]) -> None:
    """Test that input ordering is strictly preserved across arbitrary valid/invalid sequences."""
    engine = get_inference_engine()
    good_1 = dict(valid_customer_dict, tenure=1)
    bad_2 = dict(valid_customer_dict, tenure=-5)
    good_3 = dict(valid_customer_dict, tenure=72)
    bad_4 = dict(valid_customer_dict, MonthlyCharges=-10.0)
    good_5 = dict(valid_customer_dict, tenure=36)

    records = [good_1, bad_2, good_3, bad_4, good_5]
    res = run_batch_prediction(engine, records, threshold=0.19)

    assert res.total_records == 5
    assert res.successful_predictions == 3
    assert res.failed_predictions == 2

    # Check order
    assert [p.record_index for p in res.predictions] == [0, 1, 2, 3, 4]
    assert [p.is_valid for p in res.predictions] == [True, False, True, False, True]


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
