"""Unit tests for production inference engine, schema validation, and batch processing."""

import pytest
from pydantic import ValidationError

from predictlab.data.load import load_raw_dataset
from predictlab.inference.engine import InferenceEngine
from predictlab.inference.schemas import (
    CustomerInputRecord,
    InferenceValidationError,
)
from predictlab.preprocessing.split import split_dataset


def test_customer_input_record_valid() -> None:
    """Test valid CustomerInputRecord schema instantiation."""
    record = {
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
    obj = CustomerInputRecord(**record)
    assert obj.tenure == 12.0
    assert obj.MonthlyCharges == 55.20


def test_customer_input_record_negative_tenure_raises() -> None:
    """Test negative tenure fails schema validation."""
    with pytest.raises(ValidationError):
        CustomerInputRecord(
            gender="Female",
            SeniorCitizen=0,
            Partner="Yes",
            Dependents="No",
            tenure=-5,  # Invalid
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="DSL",
            OnlineSecurity="Yes",
            OnlineBackup="No",
            DeviceProtection="Yes",
            TechSupport="No",
            StreamingTV="No",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=50.0,
            TotalCharges=100.0,
        )


def test_customer_input_record_non_finite_charges_raises() -> None:
    """Test NaN / non-finite charges fail schema validation."""
    with pytest.raises(ValidationError):
        CustomerInputRecord(
            gender="Male",
            SeniorCitizen=0,
            Partner="No",
            Dependents="No",
            tenure=5,
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="No",
            OnlineSecurity="No",
            OnlineBackup="No",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="No",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="No",
            PaymentMethod="Mailed check",
            MonthlyCharges=float("nan"),  # Invalid
            TotalCharges=100.0,
        )


def test_inference_engine_single_prediction() -> None:
    """Test InferenceEngine single record prediction with explanation."""
    df = load_raw_dataset()
    X_train, _, _, _, _ = split_dataset(df, save_manifest=False)

    sample_dict = X_train.iloc[0].to_dict()
    # Inject customerID to verify engine handles it safely
    sample_dict["customerID"] = "TEST-0001"

    engine = InferenceEngine()
    result = engine.predict_single(sample_dict, include_explanation=True)

    assert 0.0 <= result.estimated_probability <= 1.0
    assert result.predicted_class in (0, 1)
    assert result.predicted_label in ("Churned", "Retained")
    assert result.risk_tier in ("Low", "Medium", "High")
    assert result.applied_threshold == engine.decision_threshold
    assert result.explanation is not None
    assert len(result.explanation["top_churn_drivers"]) > 0


def test_inference_engine_missing_column_raises() -> None:
    """Test InferenceEngine raises InferenceValidationError when required column is omitted."""
    df = load_raw_dataset()
    sample_dict = df.iloc[0].to_dict()
    sample_dict.pop("Contract")  # Remove required column

    engine = InferenceEngine()
    with pytest.raises(InferenceValidationError, match="Missing required predictor columns"):
        engine.predict_single(sample_dict)


def test_inference_engine_unexpected_column_raises() -> None:
    """Test InferenceEngine raises InferenceValidationError when unexpected column is passed."""
    df = load_raw_dataset()
    sample_dict = df.iloc[0].to_dict()
    sample_dict["unexpected_feature"] = "malformed"

    engine = InferenceEngine()
    with pytest.raises(InferenceValidationError, match="Unexpected columns found"):
        engine.predict_single(sample_dict)


def test_inference_engine_batch_prediction_ordering() -> None:
    """Test batch prediction preserves exact input row ordering."""
    df = load_raw_dataset()
    X_train, _, _, _, _ = split_dataset(df, save_manifest=False)

    records = [X_train.iloc[i].to_dict() for i in range(5)]
    for i, r in enumerate(records):
        r["customerID"] = f"CUST-{i:03d}"

    engine = InferenceEngine()
    batch_out = engine.predict_batch(records, error_handling="atomic")

    assert batch_out.total_records == 5
    assert batch_out.successful_predictions == 5
    assert batch_out.failed_predictions == 0

    # Verify order preservation
    for i in range(5):
        assert batch_out.predictions[i].record_index == i
        assert batch_out.predictions[i].customer_id == f"CUST-{i:03d}"
        assert batch_out.predictions[i].is_valid is True
        assert batch_out.predictions[i].prediction is not None


def test_inference_engine_batch_atomic_vs_per_record() -> None:
    """Test batch prediction failure modes (atomic raises vs per_record captures)."""
    df = load_raw_dataset()
    X_train, _, _, _, _ = split_dataset(df, save_manifest=False)

    good_rec = X_train.iloc[0].to_dict()
    bad_rec = dict(good_rec)
    bad_rec["tenure"] = -99  # Invalid

    batch = [good_rec, bad_rec]
    engine = InferenceEngine()

    # 1. Atomic mode raises
    with pytest.raises(InferenceValidationError, match="Atomic batch validation failed"):
        engine.predict_batch(batch, error_handling="atomic")

    # 2. Per-record mode captures failure gracefully
    batch_res = engine.predict_batch(batch, error_handling="per_record")
    assert batch_res.total_records == 2
    assert batch_res.successful_predictions == 1
    assert batch_res.failed_predictions == 1
    assert batch_res.predictions[0].is_valid is True
    assert batch_res.predictions[1].is_valid is False
    assert "tenure" in batch_res.predictions[1].error_message
