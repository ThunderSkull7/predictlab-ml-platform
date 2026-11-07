"""Unit tests for production inference engine, schema validation, and batch processing."""

import shutil
from pathlib import Path

import joblib
import pytest
from pydantic import ValidationError

from predictlab.config import get_settings
from predictlab.data.load import load_raw_dataset
from predictlab.inference.engine import InferenceEngine
from predictlab.inference.schemas import (
    CustomerInputRecord,
    InferenceValidationError,
    ModelArtifactError,
)
from predictlab.preprocessing.split import split_dataset


class _MockEstimatorNoProba:
    """Mock estimator without predict_proba for unit testing."""

    pass


class _MockEstimatorWithProba:
    """Mock estimator with predict_proba for unit testing."""

    def predict_proba(self, X: object) -> object:
        return None


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


def test_canonical_calibrated_artifact_loads_successfully() -> None:
    """Requirement 3.1: Canonical calibrated artifact exists and loads successfully with verified properties."""
    engine = InferenceEngine()
    assert engine.artifact_path.name == "model_phase5_calibrated.joblib"
    assert engine.artifact_sha256 == "b773eb58e59825c8cef7169c4166566ebddfa77b5e584853ac492dafc5dd0963"
    assert engine.calibration_method == "sigmoid"
    assert engine.decision_threshold == 0.19
    assert engine.feature_count == 47
    assert len(engine.feature_names) == 47
    assert engine.ensemble_size == 5


def test_canonical_artifact_missing_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Requirement 3.2: Canonical artifact missing: initialization fails closed with ModelArtifactError."""
    artifacts_sub = tmp_path / "artifacts"
    artifacts_sub.mkdir()
    fake_settings = get_settings().model_copy(update={"project_root": tmp_path})
    monkeypatch.setattr("predictlab.inference.engine.get_settings", lambda: fake_settings)

    with pytest.raises(ModelArtifactError, match="Canonical calibrated model artifact.*not found"):
        InferenceEngine()


def test_canonical_artifact_unreadable_or_corrupted_fails(tmp_path: Path) -> None:
    """Requirement 3.3: Canonical artifact unreadable or corrupted: initialization fails."""
    corrupt_file = tmp_path / "model_phase5_calibrated.joblib"
    corrupt_file.write_bytes(b"CORRUPTED_NON_JOBLIB_DATA_12345")

    with pytest.raises(ModelArtifactError, match="Failed to deserialize model artifact"):
        InferenceEngine(model_artifact_path=corrupt_file)


def test_uncalibrated_model_joblib_not_loaded_when_canonical_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Requirement 3.4: Uncalibrated model.joblib exists but canonical artifact missing: it is NOT loaded."""
    artifacts_sub = tmp_path / "artifacts"
    artifacts_sub.mkdir()
    real_model_joblib = get_settings().artifacts_dir / "model.joblib"
    if real_model_joblib.exists():
        shutil.copy(real_model_joblib, artifacts_sub / "model.joblib")

    fake_settings = get_settings().model_copy(update={"project_root": tmp_path})
    monkeypatch.setattr("predictlab.inference.engine.get_settings", lambda: fake_settings)

    # In production mode (allow_development_fallback=False), must fail closed
    with pytest.raises(ModelArtifactError, match="uncalibrated fallback models .* are strictly rejected"):
        InferenceEngine()


def test_archived_phase4_joblib_not_loaded_when_canonical_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Requirement 3.5: Archived model_phase4.joblib exists but canonical artifact missing: it is NOT loaded."""
    artifacts_sub = tmp_path / "artifacts"
    artifacts_sub.mkdir()
    real_p4 = get_settings().artifacts_dir / "model_phase4.joblib"
    if real_p4.exists():
        shutil.copy(real_p4, artifacts_sub / "model_phase4.joblib")

    fake_settings = get_settings().model_copy(update={"project_root": tmp_path})
    monkeypatch.setattr("predictlab.inference.engine.get_settings", lambda: fake_settings)

    with pytest.raises(ModelArtifactError, match="uncalibrated fallback models .* are strictly rejected"):
        InferenceEngine()


def test_invalid_estimator_type_or_missing_metadata_fails(tmp_path: Path) -> None:
    """Requirement 3.6: Invalid estimator type or missing calibration metadata: initialization fails."""
    # 1. Non-dict bundle
    bad_bundle_file = tmp_path / "non_dict.joblib"
    joblib.dump(["not", "a", "dict"], bad_bundle_file)
    with pytest.raises(ModelArtifactError, match="root structure must be a dictionary bundle"):
        InferenceEngine(model_artifact_path=bad_bundle_file)

    # 2. Missing predict_proba
    no_proba_file = tmp_path / "no_proba.joblib"
    joblib.dump({"pipeline": _MockEstimatorNoProba(), "metadata": {"calibration_method": "sigmoid"}}, no_proba_file)
    with pytest.raises(ModelArtifactError, match="does not support predict_proba"):
        InferenceEngine(model_artifact_path=no_proba_file)

    # 3. Missing or wrong calibration method
    wrong_calib_file = tmp_path / "wrong_calib.joblib"
    joblib.dump(
        {
            "pipeline": _MockEstimatorWithProba(),
            "metadata": {"calibration_method": "isotonic", "feature_count": 47},
        },
        wrong_calib_file,
    )
    with pytest.raises(ModelArtifactError, match="expected 'sigmoid' calibration"):
        InferenceEngine(model_artifact_path=wrong_calib_file)


def test_loaded_artifact_retains_feature_count_and_contract() -> None:
    """Requirement 3.7: Loaded artifact retains expected feature count (47) and inference contract."""
    engine = InferenceEngine()
    assert engine.feature_count == 47
    assert len(engine.feature_names) == 47
    assert engine.ensemble_size == 5

    # Check inference contract
    df = load_raw_dataset()
    sample = df.iloc[0].to_dict()
    res = engine.predict_single(sample, include_explanation=False)
    assert 0.0 <= res.estimated_probability <= 1.0
    assert res.predicted_class in (0, 1)
    assert res.predicted_label in ("Churned", "Retained")
    assert res.applied_threshold == 0.19


def test_repeated_predictions_remain_deterministic() -> None:
    """Requirement 3.8: Repeated predictions remain strictly deterministic."""
    engine = InferenceEngine()
    df = load_raw_dataset()
    sample = df.iloc[5].to_dict()

    p1 = engine.predict_single(sample, include_explanation=False)
    p2 = engine.predict_single(sample, include_explanation=False)

    assert p1.estimated_probability == p2.estimated_probability
    assert p1.predicted_class == p2.predicted_class
    assert p1.predicted_label == p2.predicted_label
    assert p1.risk_tier == p2.risk_tier


def test_complete_15000_row_batch_inference_workflow() -> None:
    """Requirement 5: Complete 15,000-row batch inference workflow verifies output generation and ordering."""
    engine = InferenceEngine()
    df = load_raw_dataset()
    sample = df.iloc[0].to_dict()
    # Build 15,000 records
    records = [dict(sample, customerID=f"BATCH-{i:05d}") for i in range(15000)]

    batch_output = engine.predict_batch(records, error_handling="per_record", include_explanations=False)

    assert batch_output.total_records == 15000
    assert batch_output.successful_predictions == 15000
    assert batch_output.failed_predictions == 0
    assert len(batch_output.predictions) == 15000

    # Verify boundary and spot-check ordering
    assert batch_output.predictions[0].record_index == 0
    assert batch_output.predictions[0].customer_id == "BATCH-00000"
    assert batch_output.predictions[7499].record_index == 7499
    assert batch_output.predictions[7499].customer_id == "BATCH-07499"
    assert batch_output.predictions[14999].record_index == 14999
    assert batch_output.predictions[14999].customer_id == "BATCH-14999"
