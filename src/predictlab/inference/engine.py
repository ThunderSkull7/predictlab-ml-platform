"""Production-ready inference engine for single and batch customer churn predictions."""

import hashlib
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from pydantic import ValidationError

from predictlab.config import get_settings
from predictlab.explainability.local_attribution import TreeShapExplainer
from predictlab.inference.schemas import (
    RAW_PREDICTORS,
    BatchPredictionOutput,
    BatchPredictionRecord,
    CustomerInputRecord,
    InferenceValidationError,
    ModelArtifactError,
    SinglePredictionOutput,
)
from predictlab.logging_config import get_logger
from predictlab.thresholding.risk_tiers import RiskTierer

logger = get_logger(__name__)

CANONICAL_ARTIFACT_FILENAME = "model_phase5_calibrated.joblib"
CANONICAL_ARTIFACT_SHA256 = "b773eb58e59825c8cef7169c4166566ebddfa77b5e584853ac492dafc5dd0963"
EXPECTED_FEATURE_COUNT = 47
EXPECTED_CALIBRATION_METHOD = "sigmoid"
EXPECTED_ENSEMBLE_SIZE = 5
EXPECTED_DEFAULT_THRESHOLD = 0.19


class InferenceEngine:
    """Production-grade inference engine providing single-record and batch churn predictions.

    Features:
    1. Schema validation rejecting missing, non-finite, negative, or invalid predictors.
    2. Strips 'customerID' so identifiers are never used as model predictors.
    3. Delegates feature transformations internally to the pipeline without manual feature duplication.
    4. Deterministic scoring with configurable decision thresholds and risk tiers.
    5. Atomic and per-record error handling modes for batch prediction.
    6. Optional local TreeSHAP prediction explanations.
    7. Fail-closed artifact resolution and strict structural pre-flight verification.
    """

    def __init__(
        self,
        model_artifact_path: Path | str | None = None,
        decision_threshold: float | None = None,
        risk_tierer: RiskTierer | None = None,
        enable_explanations: bool = True,
        require_calibrated: bool = True,
        allow_development_fallback: bool = False,
        expected_sha256: str | None = None,
    ) -> None:
        """Initialize the inference engine by loading and validating a serialized model artifact.

        Parameters
        ----------
        model_artifact_path : Path or str, optional
            Path to serialized joblib bundle. Defaults strictly to
            artifacts/model_phase5_calibrated.joblib (fail-closed).
        decision_threshold : float, optional
            Decision threshold overriding artifact default.
        risk_tierer : RiskTierer, optional
            Risk tierer instance. Defaults to standard 3-tier RiskTierer().
        enable_explanations : bool, default=True
            Whether to initialize the local SHAP explainer.
        require_calibrated : bool, default=True
            Whether to enforce strict calibrated ensemble constraints (5 estimators, sigmoid, 47 features).
        allow_development_fallback : bool, default=False
            Isolated development switch. Must NEVER be True in production paths.
        expected_sha256 : str | None, optional
            Explicit expected SHA-256 checksum to enforce. If omitted, defaults to
            the canonical calibrated checksum when loading the production model.
        """
        self._require_calibrated = require_calibrated
        self._artifact_path = self._resolve_artifact_path(
            model_artifact_path,
            allow_development_fallback=allow_development_fallback,
        )
        self._model, self._metadata, self._artifact_sha256 = self._load_and_validate_artifact(
            self._artifact_path,
            require_calibrated=require_calibrated,
            expected_sha256=expected_sha256,
        )

        # Set decision threshold
        if decision_threshold is not None:
            self._decision_threshold = float(decision_threshold)
        else:
            self._decision_threshold = float(
                self._metadata.get(
                    "decision_threshold",
                    EXPECTED_DEFAULT_THRESHOLD if self._metadata.get("calibration_method") == "sigmoid" else 0.50,
                )
            )

        # Risk tierer
        self._risk_tierer = risk_tierer if risk_tierer is not None else RiskTierer()

        # Model metadata
        self._model_version = str(self._metadata.get("model_version", self._metadata.get("saved_at_utc", "1.0.0")))
        self._model_name = str(self._metadata.get("model_name", "HistGradientBoostingClassifier"))
        self._calibration_method = str(self._metadata.get("calibration_method", "uncalibrated"))
        feature_names_meta = self._metadata.get("feature_names", [])
        self._feature_count = int(self._metadata.get("feature_count", len(feature_names_meta)))
        self._feature_names = list(feature_names_meta)
        calibrated_classifiers = getattr(self._model, "calibrated_classifiers_", None)
        self._ensemble_size = len(calibrated_classifiers) if calibrated_classifiers is not None else 1

        # Explainability
        self._enable_explanations = enable_explanations
        self._explainer: TreeShapExplainer | None = None

        logger.info(
            "InferenceEngine initialized successfully from: %s [SHA: %s, Model: '%s', Calib: '%s', Features: %d, Threshold: %.2f]",
            self._artifact_path.name,
            self._artifact_sha256[:12],
            self._model_name,
            self._calibration_method,
            self._feature_count,
            self._decision_threshold,
        )

    def _resolve_artifact_path(
        self,
        user_path: Path | str | None,
        allow_development_fallback: bool = False,
    ) -> Path:
        """Determine model artifact path with fail-closed security guarantees.

        Requirements:
        - In production (user_path is None), only the canonical calibrated artifact
          'model_phase5_calibrated.joblib' is accepted.
        - If missing, unreadable, or invalid, execution fails closed immediately.
        - Uncalibrated models ('model.joblib', 'model_phase4.joblib') are NEVER loaded
          silently in normal operation.
        - Development fallback is isolated behind explicit allow_development_fallback=True.
        """
        settings = get_settings()
        if user_path is not None:
            p = Path(user_path)
            if not p.exists() or not p.is_file():
                raise ModelArtifactError(f"Specified model artifact does not exist: {p.name}")
            return p

        # Production path: canonical calibrated artifact ONLY
        canonical_path = settings.artifacts_dir / CANONICAL_ARTIFACT_FILENAME
        if canonical_path.exists() and canonical_path.is_file():
            return canonical_path

        # Isolated development fallback (strictly disabled by default)
        if allow_development_fallback:
            dev_path = settings.artifacts_dir / "model.joblib"
            if dev_path.exists() and dev_path.is_file():
                logger.warning(
                    "DEVELOPMENT FALLBACK ACTIVE: Canonical calibrated artifact is missing. "
                    "Loaded uncalibrated development model 'model.joblib'. "
                    "This mode must NEVER be used in production."
                )
                return dev_path

        # Fail closed
        raise ModelArtifactError(
            f"Canonical calibrated model artifact '{CANONICAL_ARTIFACT_FILENAME}' was not found "
            f"in artifacts directory. Production inference fails closed: uncalibrated fallback "
            f"models ('model.joblib', 'model_phase4.joblib') are strictly rejected in the production path."
        )

    def _load_and_validate_artifact(
        self,
        path: Path,
        require_calibrated: bool = True,
        expected_sha256: str | None = None,
    ) -> tuple[Any, dict[str, Any], str]:
        """Load joblib bundle, compute checksum, and perform rigorous structural validation.

        Parameters
        ----------
        path : Path
            Path to the joblib bundle.
        require_calibrated : bool, default=True
            Whether to enforce strict calibrated ensemble constraints.
        expected_sha256 : str | None, optional
            Expected SHA-256 checksum to strictly enforce before deserialization.

        Returns
        -------
        tuple[Any, dict[str, Any], str]
            (model_or_pipeline, metadata_dictionary, sha256_checksum)

        Raises
        ------
        ModelArtifactError
            If file is unreadable, corrupted, violates structural contracts, or fails checksum verification.
        """
        # 1. Compute SHA-256 Checksum from raw file bytes
        try:
            artifact_bytes = path.read_bytes()
            sha256_hash = hashlib.sha256(artifact_bytes).hexdigest()
        except Exception as err:
            raise ModelArtifactError(
                f"Failed to read model artifact '{path.name}' for checksum verification: {err}"
            ) from err

        # 2. Strict Checksum Verification BEFORE Deserialization
        target_sha256 = expected_sha256
        if target_sha256 is None and path.name == CANONICAL_ARTIFACT_FILENAME:
            settings = get_settings()
            target_sha256 = getattr(settings, "canonical_model_sha256", CANONICAL_ARTIFACT_SHA256)

        if target_sha256 is not None:
            if sha256_hash.lower() != target_sha256.lower():
                raise ModelArtifactError(
                    f"Model artifact '{path.name}' checksum mismatch. "
                    f"Expected SHA-256 '{target_sha256}', but computed '{sha256_hash}'. "
                    "Refusing to deserialize unverified or tampered model artifact."
                )

        # 3. Safe Deserialization (Only reached after checksum verification succeeds)
        try:
            bundle = joblib.load(path)
        except Exception as err:
            raise ModelArtifactError(
                f"Failed to deserialize model artifact '{path.name}': {err}"
            ) from err

        if not isinstance(bundle, dict):
            raise ModelArtifactError(
                f"Model artifact '{path.name}' is invalid: root structure must be a dictionary bundle."
            )

        if "pipeline" not in bundle and "model" not in bundle:
            raise ModelArtifactError(
                f"Model artifact '{path.name}' is missing required 'pipeline' key in bundle."
            )

        model = bundle.get("pipeline", bundle.get("model"))
        metadata = bundle.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ModelArtifactError(
                f"Model artifact '{path.name}' contains invalid metadata structure (expected dict)."
            )

        # 3. Verify predict_proba capability
        if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
            raise ModelArtifactError(
                f"Loaded model in '{path.name}' ({type(model).__name__}) does not support predict_proba()."
            )

        # 4. Strict Calibrated Ensemble Structural Validation
        if require_calibrated:
            calib_method = metadata.get("calibration_method")
            if calib_method != EXPECTED_CALIBRATION_METHOD:
                raise ModelArtifactError(
                    f"Model artifact '{path.name}' validation failed: expected '{EXPECTED_CALIBRATION_METHOD}' "
                    f"calibration, found '{calib_method}'. Uncalibrated or incompatible models cannot serve production inference."
                )

            # Validate calibrated ensemble structure
            calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
            if calibrated_classifiers is None or len(calibrated_classifiers) != EXPECTED_ENSEMBLE_SIZE:
                actual_count = len(calibrated_classifiers) if calibrated_classifiers is not None else 0
                raise ModelArtifactError(
                    f"Model artifact '{path.name}' validation failed: expected {EXPECTED_ENSEMBLE_SIZE}-estimator "
                    f"calibrated ensemble, found {actual_count} estimators."
                )

            # Validate expected feature count
            feature_names = metadata.get("feature_names", [])
            feature_count = metadata.get("feature_count", len(feature_names))
            if feature_count != EXPECTED_FEATURE_COUNT:
                raise ModelArtifactError(
                    f"Model artifact '{path.name}' validation failed: expected {EXPECTED_FEATURE_COUNT} baseline features, "
                    f"found {feature_count} features."
                )

            # Validate expected operating threshold
            default_thresh = metadata.get("decision_threshold")
            if default_thresh is not None and abs(float(default_thresh) - EXPECTED_DEFAULT_THRESHOLD) > 1e-4:
                raise ModelArtifactError(
                    f"Model artifact '{path.name}' validation failed: expected default operating "
                    f"threshold {EXPECTED_DEFAULT_THRESHOLD}, found {default_thresh}."
                )

        return model, metadata, sha256_hash

    def _get_explainer(self) -> TreeShapExplainer:
        """Lazy-initialize SHAP explainer when local explanations are requested."""
        if self._explainer is None:
            feature_names = self._metadata.get("feature_names")
            self._explainer = TreeShapExplainer(self._model, feature_names=feature_names)
        return self._explainer

    @property
    def artifact_path(self) -> Path:
        """Return the resolved path of the loaded artifact."""
        return self._artifact_path

    @property
    def artifact_sha256(self) -> str:
        """Return the verified SHA-256 checksum of the loaded artifact."""
        return self._artifact_sha256

    @property
    def feature_count(self) -> int:
        """Return the verified input feature count (47 baseline features)."""
        return self._feature_count

    @property
    def feature_names(self) -> list[str]:
        """Return list of verified feature names."""
        return list(self._feature_names)

    @property
    def ensemble_size(self) -> int:
        """Return number of calibrated estimators in the ensemble (5)."""
        return self._ensemble_size

    @property
    def model_name(self) -> str:
        """Return descriptive model name."""
        return self._model_name

    @property
    def calibration_method(self) -> str:
        """Return calibration strategy applied."""
        return self._calibration_method

    @property
    def decision_threshold(self) -> float:
        """Return operating decision threshold."""
        return self._decision_threshold

    @property
    def risk_tierer(self) -> RiskTierer:
        """Return operational risk tierer."""
        return self._risk_tierer

    @property
    def metadata(self) -> dict[str, Any]:
        """Return complete metadata dictionary."""
        return dict(self._metadata)

    def validate_and_clean_record(self, raw_record: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """Validate input customer record, strip customerID, and check schema constraints.

        Parameters
        ----------
        raw_record : dict
            Input record key-value pairs.

        Returns
        -------
        tuple[dict, str | None]
            (cleaned_19_predictor_dict, optional_customer_id)

        Raises
        ------
        InferenceValidationError
            If required columns are missing, unexpected columns exist, or values violate constraints.
        """
        record_copy = dict(raw_record)
        customer_id = record_copy.pop("customerID", None)
        if customer_id is not None:
            customer_id = str(customer_id)

        # Remove ground-truth target if present in raw test payload
        record_copy.pop("Churn", None)

        # Check for missing predictors
        missing = [col for col in RAW_PREDICTORS if col not in record_copy]
        if missing:
            raise InferenceValidationError(f"Missing required predictor columns: {missing}")

        # Check for unexpected columns
        unexpected = [col for col in record_copy if col not in RAW_PREDICTORS]
        if unexpected:
            raise InferenceValidationError(f"Unexpected columns found in record: {unexpected}")

        # Pydantic domain and type validation
        try:
            validated = CustomerInputRecord(**record_copy)
            clean_dict = validated.model_dump()
        except ValidationError as e:
            raise InferenceValidationError(f"Record validation failed: {e}") from e

        return clean_dict, customer_id

    def predict_single(
        self,
        record: dict[str, Any],
        threshold: float | None = None,
        include_explanation: bool = False,
        top_n_drivers: int = 5,
    ) -> SinglePredictionOutput:
        """Generate prediction for a single customer record.

        Parameters
        ----------
        record : dict
            Dictionary of raw customer attributes.
        threshold : float, optional
            Threshold overriding engine default for this call.
        include_explanation : bool, default=False
            Whether to compute local TreeSHAP feature attributions.
        top_n_drivers : int, default=5
            Number of top churn/retention drivers in explanation.

        Returns
        -------
        SinglePredictionOutput
            Structured prediction output.
        """
        clean_record, _ = self.validate_and_clean_record(record)
        thresh = float(threshold) if threshold is not None else self._decision_threshold

        df_in = pd.DataFrame([clean_record])
        proba = float(self._model.predict_proba(df_in)[0, 1])
        pred_class = int(proba >= thresh)
        pred_label = "Churned" if pred_class == 1 else "Retained"
        tier = self._risk_tierer.assign_tier(proba)

        explanation_dict: dict[str, Any] | None = None
        if include_explanation and self._enable_explanations:
            explainer = self._get_explainer()
            explanation_obj = explainer.explain_record(df_in, threshold=thresh, top_n=top_n_drivers)
            explanation_dict = explanation_obj.model_dump()

        return SinglePredictionOutput(
            model_version=self._model_version,
            model_name=self._model_name,
            calibration_method=self._calibration_method,
            estimated_probability=round(proba, 4),
            predicted_class=pred_class,
            predicted_label=pred_label,
            applied_threshold=round(thresh, 4),
            risk_tier=tier,
            explanation=explanation_dict,
        )

    def predict_batch(
        self,
        records: list[dict[str, Any]] | pd.DataFrame,
        error_handling: Literal["atomic", "per_record"] = "atomic",
        threshold: float | None = None,
        include_explanations: bool = False,
    ) -> BatchPredictionOutput:
        """Generate predictions for a batch of customer records.

        Guarantees:
        1. Output ordering strictly matches input ordering.
        2. In 'atomic' mode, fails entirely if any record is invalid.
        3. In 'per_record' mode, invalid records are captured with error messages,
           while valid records are scored without failure.

        Parameters
        ----------
        records : list of dict, or pd.DataFrame
            Customer records to score.
        error_handling : {'atomic', 'per_record'}, default='atomic'
            Batch error handling policy.
        threshold : float, optional
            Decision threshold overriding engine default.
        include_explanations : bool, default=False
            Whether to attach local SHAP explanations.

        Returns
        -------
        BatchPredictionOutput
            Structured batch results.
        """
        thresh = float(threshold) if threshold is not None else self._decision_threshold

        if isinstance(records, pd.DataFrame):
            records_list = records.to_dict(orient="records")
        else:
            records_list = list(records)

        n_total = len(records_list)
        batch_records: list[BatchPredictionRecord] = []
        valid_indices: list[int] = []
        valid_cleaned_dicts: list[dict[str, Any]] = []
        customer_ids: list[str | None] = []

        # 1. Validation pass
        for idx, rec in enumerate(records_list):
            try:
                clean_dict, cid = self.validate_and_clean_record(rec)
                valid_indices.append(idx)
                valid_cleaned_dicts.append(clean_dict)
                customer_ids.append(cid)
            except InferenceValidationError as err:
                if error_handling == "atomic":
                    raise InferenceValidationError(
                        f"Atomic batch validation failed at record {idx}: {err}"
                    ) from err
                batch_records.append(
                    BatchPredictionRecord(
                        record_index=idx,
                        customer_id=rec.get("customerID"),
                        is_valid=False,
                        prediction=None,
                        error_message=str(err),
                    )
                )

        # 2. Vectorized scoring pass for all valid records
        if valid_cleaned_dicts:
            df_valid = pd.DataFrame(valid_cleaned_dicts)
            probas = self._model.predict_proba(df_valid)[:, 1]
            tiers = self._risk_tierer.assign_tiers(probas)

            for local_idx, orig_idx in enumerate(valid_indices):
                p = float(probas[local_idx])
                pred_c = int(p >= thresh)
                pred_lbl = "Churned" if pred_c == 1 else "Retained"
                cid = customer_ids[local_idx]

                expl_dict = None
                if include_explanations and self._enable_explanations:
                    expl = self._get_explainer().explain_record(df_valid.iloc[local_idx : local_idx + 1], threshold=thresh)
                    expl_dict = expl.model_dump()

                single_out = SinglePredictionOutput(
                    model_version=self._model_version,
                    model_name=self._model_name,
                    calibration_method=self._calibration_method,
                    estimated_probability=round(p, 4),
                    predicted_class=pred_c,
                    predicted_label=pred_lbl,
                    applied_threshold=round(thresh, 4),
                    risk_tier=tiers[local_idx],
                    explanation=expl_dict,
                )

                batch_records.append(
                    BatchPredictionRecord(
                        record_index=orig_idx,
                        customer_id=cid,
                        is_valid=True,
                        prediction=single_out,
                        error_message=None,
                    )
                )

        # Re-sort to preserve exact original input order
        batch_records.sort(key=lambda r: r.record_index)

        successful = sum(1 for r in batch_records if r.is_valid)
        failed = sum(1 for r in batch_records if not r.is_valid)

        return BatchPredictionOutput(
            model_version=self._model_version,
            total_records=n_total,
            successful_predictions=successful,
            failed_predictions=failed,
            error_handling_mode=error_handling,
            predictions=batch_records,
        )

