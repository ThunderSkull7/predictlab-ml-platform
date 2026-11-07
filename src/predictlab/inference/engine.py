"""Production-ready inference engine for single and batch customer churn predictions."""

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


class InferenceEngine:
    """Production-grade inference engine providing single-record and batch churn predictions.

    Features:
    1. Schema validation rejecting missing, non-finite, negative, or invalid predictors.
    2. Strips 'customerID' so identifiers are never used as model predictors.
    3. Delegates feature transformations internally to the pipeline without manual feature duplication.
    4. Deterministic scoring with configurable decision thresholds and risk tiers.
    5. Atomic and per-record error handling modes for batch prediction.
    6. Optional local TreeSHAP prediction explanations.
    """

    def __init__(
        self,
        model_artifact_path: Path | str | None = None,
        decision_threshold: float | None = None,
        risk_tierer: RiskTierer | None = None,
        enable_explanations: bool = True,
    ) -> None:
        """Initialize the inference engine by loading and validating a serialized model artifact.

        Parameters
        ----------
        model_artifact_path : Path or str, optional
            Path to serialized joblib bundle. Defaults to artifacts/model_phase5_calibrated.joblib
            with fallback to artifacts/model.joblib.
        decision_threshold : float, optional
            Decision threshold overriding artifact default.
        risk_tierer : RiskTierer, optional
            Risk tierer instance. Defaults to standard 3-tier RiskTierer().
        enable_explanations : bool, default=True
            Whether to initialize the local SHAP explainer.
        """
        self._artifact_path = self._resolve_artifact_path(model_artifact_path)
        self._model, self._metadata = self._load_and_validate_artifact(self._artifact_path)

        # Set decision threshold
        if decision_threshold is not None:
            self._decision_threshold = float(decision_threshold)
        else:
            self._decision_threshold = float(self._metadata.get("decision_threshold", 0.50))

        # Risk tierer
        self._risk_tierer = risk_tierer if risk_tierer is not None else RiskTierer()

        # Model metadata
        self._model_version = str(self._metadata.get("model_version", self._metadata.get("saved_at_utc", "1.0.0")))
        self._model_name = str(self._metadata.get("model_name", "HistGradientBoostingClassifier"))
        self._calibration_method = str(self._metadata.get("calibration_method", "uncalibrated"))

        # Explainability
        self._enable_explanations = enable_explanations
        self._explainer: TreeShapExplainer | None = None

        logger.info(
            "InferenceEngine initialized successfully from: %s [Model: '%s', Calib: '%s', Threshold: %.2f]",
            self._artifact_path.resolve(),
            self._model_name,
            self._calibration_method,
            self._decision_threshold,
        )

    def _resolve_artifact_path(self, user_path: Path | str | None) -> Path:
        """Determine model artifact path with graceful phase-aware fallback."""
        settings = get_settings()
        if user_path is not None:
            p = Path(user_path)
            if not p.exists():
                raise ModelArtifactError(f"Specified model artifact does not exist: {p.resolve()}")
            return p

        # Check default paths in preference order
        phase5_path = settings.artifacts_dir / "model_phase5_calibrated.joblib"
        phase4_path = settings.artifacts_dir / "model.joblib"

        if phase5_path.exists():
            return phase5_path
        if phase4_path.exists():
            return phase4_path

        raise ModelArtifactError(
            f"No model artifact found in {settings.artifacts_dir.resolve()}. "
            "Please train or serialize an artifact before inference."
        )

    def _load_and_validate_artifact(self, path: Path) -> tuple[Any, dict[str, Any]]:
        """Load joblib bundle and verify structural integrity."""
        try:
            bundle = joblib.load(path)
        except Exception as e:
            raise ModelArtifactError(f"Failed to deserialize model artifact at '{path}': {e}") from e

        if not isinstance(bundle, dict):
            raise ModelArtifactError(f"Model artifact at '{path}' is not a valid bundle dictionary.")

        if "pipeline" not in bundle and "model" not in bundle:
            raise ModelArtifactError(f"Model artifact bundle missing 'pipeline' key at '{path}'.")

        model = bundle.get("pipeline", bundle.get("model"))
        metadata = bundle.get("metadata", {})

        if not hasattr(model, "predict_proba"):
            raise ModelArtifactError(f"Loaded model object {type(model)} does not support predict_proba().")

        return model, metadata

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

