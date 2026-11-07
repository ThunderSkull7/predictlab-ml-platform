"""Customer-level local prediction explanations using Shapley additive feature attributions."""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from predictlab.logging_config import get_logger

logger = get_logger(__name__)


class FeatureAttribution(BaseModel):
    """Individual feature contribution to the prediction margin."""

    feature: str = Field(description="Transformed feature identifier")
    clean_feature: str = Field(description="Human-readable feature name")
    feature_value: float = Field(description="Numerical or encoded value in model space")
    shap_value: float = Field(description="Shapley additive contribution to log-odds margin")
    impact_direction: str = Field(description="'Increases Churn Risk' if positive, 'Decreases Churn Risk' if negative")


class CustomerLocalExplanation(BaseModel):
    """Structured local explanation for an individual customer prediction."""

    raw_input_features: dict[str, Any] = Field(description="Original customer input record")
    model_estimated_probability: float = Field(description="Model-estimated calibrated churn probability")
    applied_threshold: float = Field(description="Applied classification decision threshold")
    predicted_class: int = Field(description="Binary prediction: 1 (Churned) or 0 (Retained)")
    predicted_label: str = Field(description="'Churned' or 'Retained'")
    explained_target: str = Field(
        default="ensemble_base_decision_margin",
        description="Target function explained (e.g. 'ensemble_base_decision_margin' or 'base_estimator_decision_margin')",
    )
    ensemble_size: int = Field(default=1, description="Number of base tree estimators aggregated for the explanation")
    base_value_margin: float = Field(description="Expected ensemble decision margin (base value in log-odds)")
    model_output_margin: float = Field(description="Sum of base value and all SHAP values (log-odds ensemble margin)")
    top_churn_drivers: list[FeatureAttribution] = Field(description="Top features increasing churn margin")
    top_retention_drivers: list[FeatureAttribution] = Field(description="Top features decreasing churn margin")
    all_attributions: list[FeatureAttribution] = Field(description="Complete list of all feature contributions")
    methodological_limitations: str = Field(description="Scientific interpretation boundaries")


class TreeShapExplainer:
    """Provides local Shapley additive explanations for the fitted tree pipeline or calibrated ensemble."""

    def __init__(
        self,
        fitted_model: Any,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize SHAP explainer from fitted pipeline or calibrated classifier.

        Parameters
        ----------
        fitted_model : Pipeline or CalibratedClassifierCV
            Trained model object.
        feature_names : list of str, optional
            Transformed feature names.
        """
        import shap

        self._fitted_model = fitted_model
        self._explainers: list[shap.TreeExplainer] = []

        # Extract preprocessor and core tree estimator(s)
        if hasattr(fitted_model, "named_steps"):
            self._preprocessor = fitted_model.named_steps["preprocessor"]
            classifier = fitted_model.named_steps["classifier"]
            self._explainers = [shap.TreeExplainer(classifier)]
            self._is_calibrated_cv = False
            self._explained_target = "base_estimator_decision_margin"
        elif hasattr(fitted_model, "calibrated_classifiers_"):
            # CalibratedClassifierCV wrapping pipeline ensemble
            sub_clfs = list(fitted_model.calibrated_classifiers_)
            self._preprocessor = sub_clfs[0].estimator.named_steps["preprocessor"]
            self._explainers = [
                shap.TreeExplainer(sub.estimator.named_steps["classifier"])
                for sub in sub_clfs
            ]
            self._is_calibrated_cv = True
            self._explained_target = "ensemble_base_decision_margin"
        else:
            raise ValueError(f"Unsupported model object structure: {type(fitted_model)}")

        if feature_names is not None:
            self._feature_names = list(feature_names)
        elif hasattr(self._preprocessor, "get_feature_names_out"):
            self._feature_names = list(self._preprocessor.get_feature_names_out())
        else:
            self._feature_names = [f"feature_{i}" for i in range(47)]

    def explain_record(
        self,
        raw_record: dict[str, Any] | pd.DataFrame | pd.Series,
        threshold: float = 0.5,
        top_n: int = 5,
    ) -> CustomerLocalExplanation:
        """Generate local attribution explanation for a single customer record.

        Parameters
        ----------
        raw_record : dict, DataFrame, or Series
            Input customer predictor record.
        threshold : float, default=0.5
            Decision threshold.
        top_n : int, default=5
            Number of top churn/retention drivers to highlight.

        Returns
        -------
        CustomerLocalExplanation
            Structured Pydantic explanation object.
        """
        if isinstance(raw_record, dict):
            df_in = pd.DataFrame([raw_record])
            raw_dict = dict(raw_record)
        elif isinstance(raw_record, pd.Series):
            df_in = pd.DataFrame([raw_record])
            raw_dict = raw_record.to_dict()
        else:
            df_in = raw_record.copy().iloc[:1]
            raw_dict = df_in.iloc[0].to_dict()

        # Compute model-estimated calibrated probability
        proba = float(self._fitted_model.predict_proba(df_in)[0, 1])
        pred_class = int(proba >= threshold)
        pred_label = "Churned" if pred_class == 1 else "Retained"

        # Transform features
        X_trans = self._preprocessor.transform(df_in)

        # Compute Shapley values across all estimators in the ensemble and average
        shap_res_list = [expl(X_trans) for expl in self._explainers]
        shap_vals = np.mean([r.values[0] for r in shap_res_list], axis=0)
        base_val = float(np.mean([r.base_values[0] for r in shap_res_list]))
        model_margin = float(base_val + np.sum(shap_vals))

        attributions: list[FeatureAttribution] = []
        for name, val, sv in zip(self._feature_names, X_trans[0], shap_vals, strict=True):
            clean_name = name.replace("num__", "").replace("cat__", "")
            direction = "Increases Churn Risk" if sv > 0 else "Decreases Churn Risk"
            attributions.append(
                FeatureAttribution(
                    feature=name,
                    clean_feature=clean_name,
                    feature_value=round(float(val), 4),
                    shap_value=round(float(sv), 5),
                    impact_direction=direction,
                )
            )

        # Sort top drivers
        pos_drivers = sorted(
            [a for a in attributions if a.shap_value > 0],
            key=lambda a: a.shap_value,
            reverse=True,
        )[:top_n]

        neg_drivers = sorted(
            [a for a in attributions if a.shap_value < 0],
            key=lambda a: a.shap_value,
        )[:top_n]

        limitations = (
            f"Local Shapley attributions explain the {self._explained_target} "
            f"(aggregated across {len(self._explainers)} ensemble estimators) on the uncalibrated "
            "log-odds scale relative to the expected training baseline. "
            "Additivity holds strictly on this margin scale (output_margin = base_value + sum(shap_values)). "
            "Because Platt sigmoid calibration is non-linear, Shapley values do NOT additively sum to "
            "the final calibrated probability. "
            "Furthermore, Shapley values quantify statistical model behavior on observational data, "
            "NOT causal intervention outcomes. Changing a feature in reality will not causally alter "
            "churn propensity as estimated."
        )

        return CustomerLocalExplanation(
            raw_input_features=raw_dict,
            model_estimated_probability=round(proba, 4),
            applied_threshold=round(threshold, 4),
            predicted_class=pred_class,
            predicted_label=pred_label,
            explained_target=self._explained_target,
            ensemble_size=len(self._explainers),
            base_value_margin=round(base_val, 4),
            model_output_margin=round(model_margin, 4),
            top_churn_drivers=pos_drivers,
            top_retention_drivers=neg_drivers,
            all_attributions=attributions,
            methodological_limitations=limitations,
        )

