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
    model_estimated_probability: float = Field(description="Model-estimated churn probability")
    applied_threshold: float = Field(description="Applied classification decision threshold")
    predicted_class: int = Field(description="Binary prediction: 1 (Churned) or 0 (Retained)")
    predicted_label: str = Field(description="'Churned' or 'Retained'")
    base_value_margin: float = Field(description="Expected model margin (base value in log-odds)")
    model_output_margin: float = Field(description="Sum of base value and all SHAP values (log-odds)")
    top_churn_drivers: list[FeatureAttribution] = Field(description="Top features increasing churn probability")
    top_retention_drivers: list[FeatureAttribution] = Field(description="Top features decreasing churn probability")
    all_attributions: list[FeatureAttribution] = Field(description="Complete list of all feature contributions")
    methodological_limitations: str = Field(description="Scientific interpretation boundaries")


class TreeShapExplainer:
    """Provides local Shapley additive explanations for the fitted tree pipeline."""

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

        # Extract preprocessor and core tree estimator
        if hasattr(fitted_model, "named_steps"):
            self._preprocessor = fitted_model.named_steps["preprocessor"]
            self._classifier = fitted_model.named_steps["classifier"]
            self._is_calibrated_cv = False
        elif hasattr(fitted_model, "calibrated_classifiers_"):
            # CalibratedClassifierCV wrapping pipeline
            sub_clf = fitted_model.calibrated_classifiers_[0]
            if hasattr(sub_clf, "estimator") and hasattr(sub_clf.estimator, "named_steps"):
                self._preprocessor = sub_clf.estimator.named_steps["preprocessor"]
                self._classifier = sub_clf.estimator.named_steps["classifier"]
            else:
                base_pipe = fitted_model.estimator
                self._preprocessor = base_pipe.named_steps["preprocessor"]
                self._classifier = base_pipe.named_steps["classifier"]
            self._is_calibrated_cv = True
        else:
            raise ValueError(f"Unsupported model object structure: {type(fitted_model)}")

        # Initialize SHAP TreeExplainer on tree estimator
        self._explainer = shap.TreeExplainer(self._classifier)

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

        # Compute model-estimated probability
        proba = float(self._fitted_model.predict_proba(df_in)[0, 1])
        pred_class = int(proba >= threshold)
        pred_label = "Churned" if pred_class == 1 else "Retained"

        # Transform features
        X_trans = self._preprocessor.transform(df_in)
        shap_res = self._explainer(X_trans)

        shap_vals = shap_res.values[0]
        base_val = float(shap_res.base_values[0])
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
            "Local Shapley attributions allocate credit across features for the model's "
            "predicted log-odds margin relative to the expected training baseline. "
            "They quantify model behavior, NOT causal real-world intervention outcomes. "
            "Changing an input feature in reality may not alter churn propensity as estimated."
        )

        return CustomerLocalExplanation(
            raw_input_features=raw_dict,
            model_estimated_probability=round(proba, 4),
            applied_threshold=round(threshold, 4),
            predicted_class=pred_class,
            predicted_label=pred_label,
            base_value_margin=round(base_val, 4),
            model_output_margin=round(model_margin, 4),
            top_churn_drivers=pos_drivers,
            top_retention_drivers=neg_drivers,
            all_attributions=attributions,
            methodological_limitations=limitations,
        )

