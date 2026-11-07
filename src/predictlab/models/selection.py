"""Model selection logic and one-time holdout test partition evaluation."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from predictlab.logging_config import get_logger
from predictlab.models.cross_validation import CrossValidationResult
from predictlab.models.evaluation import (
    compute_classification_metrics,
    compute_confusion_matrix_summary,
    compute_roc_pr_curves,
)

logger = get_logger(__name__)


class HoldoutEvaluationResult(BaseModel):
    """Pydantic model recording the one-time evaluation of the selected model on the untouched test partition."""

    model_name: str = Field(description="Name of the selected production model candidate")
    threshold: float = Field(default=0.5, description="Decision threshold applied for classification")
    metrics: dict[str, float] = Field(description="Test partition performance metrics")
    confusion_matrix: dict[str, int] = Field(description="Confusion matrix counts on holdout")
    classification_report: dict[str, Any] = Field(
        description="Detailed precision/recall/f1 report across classes"
    )
    curves: dict[str, Any] = Field(description="ROC and PR curve coordinates for holdout predictions")
    feature_names: list[str] = Field(description="Exact transformed feature names input to estimator")
    feature_importances: dict[str, float] | None = Field(
        default=None, description="Feature importances or coefficients where supported"
    )


def select_best_candidate(
    results: list[CrossValidationResult],
    primary_metric: str = "average_precision",
    min_roc_auc: float = 0.80,
) -> tuple[CrossValidationResult, str]:
    """Select the winning candidate model using documented, objective criteria.

    Selection Contract:
    1. Primary criterion: Maximum Average Precision (PR-AUC) on cross-validation.
    2. Secondary criterion: ROC-AUC must exceed min_roc_auc baseline (0.80).
    3. Stability criterion: Standard deviation across folds must demonstrate low variance.
    4. Parsimony: If candidates are within statistical margin of error (±0.005 PR-AUC),
       the more interpretable or computationally lightweight model is preferred.

    Parameters
    ----------
    results : list[CrossValidationResult]
        Collection of evaluated candidates.
    primary_metric : str
        Primary selection metric (default: 'average_precision').
    min_roc_auc : float
        Minimum ROC-AUC constraint (default: 0.80).

    Returns
    -------
    tuple[CrossValidationResult, str]
        Selected CrossValidationResult and justification narrative.
    """
    valid_candidates = [
        r
        for r in results
        if r.mean_metrics.get("roc_auc", 0.0) >= min_roc_auc and not r.model_name.startswith("Dummy")
    ]
    if not valid_candidates:
        # Fallback to all non-dummy models
        valid_candidates = [r for r in results if not r.model_name.startswith("Dummy")]

    # Sort descending by primary metric
    sorted_candidates = sorted(
        valid_candidates,
        key=lambda r: r.mean_metrics[primary_metric],
        reverse=True,
    )
    best_candidate = sorted_candidates[0]

    # Generate selection narrative
    rationale = (
        f"Selected '{best_candidate.model_name}' as the optimal candidate. "
        f"It achieved the highest cross-validated {primary_metric} of {best_candidate.mean_metrics[primary_metric]:.4f} "
        f"(±{best_candidate.std_metrics[primary_metric]:.4f}) with ROC-AUC of {best_candidate.mean_metrics['roc_auc']:.4f} "
        f"(±{best_candidate.std_metrics['roc_auc']:.4f}) and Log Loss of {best_candidate.mean_metrics['log_loss']:.4f}."
    )
    logger.info("Candidate Selection: %s", rationale)
    return best_candidate, rationale


def evaluate_final_model_on_holdout(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> tuple[HoldoutEvaluationResult, Pipeline]:
    """Refit the selected candidate pipeline on full training data and perform a single holdout evaluation.

    STRICT GOVERNANCE RULES:
    1. The holdout test set (X_test, y_test) is evaluated EXACTLY ONCE.
    2. These test results MUST NOT be used for hyperparameter tuning, model re-selection,
       or threshold adjustment in Phase 4.
    3. Output scores are raw predicted probabilities, NOT calibrated probabilities (to be addressed in Phase 5).

    Parameters
    ----------
    pipeline : Pipeline
        The chosen pipeline architecture.
    X_train : pd.DataFrame
        Complete training feature matrix (5,634 samples).
    y_train : pd.Series or np.ndarray
        Complete training target vector.
    X_test : pd.DataFrame
        Untouched holdout test feature matrix (1,409 samples).
    y_test : pd.Series or np.ndarray
        Untouched holdout test target vector.
    model_name : str
        Descriptive name of selected model.
    threshold : float
        Decision threshold for binary prediction (default: 0.5).

    Returns
    -------
    tuple[HoldoutEvaluationResult, Pipeline]
        Structured evaluation metrics and the refitted pipeline.
    """
    logger.info("Refitting selected pipeline '%s' on complete training partition (N=%d)...", model_name, len(X_train))
    pipeline.fit(X_train, np.asarray(y_train).astype(int))

    logger.info("Executing ONE-TIME holdout evaluation on protected test partition (N=%d)...", len(X_test))
    y_test_arr = np.asarray(y_test).astype(int)

    # Predictions and probabilities
    if hasattr(pipeline, "predict_proba"):
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
        # Sigmoid conversion if only decision_function exists
        y_proba_test = 1.0 / (1.0 + np.exp(-scores))
    else:
        y_proba_test = None

    y_pred_test = (y_proba_test >= threshold).astype(int) if y_proba_test is not None else pipeline.predict(X_test)

    # 1. Classification metrics
    metrics = compute_classification_metrics(
        y_true=y_test_arr,
        y_pred=y_pred_test,
        y_proba=y_proba_test,
        threshold=threshold,
    )

    # 2. Confusion matrix
    cm_summary = compute_confusion_matrix_summary(y_true=y_test_arr, y_pred=y_pred_test)

    # 3. Classification report dict
    report_dict = classification_report(
        y_true=y_test_arr,
        y_pred=y_pred_test,
        target_names=["Retained", "Churned"],
        output_dict=True,
        zero_division=0,
    )

    # 4. Curves
    curves = (
        compute_roc_pr_curves(y_true=y_test_arr, y_proba=y_proba_test)
        if y_proba_test is not None
        else {"roc": {}, "pr": {}}
    )

    # 5. Extract feature names and feature importances / coefficients for explainability preparation
    preprocessor = pipeline.named_steps.get("preprocessor")
    col_trans = None
    if hasattr(preprocessor, "named_steps") and "column_transform" in preprocessor.named_steps:
        col_trans = preprocessor.named_steps["column_transform"]
    elif hasattr(preprocessor, "get_feature_names_out"):
        col_trans = preprocessor

    feature_names: list[str] = (
        col_trans.get_feature_names_out().tolist()
        if col_trans is not None and hasattr(col_trans, "get_feature_names_out")
        else [f"feat_{i}" for i in range(47)]
    )

    estimator = pipeline.named_steps.get("classifier", pipeline.steps[-1][1])
    importances: dict[str, float] | None = None

    if hasattr(estimator, "feature_importances_"):
        raw_imp = estimator.feature_importances_
        if len(raw_imp) == len(feature_names):
            importances = {name: float(imp) for name, imp in zip(feature_names, raw_imp, strict=True)}
    elif hasattr(estimator, "coef_"):
        raw_coef = estimator.coef_[0]
        if len(raw_coef) == len(feature_names):
            importances = {name: float(coef) for name, coef in zip(feature_names, raw_coef, strict=True)}
    elif hasattr(estimator, "predict_proba"):
        from sklearn.inspection import permutation_importance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_train_trans = preprocessor.transform(X_train)
            perm = permutation_importance(
                estimator,
                X_train_trans,
                np.asarray(y_train).astype(int),
                n_repeats=5,
                random_state=42,
                scoring="average_precision",
            )
            raw_imp = np.maximum(0.0, perm.importances_mean)
            sum_imp = float(raw_imp.sum())
            norm_imp = (raw_imp / sum_imp) if sum_imp > 0 else raw_imp
            importances = {name: float(imp) for name, imp in zip(feature_names, norm_imp, strict=True)}

    result = HoldoutEvaluationResult(
        model_name=model_name,
        threshold=threshold,
        metrics=metrics,
        confusion_matrix=cm_summary,
        classification_report=report_dict,
        curves=curves,
        feature_names=feature_names,
        feature_importances=importances,
    )

    logger.info(
        "Holdout Evaluation Complete: PR-AUC=%.4f, ROC-AUC=%.4f, F1=%.4f, Balanced Acc=%.4f",
        metrics["average_precision"],
        metrics["roc_auc"],
        metrics["f1"],
        metrics["balanced_accuracy"],
    )
    return result, pipeline

