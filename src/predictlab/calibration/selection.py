"""Selection and final training fitting of optimal probability calibration method."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from predictlab.calibration.cross_validation import CalibrationEvaluationResult
from predictlab.logging_config import get_logger

logger = get_logger(__name__)


def select_best_calibration_method(
    results: dict[str, CalibrationEvaluationResult],
) -> tuple[str, str, dict[str, Any]]:
    """Select the optimal probability calibration method using training-only CV evidence.

    Selection Principles:
    1. Primary Criterion: Minimization of proper probabilistic scoring losses (Brier Score and Log Loss).
    2. Reliability Criterion: Minimization of Expected Calibration Error (ECE) and Brier Reliability.
    3. Discrimination Preservation: Ranking capacity (PR-AUC and ROC-AUC) must not be degraded.
    4. Parsimony & Stability: Avoid non-parametric overfitting (e.g. isotonic step artifacts) if
       parametric sigmoid achieves comparable or superior loss with lower variance.

    Parameters
    ----------
    results : dict[str, CalibrationEvaluationResult]
        Out-of-fold calibration evaluation results for 'uncalibrated', 'sigmoid', and 'isotonic'.

    Returns
    -------
    tuple[str, str, dict[str, Any]]
        (selected_method_name, selection_rationale_text, comparison_summary_dict)
    """
    summary: dict[str, dict[str, float]] = {}
    for name, res in results.items():
        summary[name] = {
            "brier_score": res.mean_metrics["brier_score"],
            "brier_std": res.std_metrics["brier_score"],
            "log_loss": res.mean_metrics["log_loss"],
            "log_loss_std": res.std_metrics["log_loss"],
            "ece_uniform": res.mean_metrics["ece_uniform"],
            "ece_quantile": res.mean_metrics["ece_quantile"],
            "brier_reliability": res.mean_metrics["brier_reliability"],
            "average_precision": res.mean_metrics["average_precision"],
            "roc_auc": res.mean_metrics["roc_auc"],
        }

    # Evaluate candidate scores
    brier_scores = {k: v["brier_score"] for k, v in summary.items()}
    log_losses = {k: v["log_loss"] for k, v in summary.items()}
    eces = {k: v["ece_uniform"] for k, v in summary.items()}

    # Rank candidates by combined score improvement
    # Sigmoid vs Isotonic vs Uncalibrated
    uncal_brier = brier_scores["uncalibrated"]
    uncal_ll = log_losses["uncalibrated"]
    uncal_ece = eces["uncalibrated"]

    best_method = min(brier_scores, key=brier_scores.get)

    # If differences are negligible (< 0.001 Brier), compare log_loss
    if abs(brier_scores["sigmoid"] - brier_scores["isotonic"]) < 0.001:
        if log_losses["sigmoid"] <= log_losses["isotonic"]:
            best_method = "sigmoid"
        else:
            best_method = "isotonic"

    # Check whether calibration actually improved over uncalibrated
    improved = (brier_scores[best_method] < uncal_brier) or (log_losses[best_method] < uncal_ll)

    if not improved:
        selected_method = "uncalibrated"
        rationale = (
            f"Retained 'uncalibrated' model. Neither Sigmoid (Brier={brier_scores['sigmoid']:.4f}) "
            f"nor Isotonic (Brier={brier_scores['isotonic']:.4f}) achieved meaningful improvement "
            f"over uncalibrated baseline (Brier={uncal_brier:.4f}, LogLoss={uncal_ll:.4f})."
        )
    else:
        selected_method = best_method
        delta_brier = uncal_brier - brier_scores[selected_method]
        delta_ll = uncal_ll - log_losses[selected_method]
        delta_ece = uncal_ece - eces[selected_method]
        rationale = (
            f"Selected '{selected_method}' calibration. It achieved superior proper scoring and reliability: "
            f"Brier Score={brier_scores[selected_method]:.4f} (Δ={-delta_brier:+.4f}), "
            f"Log Loss={log_losses[selected_method]:.4f} (Δ={-delta_ll:+.4f}), "
            f"ECE={eces[selected_method]:.4f} (Δ={-delta_ece:+.4f}), while preserving ranking discrimination "
            f"(PR-AUC={summary[selected_method]['average_precision']:.4f}, ROC-AUC={summary[selected_method]['roc_auc']:.4f})."
        )

    logger.info("Calibration Selection Decision: %s", rationale)
    return selected_method, rationale, summary


def fit_final_calibrated_pipeline(
    base_pipeline: Pipeline,
    selected_method: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    inner_cv: int = 5,
) -> Any:
    """Fit the chosen calibration architecture on the complete training partition.

    Parameters
    ----------
    base_pipeline : Pipeline
        The unfitted production base pipeline (preprocessing + classifier).
    selected_method : str
        'uncalibrated', 'sigmoid', or 'isotonic'.
    X_train : pd.DataFrame
        Complete training feature set (N=5,634).
    y_train : pd.Series or np.ndarray
        Complete training binary targets.
    inner_cv : int, default=5
        Inner cross-validation folds for CalibratedClassifierCV.

    Returns
    -------
    Pipeline or CalibratedClassifierCV
        Fitted model object ready for production inference.
    """
    y_arr = np.asarray(y_train).astype(int)

    if selected_method == "uncalibrated":
        logger.info("Fitting uncalibrated pipeline on full training data (N=%d)...", len(X_train))
        pipeline = clone(base_pipeline)
        pipeline.fit(X_train, y_arr)
        return pipeline

    logger.info(
        "Fitting CalibratedClassifierCV (method='%s', cv=%d) on full training data (N=%d)...",
        selected_method,
        inner_cv,
        len(X_train),
    )
    calibrated_clf = CalibratedClassifierCV(
        estimator=clone(base_pipeline),
        method=selected_method,
        cv=inner_cv,
    )
    calibrated_clf.fit(X_train, y_arr)
    return calibrated_clf

