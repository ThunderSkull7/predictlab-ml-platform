"""Threshold optimization and selection optimism quantification under nested validation."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.model_selection import StratifiedKFold

from predictlab.logging_config import get_logger
from predictlab.thresholding.costs import CostMatrix, evaluate_cost_sensitive_thresholds
from predictlab.thresholding.metrics import evaluate_threshold_grid

logger = get_logger(__name__)


class ThresholdPolicyResult(BaseModel):
    """Encapsulates selected decision threshold and its policy evaluation metrics."""

    objective: str = Field(description="Optimization objective ('cost', 'f1', 'f2', 'balanced_accuracy', 'default')")
    selected_threshold: float = Field(description="Selected operating decision threshold")
    in_sample_metrics: dict[str, float] = Field(description="Performance on training data used for selection")
    out_of_fold_metrics: dict[str, float] = Field(description="Honest nested out-of-fold evaluated performance")
    optimism_delta: dict[str, float] = Field(description="In-sample minus out-of-fold metric differences")
    selection_notes: str = Field(description="Methodological explanation and limitations")


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: Literal["cost", "f1", "f2", "balanced_accuracy", "default"] = "cost",
    cost_matrix: CostMatrix | None = None,
    threshold_grid: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the threshold optimizing a specific objective on given predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    objective : {'cost', 'f1', 'f2', 'balanced_accuracy', 'default'}, default='cost'
        Optimization target.
    cost_matrix : CostMatrix, optional
        Required if objective == 'cost'.
    threshold_grid : np.ndarray, optional
        Threshold candidate values.

    Returns
    -------
    tuple[float, dict[str, float]]
        (optimal_threshold, metric_dictionary_at_optimal)
    """
    if objective == "default":
        df = evaluate_threshold_grid(y_true, y_prob, thresholds=[0.50])
        row = df.iloc[0].to_dict()
        return 0.50, row

    df = evaluate_threshold_grid(y_true, y_prob, thresholds=threshold_grid)

    if objective == "cost":
        if cost_matrix is None:
            raise ValueError("cost_matrix must be provided when objective='cost'.")
        df_eval = evaluate_cost_sensitive_thresholds(df, cost_matrix)
        best_row = df_eval.sort_values(by="total_cost", ascending=True).iloc[0]
    elif objective in ["f1", "f2", "balanced_accuracy"]:
        best_row = df.sort_values(by=objective, ascending=False).iloc[0]
    else:
        raise ValueError(f"Unknown objective: {objective}")

    best_thresh = float(best_row["threshold"])
    return best_thresh, best_row.to_dict()


def evaluate_threshold_policy_nested(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: Literal["cost", "f1", "f2", "balanced_accuracy", "default"] = "cost",
    cost_matrix: CostMatrix | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> ThresholdPolicyResult:
    """Evaluate threshold selection policy using nested cross-validation to quantify selection optimism.

    Methodological Warning on Threshold Optimism:
    Evaluating a selected threshold on the identical dataset partition used to search for it
    yields optimistically biased performance estimates. Nested cross-validation isolates
    threshold search to inner training splits and evaluates the policy out-of-fold.

    Parameters
    ----------
    y_true : np.ndarray
        Training labels.
    y_prob : np.ndarray
        Training cross-fitted probabilities.
    objective : str
        Target optimization criterion.
    cost_matrix : CostMatrix, optional
        Cost specifications.
    n_splits : int, default=5
        Number of cross-validation splits.
    random_state : int, default=42
        Seed for split reproducibility.

    Returns
    -------
    ThresholdPolicyResult
        Full audit of in-sample vs out-of-fold performance.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    # 1. Direct in-sample optimization across all training predictions
    in_sample_thresh, in_sample_row = optimize_threshold(
        y_true_arr, y_prob_arr, objective=objective, cost_matrix=cost_matrix
    )

    # 2. Nested out-of-fold policy evaluation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros_like(y_true_arr)
    fold_thresholds = []

    for train_idx, val_idx in cv.split(y_prob_arr, y_true_arr):
        y_tr, p_tr = y_true_arr[train_idx], y_prob_arr[train_idx]
        p_va = y_prob_arr[val_idx]

        fold_th, _ = optimize_threshold(y_tr, p_tr, objective=objective, cost_matrix=cost_matrix)
        fold_thresholds.append(fold_th)
        oof_preds[val_idx] = (p_va >= fold_th).astype(int)

    # Compute out-of-fold confusion matrix and metrics
    tp = int(np.sum((oof_preds == 1) & (y_true_arr == 1)))
    fp = int(np.sum((oof_preds == 1) & (y_true_arr == 0)))
    tn = int(np.sum((oof_preds == 0) & (y_true_arr == 0)))
    fn = int(np.sum((oof_preds == 0) & (y_true_arr == 1)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = (5.0 * prec * rec) / ((4.0 * prec) + rec) if ((4.0 * prec) + rec) > 0 else 0.0
    bacc = 0.5 * (rec + spec)

    oof_metrics = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "balanced_accuracy": round(bacc, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "mean_fold_threshold": round(float(np.mean(fold_thresholds)), 4),
    }

    if objective == "cost" and cost_matrix is not None:
        total_c = cost_matrix.compute_total_cost(tp, fp, tn, fn)
        oof_metrics["total_cost"] = round(total_c, 2)
        oof_metrics["cost_per_customer"] = round(total_c / len(y_true_arr), 4)

    # Calculate selection optimism
    optimism: dict[str, float] = {}
    for k in ["precision", "recall", "f1", "f2", "balanced_accuracy"]:
        if k in in_sample_row and k in oof_metrics:
            optimism[k] = round(in_sample_row[k] - oof_metrics[k], 4)

    notes = (
        f"Selected threshold {in_sample_thresh:.4f} under objective '{objective}'. "
        f"Mean fold threshold was {np.mean(fold_thresholds):.4f} (std={np.std(fold_thresholds):.4f}). "
        f"Out-of-fold F1={f1:.4f} (optimism delta={optimism.get('f1', 0.0):+.4f})."
    )

    logger.info("Threshold Policy Evaluation: %s", notes)

    return ThresholdPolicyResult(
        objective=objective,
        selected_threshold=in_sample_thresh,
        in_sample_metrics=in_sample_row,
        out_of_fold_metrics=oof_metrics,
        optimism_delta=optimism,
        selection_notes=notes,
    )

