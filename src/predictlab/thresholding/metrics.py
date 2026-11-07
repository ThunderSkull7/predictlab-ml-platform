"""Threshold-dependent classification metrics evaluation and grid generation."""

import numpy as np
import pandas as pd


def evaluate_threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate classification metrics across a fine grid of decision thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True binary targets (0 or 1).
    y_prob : np.ndarray
        Calibrated model-estimated positive-class probabilities.
    thresholds : array-like, optional
        Grid of thresholds in (0, 1). Defaults to 81 values from 0.10 to 0.90 (step 0.01).

    Returns
    -------
    pd.DataFrame
        Table with metrics for each threshold:
        - 'threshold'
        - 'precision', 'recall', 'specificity', 'f1', 'f2', 'f0_5', 'balanced_accuracy'
        - 'true_positives', 'false_positives', 'true_negatives', 'false_negatives'
        - 'predicted_positive_rate'
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    n_total = len(y_true_arr)

    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 81)
    else:
        thresholds = np.asarray(thresholds)

    actual_pos = int(np.sum(y_true_arr == 1))
    actual_neg = int(np.sum(y_true_arr == 0))

    rows = []
    for t in thresholds:
        t_val = float(t)
        y_pred = (y_prob_arr >= t_val).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true_arr == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true_arr == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true_arr == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true_arr == 1)))

        # Precision & Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / actual_pos if actual_pos > 0 else 0.0
        specificity = tn / actual_neg if actual_neg > 0 else 0.0

        # F-beta scores
        # F1 (beta = 1.0)
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # F2 (beta = 2.0, weights recall twice as much as precision)
        beta_2 = 2.0
        f2 = (
            (1.0 + beta_2**2) * (precision * recall) / ((beta_2**2 * precision) + recall)
            if ((beta_2**2 * precision) + recall) > 0
            else 0.0
        )

        # F0.5 (beta = 0.5, weights precision twice as much as recall)
        beta_05 = 0.5
        f0_5 = (
            (1.0 + beta_05**2) * (precision * recall) / ((beta_05**2 * precision) + recall)
            if ((beta_05**2 * precision) + recall) > 0
            else 0.0
        )

        balanced_acc = 0.5 * (recall + specificity)
        pred_pos_rate = (tp + fp) / n_total if n_total > 0 else 0.0

        rows.append(
            {
                "threshold": round(t_val, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "specificity": round(specificity, 4),
                "f1": round(f1, 4),
                "f2": round(f2, 4),
                "f0_5": round(f0_5, 4),
                "balanced_accuracy": round(balanced_acc, 4),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "predicted_positive_rate": round(pred_pos_rate, 4),
            }
        )

    return pd.DataFrame(rows)

