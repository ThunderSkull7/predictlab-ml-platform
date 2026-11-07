"""Evaluation module for classification metrics, curves, and confusion matrices."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ClassificationMetrics(BaseModel):
    """Pydantic model encapsulating standard classification performance metrics."""

    average_precision: float = Field(
        description="Average Precision (Area Under Precision-Recall Curve), primary selection metric"
    )
    roc_auc: float = Field(description="Area Under the Receiver Operating Characteristic Curve")
    f1: float = Field(description="F1-Score at evaluation threshold")
    precision: float = Field(description="Precision (Positive Predictive Value) at evaluation threshold")
    recall: float = Field(description="Recall (Sensitivity / True Positive Rate) at evaluation threshold")
    balanced_accuracy: float = Field(description="Balanced Accuracy (unweighted average of class recalls)")
    log_loss: float = Field(description="Cross-entropy / Logarithmic Loss of predicted probabilities")
    brier_score: float = Field(description="Brier Score Loss (mean squared probability error)")
    threshold: float = Field(default=0.5, description="Decision threshold applied to predicted probabilities")


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray | None = None,
    y_proba: pd.Series | np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute comprehensive classification metrics for binary churn prediction.

    Distinguishes threshold-free ranking metrics (Average Precision, ROC-AUC) from
    threshold-dependent classification metrics (F1, Precision, Recall, Balanced Accuracy).

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        Ground truth binary target labels (0 or 1).
    y_pred : pd.Series or np.ndarray, optional
        Predicted binary class labels. If None and y_proba is provided, derived via threshold.
    y_proba : pd.Series or np.ndarray, optional
        Predicted probability of the positive class (Churn=1).
    threshold : float
        Decision threshold used to convert probabilities into predicted classes (default: 0.5).

    Returns
    -------
    dict[str, float]
        Dictionary of calculated metric values.
    """
    y_true_arr = np.asarray(y_true).astype(int)

    if y_pred is None:
        if y_proba is not None:
            y_pred_arr = (np.asarray(y_proba) >= threshold).astype(int)
        else:
            raise ValueError("Either y_pred or y_proba must be provided to compute metrics.")
    else:
        y_pred_arr = np.asarray(y_pred).astype(int)

    # Threshold-dependent metrics
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    # Probability-based ranking and scoring metrics
    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba, dtype=np.float64)
        # Check if single-class probabilities (e.g. DummyClassifier most_frequent)
        try:
            ap = float(average_precision_score(y_true_arr, y_proba_arr))
        except ValueError:
            # Fallback if constant prediction or single class
            ap = float(y_true_arr.mean())

        try:
            auc = float(roc_auc_score(y_true_arr, y_proba_arr))
        except ValueError:
            auc = 0.5

        try:
            # Clip probabilities slightly to avoid log(0)
            eps = 1e-15
            clipped_proba = np.clip(y_proba_arr, eps, 1 - eps)
            ll = float(log_loss(y_true_arr, clipped_proba, labels=[0, 1]))
        except ValueError:
            ll = float(np.nan)

        brier = float(brier_score_loss(y_true_arr, y_proba_arr))
    else:
        # If no probability supplied, fallback to class predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ap = float(average_precision_score(y_true_arr, y_pred_arr))
            auc = float(roc_auc_score(y_true_arr, y_pred_arr))
            ll = float(np.nan)
            brier = float(brier_score_loss(y_true_arr, y_pred_arr))

    return {
        "average_precision": ap,
        "roc_auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": balanced_acc,
        "log_loss": ll,
        "brier_score": brier,
    }


def compute_confusion_matrix_summary(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, int]:
    """Compute structured confusion matrix counts (TN, FP, FN, TP).

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True binary target values.
    y_pred : pd.Series or np.ndarray
        Predicted binary target values.

    Returns
    -------
    dict[str, int]
        Dictionary with counts: true_negatives, false_positives, false_negatives, true_positives.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": int(len(y_true_arr)),
    }


def compute_roc_pr_curves(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Compute ROC and Precision-Recall curve coordinates.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True binary labels.
    y_proba : pd.Series or np.ndarray
        Predicted probabilities for positive class (Churn=1).

    Returns
    -------
    dict[str, Any]
        Dictionary containing curve coordinates for ROC and Precision-Recall.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba, dtype=np.float64)

    fpr, tpr, roc_thresh = roc_curve(y_true_arr, y_proba_arr)
    prec, rec, pr_thresh = precision_recall_curve(y_true_arr, y_proba_arr)

    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
            "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
        },
        "pr": {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": pr_thresh.tolist(),
            "average_precision": float(average_precision_score(y_true_arr, y_proba_arr)),
            "baseline_prevalence": float(y_true_arr.mean()),
        },
    }

