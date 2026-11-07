"""Unit tests for classification evaluation metrics, confusion matrices, and curves."""

import numpy as np
import pytest

from predictlab.models.evaluation import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_confusion_matrix_summary,
    compute_roc_pr_curves,
)


def test_classification_metrics_perfect_prediction() -> None:
    """Test metric calculations with perfect predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.9, 0.95])

    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba)

    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["brier_score"] < 0.05


def test_classification_metrics_known_imbalanced_case() -> None:
    """Test metric calculation on known imbalanced scenario."""
    # 7 negative, 3 positive
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    # Predictions: 2 true positive, 1 false negative, 1 false positive, 6 true negative
    y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.1, 0.15, 0.3, 0.25, 0.6, 0.4, 0.8, 0.9])

    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba)

    # Precision = 2 / (2 + 1) = 2/3
    assert pytest.approx(metrics["precision"], abs=1e-3) == 2 / 3
    # Recall = 2 / 3
    assert pytest.approx(metrics["recall"], abs=1e-3) == 2 / 3
    # F1 = 2/3
    assert pytest.approx(metrics["f1"], abs=1e-3) == 2 / 3
    # Balanced accuracy = (6/7 + 2/3) / 2
    expected_bacc = ((6 / 7) + (2 / 3)) / 2
    assert pytest.approx(metrics["balanced_accuracy"], abs=1e-3) == expected_bacc
    assert 0.0 < metrics["average_precision"] <= 1.0
    assert 0.0 < metrics["roc_auc"] <= 1.0


def test_confusion_matrix_summary() -> None:
    """Test structured confusion matrix breakdown."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1])

    cm = compute_confusion_matrix_summary(y_true, y_pred)
    assert cm["true_negatives"] == 2
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_positives"] == 1
    assert cm["total_samples"] == 5


def test_roc_pr_curves_output_structure() -> None:
    """Test coordinates and metadata returned by compute_roc_pr_curves."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.9])

    curves = compute_roc_pr_curves(y_true, y_proba)

    assert "roc" in curves and "pr" in curves
    assert len(curves["roc"]["fpr"]) > 0
    assert len(curves["roc"]["tpr"]) > 0
    assert curves["roc"]["roc_auc"] == 1.0

    assert len(curves["pr"]["precision"]) > 0
    assert len(curves["pr"]["recall"]) > 0
    assert curves["pr"]["average_precision"] == 1.0
    assert curves["pr"]["baseline_prevalence"] == 0.5


def test_pydantic_classification_metrics_validation() -> None:
    """Test Pydantic model validation for metrics."""
    metrics_dict = compute_classification_metrics(
        y_true=np.array([0, 1]), y_pred=np.array([0, 1]), y_proba=np.array([0.2, 0.8])
    )
    validated = ClassificationMetrics(**metrics_dict)
    assert validated.average_precision == 1.0
    assert validated.f1 == 1.0

