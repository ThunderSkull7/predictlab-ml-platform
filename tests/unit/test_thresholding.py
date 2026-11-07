"""Unit tests for decision threshold analysis, cost-sensitive optimization, and sensitivity."""

import numpy as np

from predictlab.thresholding.costs import (
    CostMatrix,
    evaluate_cost_sensitive_thresholds,
    run_cost_ratio_sensitivity,
)
from predictlab.thresholding.metrics import evaluate_threshold_grid
from predictlab.thresholding.optimization import (
    evaluate_threshold_policy_nested,
    optimize_threshold,
)


def test_threshold_grid_generation() -> None:
    """Test generating precision, recall, and F-scores across threshold grid."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    df = evaluate_threshold_grid(y_true, y_prob, thresholds=[0.25, 0.50, 0.75])
    assert len(df) == 3
    assert "precision" in df.columns
    assert "recall" in df.columns
    assert "f1" in df.columns
    assert "f2" in df.columns
    assert "f0_5" in df.columns
    assert "balanced_accuracy" in df.columns

    # At threshold 0.50, perfect separation
    row_50 = df[df["threshold"] == 0.50].iloc[0]
    assert row_50["precision"] == 1.0
    assert row_50["recall"] == 1.0
    assert row_50["f1"] == 1.0


def test_cost_matrix_calculation() -> None:
    """Test CostMatrix calculation with custom asymmetric costs."""
    cm = CostMatrix(cost_fp=2.0, cost_fn=10.0, benefit_tp=5.0)

    # 10 TP, 5 FP, 20 TN, 2 FN
    # Cost = (5 * 2.0) + (2 * 10.0) - (10 * 5.0) = 10 + 20 - 50 = -20.0
    cost = cm.compute_total_cost(tp=10, fp=5, tn=20, fn=2)
    assert cost == -20.0


def test_cost_sensitive_threshold_evaluation() -> None:
    """Test evaluating threshold table under cost matrix."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.4, 0.6, 0.8])

    df_grid = evaluate_threshold_grid(y_true, y_prob, thresholds=[0.3, 0.5, 0.7])
    cm = CostMatrix(cost_fp=1.0, cost_fn=5.0)
    df_cost = evaluate_cost_sensitive_thresholds(df_grid, cm)

    assert "total_cost" in df_cost.columns
    assert "cost_per_customer" in df_cost.columns
    assert "cost_savings_vs_default" in df_cost.columns


def test_cost_ratio_sensitivity() -> None:
    """Test sensitivity analysis across multiple hypothetical cost ratios."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8])

    df_grid = evaluate_threshold_grid(y_true, y_prob, thresholds=np.linspace(0.1, 0.9, 9))
    df_sens = run_cost_ratio_sensitivity(df_grid, cost_ratios=[1.0, 5.0, 10.0])

    assert len(df_sens) == 3
    assert "cost_ratio_fn_to_fp" in df_sens.columns
    assert "optimal_threshold" in df_sens.columns
    # As FN cost increases, optimal threshold should decrease or stay same (higher recall)
    thresh_1 = df_sens[df_sens["cost_ratio_fn_to_fp"] == "1.0:1"]["optimal_threshold"].iloc[0]
    thresh_10 = df_sens[df_sens["cost_ratio_fn_to_fp"] == "10.0:1"]["optimal_threshold"].iloc[0]
    assert thresh_10 <= thresh_1


def test_optimize_threshold_objectives() -> None:
    """Test optimize_threshold across f1, balanced_accuracy, and default."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.9])

    th_f1, row_f1 = optimize_threshold(y_true, y_prob, objective="f1")
    assert 0.0 < th_f1 < 1.0

    th_def, row_def = optimize_threshold(y_true, y_prob, objective="default")
    assert th_def == 0.50


def test_nested_threshold_policy_evaluation() -> None:
    """Test nested cross-validation of threshold policy to measure selection optimism."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.25, 0.3, 0.45, 0.55, 0.7, 0.8, 0.85, 0.95])

    res = evaluate_threshold_policy_nested(
        y_true=y_true,
        y_prob=y_prob,
        objective="f1",
        n_splits=2,
        random_state=42,
    )

    assert 0.0 < res.selected_threshold < 1.0
    assert "f1" in res.in_sample_metrics
    assert "f1" in res.out_of_fold_metrics
    assert "f1" in res.optimism_delta
