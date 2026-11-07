"""Unit tests for probability calibration metrics, reliability curves, and cross-validation."""

import numpy as np
import pytest

from predictlab.calibration.curves import (
    compute_probability_distribution,
    compute_reliability_curve,
)
from predictlab.calibration.metrics import (
    compute_brier_score_decomposition,
    compute_calibration_intercept_and_slope,
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    evaluate_full_calibration_metrics,
)


def test_ece_uniform_perfect_calibration() -> None:
    """Test that a perfectly calibrated set of predictions yields near-zero ECE."""
    # 100 samples with probability 0.8 where exactly 80 are positive
    # 100 samples with probability 0.2 where exactly 20 are positive
    y_prob = np.array([0.8] * 100 + [0.2] * 100)
    y_true = np.array([1] * 80 + [0] * 20 + [1] * 20 + [0] * 80)

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5, strategy="uniform")
    assert ece == pytest.approx(0.0, abs=1e-5)


def test_ece_quantile_strategy() -> None:
    """Test ECE quantile strategy operates without crashing and returns valid float."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.65, 0.35, 0.85, 0.15])

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=3, strategy="quantile")
    assert 0.0 <= ece <= 1.0


def test_mce_calculation() -> None:
    """Test Maximum Calibration Error identifies peak bin error."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])

    mce = compute_maximum_calibration_error(y_true, y_prob, n_bins=2)
    assert 0.0 <= mce <= 1.0


def test_brier_score_decomposition() -> None:
    """Test Murphy Brier score decomposition components."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    decomp = compute_brier_score_decomposition(y_true, y_prob, n_bins=5)

    assert "brier_score" in decomp
    assert "reliability" in decomp
    assert "resolution" in decomp
    assert "uncertainty" in decomp

    # Uncertainty for base_rate = 0.5 is 0.25
    assert decomp["uncertainty"] == pytest.approx(0.25, abs=1e-5)
    assert decomp["reliability"] >= 0.0
    assert decomp["resolution"] >= 0.0
    assert decomp["brier_score"] >= 0.0


def test_calibration_intercept_and_slope() -> None:
    """Test logistic calibration intercept and slope estimation."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9])

    intercept, slope = compute_calibration_intercept_and_slope(y_true, y_prob)
    assert isinstance(intercept, float)
    assert isinstance(slope, float)
    # Well-separated predictions should have slope > 0
    assert slope > 0.0


def test_evaluate_full_calibration_metrics() -> None:
    """Test full calibration metrics packaging into Pydantic model."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])

    m = evaluate_full_calibration_metrics(y_true, y_prob, n_bins=3)
    assert 0.0 <= m.average_precision <= 1.0
    assert 0.0 <= m.roc_auc <= 1.0
    assert m.log_loss > 0.0
    assert m.brier_score >= 0.0
    assert 0.0 <= m.ece_uniform <= 1.0
    assert 0.0 <= m.ece_quantile <= 1.0


def test_reliability_curve_and_distribution() -> None:
    """Test reliability diagram data generation."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9])

    curve = compute_reliability_curve(y_true, y_prob, n_bins=3)
    assert len(curve["mean_predicted_value"]) > 0
    assert len(curve["fraction_of_positives"]) > 0
    assert len(curve["bin_counts"]) > 0

    dist = compute_probability_distribution(y_prob, y_true=y_true, n_bins=5)
    assert "counts_total" in dist
    assert "counts_retained" in dist
    assert "counts_churned" in dist
    assert sum(dist["counts_total"]) == 5
