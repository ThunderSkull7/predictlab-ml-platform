"""Probability calibration module for PredictLab."""

from predictlab.calibration.cross_validation import (
    CalibrationEvaluationResult,
    CalibrationFoldMetrics,
    run_leakage_free_calibration_cv,
)
from predictlab.calibration.curves import (
    compute_probability_distribution,
    compute_reliability_curve,
)
from predictlab.calibration.metrics import (
    CalibrationMetrics,
    compute_brier_score_decomposition,
    compute_calibration_intercept_and_slope,
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    evaluate_full_calibration_metrics,
)
from predictlab.calibration.selection import (
    fit_final_calibrated_pipeline,
    select_best_calibration_method,
)

__all__ = [
    "CalibrationMetrics",
    "CalibrationFoldMetrics",
    "CalibrationEvaluationResult",
    "compute_expected_calibration_error",
    "compute_maximum_calibration_error",
    "compute_brier_score_decomposition",
    "compute_calibration_intercept_and_slope",
    "evaluate_full_calibration_metrics",
    "compute_reliability_curve",
    "compute_probability_distribution",
    "run_leakage_free_calibration_cv",
    "select_best_calibration_method",
    "fit_final_calibrated_pipeline",
]

