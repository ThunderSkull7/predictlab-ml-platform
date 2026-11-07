"""Leakage-free nested cross-validation for probability calibration comparison."""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from predictlab.calibration.curves import (
    compute_probability_distribution,
    compute_reliability_curve,
)
from predictlab.calibration.metrics import evaluate_full_calibration_metrics
from predictlab.logging_config import get_logger

logger = get_logger(__name__)


class CalibrationFoldMetrics(BaseModel):
    """Metrics for a single outer validation fold."""

    fold_index: int
    metrics: dict[str, float]


class CalibrationEvaluationResult(BaseModel):
    """Encapsulates full out-of-fold calibration evaluation for a single calibration strategy."""

    method_name: str = Field(description="Name of calibration method: 'uncalibrated', 'sigmoid', or 'isotonic'")
    fold_metrics: list[CalibrationFoldMetrics] = Field(description="Evaluation metrics for each outer fold")
    mean_metrics: dict[str, float] = Field(description="Mean metrics across outer validation folds")
    std_metrics: dict[str, float] = Field(description="Standard deviation across outer validation folds")
    overall_metrics: dict[str, float] = Field(description="Metrics computed on pooled out-of-fold predictions")
    reliability_curve_uniform: dict[str, Any] = Field(description="Reliability curve data (uniform bins)")
    reliability_curve_quantile: dict[str, Any] = Field(description="Reliability curve data (quantile bins)")
    probability_distribution: dict[str, Any] = Field(description="Probability distribution histogram data")


def run_leakage_free_calibration_cv(
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_outer_splits: int = 5,
    inner_cv: int = 5,
    random_state: int = 42,
) -> dict[str, tuple[CalibrationEvaluationResult, np.ndarray]]:
    """Execute leakage-free nested cross-validation comparing uncalibrated, sigmoid, and isotonic models.

    Methodological & Leakage Controls:
    1. Outer CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) strictly on training data.
    2. Inner Calibration CV: CalibratedClassifierCV uses inner stratified folds strictly within each
       outer training fold. The outer validation fold is NEVER accessed during fitting or calibration.
    3. Complete Composite Pipeline: Preprocessing (cleaning, service counts, column transformation)
       is encapsulated inside the pipeline and fitted strictly within each inner/outer fold.
    4. Out-of-Fold Predictions: Unbiased out-of-sample probability estimates are collected across all
       training observations for apples-to-apples comparison.

    Parameters
    ----------
    base_pipeline : Pipeline
        Unfitted base scikit-learn Pipeline (preprocessing + HistGradientBoostingClassifier).
    X_train : pd.DataFrame
        Complete training feature matrix (N=5,634).
    y_train : pd.Series or np.ndarray
        Complete training binary target vector.
    n_outer_splits : int, default=5
        Number of outer cross-validation folds.
    inner_cv : int, default=5
        Number of inner cross-validation folds for CalibratedClassifierCV.
    random_state : int, default=42
        Reproducibility seed for outer cross-validation.

    Returns
    -------
    dict[str, tuple[CalibrationEvaluationResult, np.ndarray]]
        Mapping of method name to (CalibrationEvaluationResult, oof_probabilities_array).
    """
    y_train_arr = np.asarray(y_train).astype(int)
    n_samples = len(X_train)

    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)

    methods = ["uncalibrated", "sigmoid", "isotonic"]
    oof_predictions = {m: np.zeros(n_samples, dtype=float) for m in methods}
    fold_metric_records: dict[str, list[CalibrationFoldMetrics]] = {m: [] for m in methods}

    logger.info(
        "Starting leakage-free calibration cross-validation (%d outer folds, %d inner folds)...",
        n_outer_splits,
        inner_cv,
    )

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train_arr), start=1):
        logger.info("Executing Calibration Outer Fold %d/%d...", fold_idx, n_outer_splits)

        X_tr, y_tr = X_train.iloc[train_idx], y_train_arr[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train_arr[val_idx]

        # 1. Uncalibrated Model
        pipe_uncal = clone(base_pipeline)
        pipe_uncal.fit(X_tr, y_tr)
        proba_uncal = pipe_uncal.predict_proba(X_val)[:, 1]
        oof_predictions["uncalibrated"][val_idx] = proba_uncal
        m_uncal = evaluate_full_calibration_metrics(y_val, proba_uncal).model_dump()
        fold_metric_records["uncalibrated"].append(CalibrationFoldMetrics(fold_index=fold_idx, metrics=m_uncal))

        # 2. Sigmoid (Platt Scaling) Calibration
        pipe_sig_base = clone(base_pipeline)
        cal_sig = CalibratedClassifierCV(estimator=pipe_sig_base, method="sigmoid", cv=inner_cv)
        cal_sig.fit(X_tr, y_tr)
        proba_sig = cal_sig.predict_proba(X_val)[:, 1]
        oof_predictions["sigmoid"][val_idx] = proba_sig
        m_sig = evaluate_full_calibration_metrics(y_val, proba_sig).model_dump()
        fold_metric_records["sigmoid"].append(CalibrationFoldMetrics(fold_index=fold_idx, metrics=m_sig))

        # 3. Isotonic Regression Calibration
        pipe_iso_base = clone(base_pipeline)
        cal_iso = CalibratedClassifierCV(estimator=pipe_iso_base, method="isotonic", cv=inner_cv)
        cal_iso.fit(X_tr, y_tr)
        proba_iso = cal_iso.predict_proba(X_val)[:, 1]
        oof_predictions["isotonic"][val_idx] = proba_iso
        m_iso = evaluate_full_calibration_metrics(y_val, proba_iso).model_dump()
        fold_metric_records["isotonic"].append(CalibrationFoldMetrics(fold_index=fold_idx, metrics=m_iso))

    results: dict[str, tuple[CalibrationEvaluationResult, np.ndarray]] = {}

    for m in methods:
        fold_records = fold_metric_records[m]
        metric_keys = list(fold_records[0].metrics.keys())

        mean_metrics = {
            k: float(np.mean([r.metrics[k] for r in fold_records])) for k in metric_keys
        }
        std_metrics = {
            k: float(np.std([r.metrics[k] for r in fold_records], ddof=1)) for k in metric_keys
        }

        # Overall pooled OOF metrics
        overall_metrics = evaluate_full_calibration_metrics(y_train_arr, oof_predictions[m]).model_dump()

        # Curves on pooled OOF predictions
        curve_uniform = compute_reliability_curve(y_train_arr, oof_predictions[m], n_bins=10, strategy="uniform")
        curve_quantile = compute_reliability_curve(y_train_arr, oof_predictions[m], n_bins=10, strategy="quantile")
        dist_data = compute_probability_distribution(oof_predictions[m], y_true=y_train_arr, n_bins=20)

        eval_res = CalibrationEvaluationResult(
            method_name=m,
            fold_metrics=fold_records,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            overall_metrics=overall_metrics,
            reliability_curve_uniform=curve_uniform,
            reliability_curve_quantile=curve_quantile,
            probability_distribution=dist_data,
        )

        results[m] = (eval_res, oof_predictions[m])

        logger.info(
            "Calibration Result [%s]: Brier=%.4f (±%.4f), LogLoss=%.4f (±%.4f), ECE_unif=%.4f (±%.4f), PR-AUC=%.4f",
            m,
            mean_metrics["brier_score"],
            std_metrics["brier_score"],
            mean_metrics["log_loss"],
            std_metrics["log_loss"],
            mean_metrics["ece_uniform"],
            mean_metrics["average_precision"],
        )

    return results

