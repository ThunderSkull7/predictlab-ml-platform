"""Cross-validation module enforcing strict leakage-free fold evaluation."""

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from predictlab.logging_config import get_logger
from predictlab.models.evaluation import compute_classification_metrics

logger = get_logger(__name__)


class CrossValidationResult(BaseModel):
    """Encapsulates full fold-level and summary cross-validation performance metrics."""

    model_name: str = Field(description="Identifier of evaluated candidate model")
    n_splits: int = Field(default=5, description="Number of cross-validation folds")
    fold_metrics: list[dict[str, float]] = Field(description="Individual metric scores for each fold")
    mean_metrics: dict[str, float] = Field(description="Mean metric score across all folds")
    std_metrics: dict[str, float] = Field(description="Standard deviation across all folds")
    mean_fit_time: float = Field(description="Average fold training duration in seconds")
    mean_score_time: float = Field(description="Average fold prediction duration in seconds")
    warnings_encountered: list[str] = Field(
        default_factory=list, description="Warnings captured during cross-validation"
    )


def evaluate_pipeline_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model_name: str = "CandidateModel",
    n_splits: int = 5,
    random_state: int = 42,
    threshold: float = 0.5,
) -> CrossValidationResult:
    """Execute leakage-free stratified cross-validation on a scikit-learn Pipeline.

    Enforces critical methodological controls:
    1. Preprocessing and classifier are fitted strictly within each training fold.
    2. Validation folds are transformed using the parameters learned on the training fold.
    3. The holdout test set is completely excluded from this procedure.
    4. Stratification preserves target class balance across all folds.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted scikit-learn Pipeline containing both preprocessing and estimator steps.
    X : pd.DataFrame
        Training feature matrix (X_train).
    y : pd.Series or np.ndarray
        Training target vector (y_train).
    model_name : str
        Human-readable name of model.
    n_splits : int
        Number of cross-validation folds (default: 5).
    random_state : int
        Deterministic random seed for StratifiedKFold shuffling (default: 42).
    threshold : float
        Decision threshold for class prediction (default: 0.5).

    Returns
    -------
    CrossValidationResult
        Detailed record of fold-level and aggregate metrics.
    """
    y_arr = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics: list[dict[str, float]] = []
    fit_times: list[float] = []
    score_times: list[float] = []
    captured_warnings: list[str] = []

    logger.info("Starting %d-fold stratified CV for '%s'...", n_splits, model_name)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_arr)):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y_arr[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y_arr[val_idx]

        # 1. Fit fold pipeline with warning capture
        t_fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as fold_warns:
            warnings.simplefilter("always")
            pipeline.fit(X_fold_train, y_fold_train)
            for w in fold_warns:
                msg = f"Fold {fold_idx + 1}: {w.category.__name__}: {w.message}"
                if msg not in captured_warnings:
                    captured_warnings.append(msg)
        t_fit_end = time.perf_counter()
        fit_times.append(t_fit_end - t_fit_start)

        # 2. Score fold validation data
        t_score_start = time.perf_counter()
        if hasattr(pipeline, "predict_proba"):
            y_proba_val = pipeline.predict_proba(X_fold_val)[:, 1]
            y_pred_val = (y_proba_val >= threshold).astype(int)
        else:
            y_proba_val = None
            y_pred_val = pipeline.predict(X_fold_val)
        t_score_end = time.perf_counter()
        score_times.append(t_score_end - t_score_start)

        # 3. Compute metrics for fold
        metrics = compute_classification_metrics(
            y_true=y_fold_val,
            y_pred=y_pred_val,
            y_proba=y_proba_val,
            threshold=threshold,
        )
        fold_metrics.append(metrics)

    # Aggregate metric statistics
    metric_keys = list(fold_metrics[0].keys())
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in metric_keys}
    std_metrics = {k: float(np.std([m[k] for m in fold_metrics], ddof=1)) for k in metric_keys}

    logger.info(
        "Finished CV for '%s': Mean PR-AUC=%.4f (±%.4f), ROC-AUC=%.4f (±%.4f)",
        model_name,
        mean_metrics["average_precision"],
        std_metrics["average_precision"],
        mean_metrics["roc_auc"],
        std_metrics["roc_auc"],
    )

    return CrossValidationResult(
        model_name=model_name,
        n_splits=n_splits,
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        mean_fit_time=float(np.mean(fit_times)),
        mean_score_time=float(np.mean(score_times)),
        warnings_encountered=captured_warnings,
    )


def cv_results_to_dataframe(results: list[CrossValidationResult]) -> pd.DataFrame:
    """Format a collection of CrossValidationResult objects into a structured comparison DataFrame.

    Parameters
    ----------
    results : list[CrossValidationResult]
        List of cross-validation results.

    Returns
    -------
    pd.DataFrame
        Tabular summary ranking models by Average Precision (PR-AUC).
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {"Model": r.model_name}
        for metric, mean_val in r.mean_metrics.items():
            if metric == "threshold":
                continue
            std_val = r.std_metrics.get(metric, 0.0)
            row[f"{metric}_mean"] = round(mean_val, 4)
            row[f"{metric}_std"] = round(std_val, 4)
            row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
        row["mean_fit_time_s"] = round(r.mean_fit_time, 3)
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    if "average_precision_mean" in df_summary.columns:
        df_summary = df_summary.sort_values(by="average_precision_mean", ascending=False).reset_index(
            drop=True
        )
    return df_summary

