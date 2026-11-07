"""Experiment runner for candidate model benchmarking and feature ablation studies."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from predictlab.logging_config import get_logger
from predictlab.models.cross_validation import (
    CrossValidationResult,
    cv_results_to_dataframe,
    evaluate_pipeline_cv,
)
from predictlab.models.estimators import (
    build_model_pipeline,
    get_candidate_models,
)
from predictlab.preprocessing.pipelines import (
    create_experimental_preprocessor,
    create_tree_preprocessor,
)

logger = get_logger(__name__)


def run_model_benchmarking(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    candidate_models: dict[str, Pipeline] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[CrossValidationResult], pd.DataFrame]:
    """Execute stratified cross-validation across all candidate model families.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target vector.
    candidate_models : dict[str, Pipeline], optional
        Dictionary mapping model names to configured Pipelines. Defaults to get_candidate_models().
    n_splits : int
        Number of CV folds (default: 5).
    random_state : int
        Deterministic random seed (default: 42).

    Returns
    -------
    tuple[list[CrossValidationResult], pd.DataFrame]
        List of detailed CV results and ranked summary comparison DataFrame.
    """
    models = candidate_models if candidate_models is not None else get_candidate_models(random_state)
    results: list[CrossValidationResult] = []

    logger.info("Executing benchmark across %d candidate model configurations...", len(models))

    for name, pipeline in models.items():
        res = evaluate_pipeline_cv(
            pipeline=pipeline,
            X=X_train,
            y=y_train,
            model_name=name,
            n_splits=n_splits,
            random_state=random_state,
        )
        results.append(res)

    df_summary = cv_results_to_dataframe(results)
    return results, df_summary


def run_feature_ablation_study(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    estimator_factory: Callable[[], BaseEstimator] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[CrossValidationResult], pd.DataFrame]:
    """Execute controlled feature ablation comparing 47 baseline features with 52 experimental features.

    Evaluates:
    1. Default Baseline (47 features)
    2. Full Experimental Pipeline (52 features)
    3. Ablation: Exclude ChargesRatio & Defined Indicator (50 features)
    4. Ablation: Exclude AutomaticPayment (51 features)
    5. Ablation: Exclude Service Subtotals (TechnicalProtection & StreamingEntertainment) (50 features)

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target vector.
    estimator_factory : callable, optional
        Function returning a fresh instance of the estimator to test (defaults to HistGradientBoosting).
    n_splits : int
        Number of CV folds (default: 5).
    random_state : int
        Deterministic random seed (default: 42).

    Returns
    -------
    tuple[list[CrossValidationResult], pd.DataFrame]
        List of CV results and comparative delta summary DataFrame.
    """
    if estimator_factory is None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        def estimator_factory() -> BaseEstimator:
            return HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=random_state,
            )

    # Define ablation configurations
    ablation_configs: list[tuple[str, Pipeline, int]] = [
        (
            "Baseline (47 features)",
            create_tree_preprocessor(),
            47,
        ),
        (
            "Full Experimental (52 features)",
            create_experimental_preprocessor(
                include_charges_ratio=True,
                include_automatic_payment=True,
                include_service_subtotals=True,
                scale_numeric=False,
            ),
            52,
        ),
        (
            "Ablate ChargesRatio (50 features)",
            create_experimental_preprocessor(
                include_charges_ratio=False,
                include_automatic_payment=True,
                include_service_subtotals=True,
                scale_numeric=False,
            ),
            50,
        ),
        (
            "Ablate AutomaticPayment (51 features)",
            create_experimental_preprocessor(
                include_charges_ratio=True,
                include_automatic_payment=False,
                include_service_subtotals=True,
                scale_numeric=False,
            ),
            51,
        ),
        (
            "Ablate Service Subtotals (50 features)",
            create_experimental_preprocessor(
                include_charges_ratio=True,
                include_automatic_payment=True,
                include_service_subtotals=False,
                scale_numeric=False,
            ),
            50,
        ),
    ]

    results: list[CrossValidationResult] = []
    logger.info("Executing feature ablation study across %d pipeline configurations...", len(ablation_configs))

    for config_name, preprocessor, _expected_features in ablation_configs:
        pipeline = build_model_pipeline(preprocessor, estimator_factory())
        res = evaluate_pipeline_cv(
            pipeline=pipeline,
            X=X_train,
            y=y_train,
            model_name=config_name,
            n_splits=n_splits,
            random_state=random_state,
        )
        results.append(res)

    # Compute delta table against baseline
    baseline_res = results[0]
    base_ap = baseline_res.mean_metrics["average_precision"]
    base_roc = baseline_res.mean_metrics["roc_auc"]
    base_f1 = baseline_res.mean_metrics["f1"]
    base_ll = baseline_res.mean_metrics["log_loss"]

    rows: list[dict[str, Any]] = []
    for r in results:
        ap = r.mean_metrics["average_precision"]
        roc = r.mean_metrics["roc_auc"]
        f1 = r.mean_metrics["f1"]
        ll = r.mean_metrics["log_loss"]

        delta_ap = ap - base_ap
        delta_roc = roc - base_roc
        delta_f1 = f1 - base_f1
        delta_ll = ll - base_ll

        rows.append(
            {
                "Configuration": r.model_name,
                "PR-AUC (Mean)": round(ap, 4),
                "PR-AUC (Std)": round(r.std_metrics["average_precision"], 4),
                "Δ PR-AUC": round(delta_ap, 4),
                "ROC-AUC (Mean)": round(roc, 4),
                "Δ ROC-AUC": round(delta_roc, 4),
                "F1 (Mean)": round(f1, 4),
                "Δ F1": round(delta_f1, 4),
                "Log Loss (Mean)": round(ll, 4),
                "Δ Log Loss": round(delta_ll, 4),
                "Fit Time (s)": round(r.mean_fit_time, 3),
            }
        )

    df_deltas = pd.DataFrame(rows)
    return results, df_deltas
