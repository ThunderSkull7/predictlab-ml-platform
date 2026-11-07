"""Controlled hyperparameter tuning using training-only stratified cross-validation."""

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from predictlab.logging_config import get_logger
from predictlab.models.estimators import (
    create_hist_gradient_boosting_pipeline,
    create_logistic_regression_pipeline,
)

logger = get_logger(__name__)


class TuningResult(BaseModel):
    """Pydantic model recording hyperparameter search metadata and performance."""

    model_name: str = Field(description="Name of tuned model family")
    best_params: dict[str, Any] = Field(description="Optimal hyperparameters discovered")
    best_score: float = Field(description="Cross-validated mean score on primary tuning metric")
    scoring_metric: str = Field(description="Scoring metric used for optimization")
    search_space: dict[str, Any] = Field(description="Hyperparameter space searched")
    search_type: str = Field(description="Search algorithm ('grid' or 'randomized')")
    n_candidates_evaluated: int = Field(description="Number of parameter configurations evaluated")
    search_duration_s: float = Field(description="Total tuning duration in seconds")
    cv_summary_table: list[dict[str, Any]] = Field(
        default_factory=list, description="Top parameter configurations and scores"
    )


def tune_pipeline_hyperparameters(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    model_name: str = "TunedModel",
    scoring: str = "average_precision",
    n_splits: int = 5,
    search_type: Literal["grid", "randomized"] = "grid",
    n_iter: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[TuningResult, Pipeline]:
    """Execute controlled hyperparameter search strictly within the training partition.

    Enforces that hyperparameter evaluation is performed strictly via internal
    StratifiedKFold cross-validation on X_train. The holdout test set is never touched.

    Parameters
    ----------
    pipeline : Pipeline
        Base pipeline containing preprocessor and classifier.
    param_grid : dict[str, list[Any]]
        Dictionary of parameters (e.g. {'classifier__C': [0.1, 1.0]}).
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target vector.
    model_name : str
        Human-readable model name.
    scoring : str
        Scoring metric for ranking candidates (default: 'average_precision').
    n_splits : int
        Number of internal CV folds (default: 5).
    search_type : str
        Search strategy ('grid' or 'randomized').
    n_iter : int
        Number of parameter settings sampled if search_type is 'randomized'.
    random_state : int
        Deterministic random seed.
    n_jobs : int
        Parallel CPU jobs.

    Returns
    -------
    tuple[TuningResult, Pipeline]
        TuningResult metadata and the refitted best estimator pipeline.
    """
    y_arr = np.asarray(y_train).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    logger.info(
        "Starting %s search for '%s' optimizing '%s' over %d folds...",
        search_type,
        model_name,
        scoring,
        n_splits,
    )

    t_start = time.perf_counter()
    if search_type == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=skf,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=skf,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
            return_train_score=False,
        )

    search.fit(X_train, y_arr)
    t_end = time.perf_counter()

    cv_results = pd.DataFrame(search.cv_results_)
    top_cols = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    top_summary = (
        cv_results[top_cols]
        .sort_values(by="rank_test_score")
        .head(10)
        .to_dict(orient="records")
    )

    # Clean params dict keys for serialization
    cleaned_best_params = {
        k.replace("classifier__", ""): v for k, v in search.best_params_.items()
    }

    result = TuningResult(
        model_name=model_name,
        best_params=cleaned_best_params,
        best_score=float(search.best_score_),
        scoring_metric=scoring,
        search_space={k.replace("classifier__", ""): v for k, v in param_grid.items()},
        search_type=search_type,
        n_candidates_evaluated=len(cv_results),
        search_duration_s=float(t_end - t_start),
        cv_summary_table=top_summary,
    )

    logger.info(
        "Finished tuning '%s': Best %s=%.4f with params %s in %.2fs",
        model_name,
        scoring,
        result.best_score,
        result.best_params,
        result.search_duration_s,
    )

    return result, search.best_estimator_


def tune_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    preprocessor: Pipeline | None = None,
    random_state: int = 42,
) -> tuple[TuningResult, Pipeline]:
    """Execute controlled hyperparameter search for HistGradientBoostingClassifier."""
    base_pipeline = create_hist_gradient_boosting_pipeline(
        preprocessor=preprocessor, random_state=random_state
    )
    param_grid = {
        "classifier__learning_rate": [0.03, 0.05, 0.08],
        "classifier__max_iter": [100, 150, 200],
        "classifier__max_leaf_nodes": [15, 31],
        "classifier__min_samples_leaf": [20, 40],
        "classifier__l2_regularization": [0.1, 1.0, 5.0],
    }

    return tune_pipeline_hyperparameters(
        pipeline=base_pipeline,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        model_name="Hist Gradient Boosting",
        search_type="grid",
        random_state=random_state,
    )


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    preprocessor: Pipeline | None = None,
    random_state: int = 42,
) -> tuple[TuningResult, Pipeline]:
    """Execute controlled hyperparameter search for Logistic Regression."""
    base_pipeline = create_logistic_regression_pipeline(
        solver="saga", preprocessor=preprocessor, max_iter=2000, random_state=random_state
    )
    param_grid = {
        "classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"],
    }

    return tune_pipeline_hyperparameters(
        pipeline=base_pipeline,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        model_name="Logistic Regression",
        search_type="grid",
        random_state=random_state,
    )

