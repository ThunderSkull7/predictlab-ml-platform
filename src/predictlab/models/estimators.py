"""Candidate model factories and pipeline composition for PredictLab."""

from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from predictlab.preprocessing.pipelines import (
    create_linear_preprocessor,
    create_tree_preprocessor,
)


def build_model_pipeline(
    preprocessor: Pipeline,
    estimator: BaseEstimator,
) -> Pipeline:
    """Compose a preprocessing pipeline with a classification estimator.

    Ensures that preprocessing transformations and model fitting are strictly encapsulated
    in a single scikit-learn Pipeline object, preventing leakage during cross-validation.

    Parameters
    ----------
    preprocessor : Pipeline
        Fitted or unfitted scikit-learn preprocessing pipeline.
    estimator : BaseEstimator
        Classification estimator.

    Returns
    -------
    Pipeline
        End-to-end Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)]).
    """
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])


def create_logistic_regression_pipeline(
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
    c_param: float = 1.0,
    class_weight: Literal["balanced"] | None = None,
    solver: Literal["lbfgs", "saga", "liblinear"] = "lbfgs",
    max_iter: int = 2000,
    random_state: int = 42,
    preprocessor: Pipeline | None = None,
    **kwargs: Any,
) -> Pipeline:
    """Create a Logistic Regression pipeline with standardized numerical scaling.

    Parameters
    ----------
    penalty : str
        Regularization penalty ('l1' or 'l2').
    c_param : float
        Inverse regularization strength (smaller values specify stronger regularization).
    class_weight : str or None
        Class weighting ('balanced' or None).
    solver : str
        Optimization algorithm ('lbfgs' for l2, 'saga' for l1).
    max_iter : int
        Maximum optimization iterations to guarantee convergence (default: 2000).
    random_state : int
        Deterministic random seed (default: 42).
    preprocessor : Pipeline, optional
        Custom preprocessing pipeline (defaults to create_linear_preprocessor()).
    **kwargs : Any
        Additional keyword arguments passed to LogisticRegression.

    Returns
    -------
    Pipeline
        Composed scikit-learn Pipeline.
    """
    # Auto-adjust solver for L1 penalty if default solver was kept
    if penalty == "l1" and solver == "lbfgs":
        solver = "saga"

    prep = preprocessor if preprocessor is not None else create_linear_preprocessor()
    estimator = LogisticRegression(
        penalty=penalty,
        C=c_param,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs,
    )
    return build_model_pipeline(prep, estimator)


def create_random_forest_pipeline(
    n_estimators: int = 150,
    max_depth: int | None = 8,
    min_samples_leaf: int = 5,
    max_features: Literal["sqrt", "log2"] | float | None = "sqrt",
    class_weight: Literal["balanced", "balanced_subsample"] | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    preprocessor: Pipeline | None = None,
    **kwargs: Any,
) -> Pipeline:
    """Create a Random Forest pipeline using the tree preprocessor (unscaled numericals).

    Parameters
    ----------
    n_estimators : int
        Number of decision trees in forest (default: 150).
    max_depth : int or None
        Maximum tree depth constraint to prevent memorization (default: 8).
    min_samples_leaf : int
        Minimum samples required at a leaf node (default: 5).
    max_features : str or float or None
        Features to consider for each split (default: 'sqrt').
    class_weight : str or None
        Class weighting scheme ('balanced', 'balanced_subsample', or None).
    random_state : int
        Deterministic random seed (default: 42).
    n_jobs : int
        Parallel CPU threads (default: -1 for all available).
    preprocessor : Pipeline, optional
        Custom preprocessing pipeline (defaults to create_tree_preprocessor()).
    **kwargs : Any
        Additional keyword arguments passed to RandomForestClassifier.

    Returns
    -------
    Pipeline
        Composed scikit-learn Pipeline.
    """
    prep = preprocessor if preprocessor is not None else create_tree_preprocessor()
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
    return build_model_pipeline(prep, estimator)


def create_hist_gradient_boosting_pipeline(
    learning_rate: float = 0.05,
    max_iter: int = 150,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    class_weight: Literal["balanced"] | None = None,
    random_state: int = 42,
    preprocessor: Pipeline | None = None,
    **kwargs: Any,
) -> Pipeline:
    """Create a Histogram Gradient Boosting pipeline using the tree preprocessor.

    Parameters
    ----------
    learning_rate : float
        Shrinkage factor (default: 0.05).
    max_iter : int
        Maximum boosting iterations / number of trees (default: 150).
    max_leaf_nodes : int
        Maximum leaves per tree controlling tree complexity (default: 31).
    min_samples_leaf : int
        Minimum samples per leaf (default: 20).
    l2_regularization : float
        L2 penalty on leaf values (default: 1.0).
    class_weight : str or None
        Class weighting ('balanced' or None).
    random_state : int
        Deterministic random seed (default: 42).
    preprocessor : Pipeline, optional
        Custom preprocessing pipeline (defaults to create_tree_preprocessor()).
    **kwargs : Any
        Additional keyword arguments passed to HistGradientBoostingClassifier.

    Returns
    -------
    Pipeline
        Composed scikit-learn Pipeline.
    """
    prep = preprocessor if preprocessor is not None else create_tree_preprocessor()
    estimator = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs,
    )
    return build_model_pipeline(prep, estimator)


def get_candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return dictionary of canonical candidate models for Phase 4 benchmarking.

    Includes dummy baselines, regularized linear models, random forests, and gradient boosting.

    Parameters
    ----------
    random_state : int
        Deterministic seed across estimators.

    Returns
    -------
    dict[str, Pipeline]
        Mapping of model identifier to configured Pipeline.
    """
    from predictlab.models.baselines import create_dummy_baseline

    return {
        "Dummy (Prior)": create_dummy_baseline(strategy="prior", random_state=random_state),
        "Dummy (Most Frequent)": create_dummy_baseline(
            strategy="most_frequent", random_state=random_state
        ),
        "Logistic Regression (L2)": create_logistic_regression_pipeline(
            penalty="l2", c_param=1.0, class_weight=None, random_state=random_state
        ),
        "Logistic Regression (L2 Balanced)": create_logistic_regression_pipeline(
            penalty="l2", c_param=1.0, class_weight="balanced", random_state=random_state
        ),
        "Logistic Regression (L1)": create_logistic_regression_pipeline(
            penalty="l1", c_param=1.0, class_weight=None, solver="saga", random_state=random_state
        ),
        "Random Forest (Baseline)": create_random_forest_pipeline(
            n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=random_state
        ),
        "Random Forest (Constrained)": create_random_forest_pipeline(
            n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=random_state
        ),
        "Random Forest (Balanced)": create_random_forest_pipeline(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Hist Gradient Boosting (Default)": create_hist_gradient_boosting_pipeline(
            learning_rate=0.05, max_iter=150, random_state=random_state
        ),
        "Hist Gradient Boosting (Balanced)": create_hist_gradient_boosting_pipeline(
            learning_rate=0.05, max_iter=150, class_weight="balanced", random_state=random_state
        ),
    }

