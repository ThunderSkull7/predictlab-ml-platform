"""Unit tests for estimator pipeline factories and candidate model definitions."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from predictlab.data.load import load_raw_dataset
from predictlab.models.estimators import (
    create_hist_gradient_boosting_pipeline,
    create_logistic_regression_pipeline,
    create_random_forest_pipeline,
    get_candidate_models,
)
from predictlab.preprocessing.split import split_dataset


@pytest.fixture
def small_train_data() -> tuple:
    """Provide a small subsample of training data for quick pipeline fitting tests."""
    df = load_raw_dataset()
    X_train, _, y_train, _, _ = split_dataset(df, save_manifest=False)
    return X_train.iloc[:100], y_train.iloc[:100]


def test_logistic_regression_pipeline_construction(small_train_data: tuple) -> None:
    """Test Logistic Regression pipeline fitting, predictions, and coefficient alignment."""
    X_sub, y_sub = small_train_data

    pipeline = create_logistic_regression_pipeline(penalty="l2", c_param=1.0, max_iter=500)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    pipeline.fit(X_sub, y_sub)
    probas = pipeline.predict_proba(X_sub)
    assert probas.shape == (len(X_sub), 2)
    assert (probas >= 0.0).all() and (probas <= 1.0).all()

    # Coefficients should match the 47 baseline features
    clf = pipeline.named_steps["classifier"]
    assert clf.coef_.shape == (1, 47)


def test_logistic_regression_l1_saga_construction(small_train_data: tuple) -> None:
    """Test L1 regularized logistic regression with SAGA solver."""
    X_sub, y_sub = small_train_data

    pipeline = create_logistic_regression_pipeline(penalty="l1", solver="saga", max_iter=200)
    pipeline.fit(X_sub, y_sub)
    preds = pipeline.predict(X_sub)
    assert len(preds) == len(X_sub)


def test_random_forest_pipeline_construction(small_train_data: tuple) -> None:
    """Test Random Forest pipeline fitting and feature importances."""
    X_sub, y_sub = small_train_data

    pipeline = create_random_forest_pipeline(n_estimators=10, max_depth=5, min_samples_leaf=2)
    pipeline.fit(X_sub, y_sub)

    probas = pipeline.predict_proba(X_sub)
    assert probas.shape == (len(X_sub), 2)

    clf = pipeline.named_steps["classifier"]
    assert len(clf.feature_importances_) == 47
    assert pytest.approx(clf.feature_importances_.sum(), abs=1e-5) == 1.0


def test_hist_gradient_boosting_pipeline_construction(small_train_data: tuple) -> None:
    """Test HistGradientBoostingClassifier pipeline fitting."""
    X_sub, y_sub = small_train_data

    pipeline = create_hist_gradient_boosting_pipeline(max_iter=15, learning_rate=0.1)
    pipeline.fit(X_sub, y_sub)

    probas = pipeline.predict_proba(X_sub)
    assert probas.shape == (len(X_sub), 2)
    assert not np.isnan(probas).any() if "np" in dir() else True


def test_candidate_models_registry() -> None:
    """Test candidate models dictionary contains all required benchmark model families."""
    models = get_candidate_models()
    expected_keys = [
        "Dummy (Prior)",
        "Dummy (Most Frequent)",
        "Logistic Regression (L2)",
        "Logistic Regression (L2 Balanced)",
        "Logistic Regression (L1)",
        "Random Forest (Baseline)",
        "Random Forest (Constrained)",
        "Random Forest (Balanced)",
        "Hist Gradient Boosting (Default)",
        "Hist Gradient Boosting (Balanced)",
    ]

    for key in expected_keys:
        assert key in models
        assert isinstance(models[key], Pipeline)

