"""Unit tests for dummy baseline models."""

import numpy as np
import pytest

from predictlab.data.load import load_raw_dataset
from predictlab.models.baselines import create_dummy_baseline
from predictlab.preprocessing.split import split_dataset


@pytest.fixture
def sample_data() -> tuple:
    """Load canonical train/test split."""
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, _ = split_dataset(df, save_manifest=False)
    return X_train, X_test, y_train, y_test


def test_dummy_baseline_prior_predictions(sample_data: tuple) -> None:
    """Test DummyClassifier(strategy='prior') predicts probability matching training class prior."""
    X_train, X_test, y_train, _ = sample_data

    baseline = create_dummy_baseline(strategy="prior", random_state=42)
    baseline.fit(X_train, y_train)

    probas = baseline.predict_proba(X_test)
    assert probas.shape == (len(X_test), 2)

    # Class 1 probability must equal empirical training churn rate
    train_prior = float(np.mean(y_train))
    assert pytest.approx(probas[0, 1], abs=1e-3) == train_prior
    assert (probas[:, 1] == probas[0, 1]).all()


def test_dummy_baseline_most_frequent(sample_data: tuple) -> None:
    """Test DummyClassifier(strategy='most_frequent') always predicts dominant class (0)."""
    X_train, X_test, y_train, _ = sample_data

    baseline = create_dummy_baseline(strategy="most_frequent", random_state=42)
    baseline.fit(X_train, y_train)

    preds = baseline.predict(X_test)
    assert (preds == 0).all()


def test_dummy_baseline_pipeline_drops_customer_id(sample_data: tuple) -> None:
    """Test that customerID is safely dropped inside the dummy baseline preprocessor."""
    X_train, _, y_train, _ = sample_data

    # Add customerID explicitly to input features
    X_with_id = X_train.copy()
    X_with_id["customerID"] = [f"CUST-{i}" for i in range(len(X_train))]

    baseline = create_dummy_baseline(strategy="prior")
    # Must fit and predict without error
    baseline.fit(X_with_id, y_train)
    preds = baseline.predict(X_with_id.iloc[:5])
    assert len(preds) == 5

