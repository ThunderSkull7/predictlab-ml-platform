"""Baseline model factories for establishing performance benchmarks."""

from typing import Literal

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from predictlab.preprocessing.pipelines import create_linear_preprocessor


def create_dummy_baseline(
    strategy: Literal["most_frequent", "prior", "stratified"] = "prior",
    random_state: int = 42,
) -> Pipeline:
    """Create a standardized dummy baseline pipeline.

    Establishes the minimal performance benchmark against which machine learning models
    must be compared:
    - 'most_frequent': Always predicts dominant negative class (0 / Retained).
    - 'prior': Always predicts class with probabilities equal to training class prior.
    - 'stratified': Generates random predictions respecting class priors.

    Parameters
    ----------
    strategy : str
        Dummy prediction strategy ('most_frequent', 'prior', or 'stratified').
    random_state : int
        Deterministic random seed (default: 42).

    Returns
    -------
    Pipeline
        Composed scikit-learn Pipeline with preprocessing and DummyClassifier.
    """
    preprocessor = create_linear_preprocessor()
    estimator = DummyClassifier(strategy=strategy, random_state=random_state)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])

