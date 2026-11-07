"""Unit tests for stratified cross-validation and fold-level evaluation."""

from predictlab.data.load import load_raw_dataset
from predictlab.models.cross_validation import (
    CrossValidationResult,
    cv_results_to_dataframe,
    evaluate_pipeline_cv,
)
from predictlab.models.estimators import create_logistic_regression_pipeline
from predictlab.preprocessing.split import split_dataset


def test_cross_validation_execution_and_metrics() -> None:
    """Test 3-fold cross-validation on subsample to verify fold metric generation and aggregation."""
    df = load_raw_dataset()
    X_train, _, y_train, _, _ = split_dataset(df, save_manifest=False)

    # Use 200 samples for swift test execution
    X_sub = X_train.iloc[:200]
    y_sub = y_train.iloc[:200]

    pipeline = create_logistic_regression_pipeline(max_iter=300)
    result = evaluate_pipeline_cv(
        pipeline=pipeline,
        X=X_sub,
        y=y_sub,
        model_name="FastLR",
        n_splits=3,
        random_state=42,
    )

    assert isinstance(result, CrossValidationResult)
    assert result.n_splits == 3
    assert len(result.fold_metrics) == 3

    # All required metrics present in fold scores
    required_metrics = [
        "average_precision",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "log_loss",
        "brier_score",
    ]
    for metric in required_metrics:
        assert metric in result.mean_metrics
        assert metric in result.std_metrics
        assert 0.0 <= result.mean_metrics[metric] <= 1.0 or metric == "log_loss"

    assert result.mean_fit_time > 0.0
    assert result.mean_score_time >= 0.0


def test_cv_results_to_dataframe() -> None:
    """Test converting CV results into ranked summary DataFrame."""
    res1 = CrossValidationResult(
        model_name="ModelA",
        n_splits=5,
        fold_metrics=[{"average_precision": 0.65, "roc_auc": 0.82}],
        mean_metrics={"average_precision": 0.65, "roc_auc": 0.82},
        std_metrics={"average_precision": 0.02, "roc_auc": 0.01},
        mean_fit_time=0.1,
        mean_score_time=0.01,
        warnings_encountered=[],
    )
    res2 = CrossValidationResult(
        model_name="ModelB",
        n_splits=5,
        fold_metrics=[{"average_precision": 0.70, "roc_auc": 0.85}],
        mean_metrics={"average_precision": 0.70, "roc_auc": 0.85},
        std_metrics={"average_precision": 0.01, "roc_auc": 0.01},
        mean_fit_time=0.2,
        mean_score_time=0.02,
        warnings_encountered=[],
    )

    df_summary = cv_results_to_dataframe([res1, res2])
    assert len(df_summary) == 2
    # ModelB has higher PR-AUC, should be ranked first
    assert df_summary.iloc[0]["Model"] == "ModelB"
    assert "average_precision" in df_summary.columns

