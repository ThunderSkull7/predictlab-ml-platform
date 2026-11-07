"""Unit tests for model selection, holdout evaluation, tuning, and persistence."""

import tempfile
from pathlib import Path

import numpy as np

from predictlab.data.load import load_raw_dataset
from predictlab.models.cross_validation import CrossValidationResult
from predictlab.models.estimators import create_logistic_regression_pipeline
from predictlab.models.persistence import (
    load_model_artifact,
    save_experiment_json,
    save_model_artifact,
)
from predictlab.models.selection import (
    HoldoutEvaluationResult,
    evaluate_final_model_on_holdout,
    select_best_candidate,
)
from predictlab.models.tuning import TuningResult, tune_pipeline_hyperparameters
from predictlab.preprocessing.split import split_dataset


def test_select_best_candidate() -> None:
    """Test candidate model selection based on primary PR-AUC metric."""
    cand1 = CrossValidationResult(
        model_name="ModelX",
        n_splits=5,
        fold_metrics=[],
        mean_metrics={"average_precision": 0.60, "roc_auc": 0.82, "log_loss": 0.42},
        std_metrics={"average_precision": 0.02, "roc_auc": 0.01},
        mean_fit_time=0.1,
        mean_score_time=0.01,
        warnings_encountered=[],
    )
    cand2 = CrossValidationResult(
        model_name="ModelY",
        n_splits=5,
        fold_metrics=[],
        mean_metrics={"average_precision": 0.68, "roc_auc": 0.85, "log_loss": 0.40},
        std_metrics={"average_precision": 0.01, "roc_auc": 0.01},
        mean_fit_time=0.2,
        mean_score_time=0.01,
        warnings_encountered=[],
    )

    chosen, rationale = select_best_candidate([cand1, cand2], primary_metric="average_precision")
    assert chosen.model_name == "ModelY"
    assert "ModelY" in rationale
    assert "0.6800" in rationale


def test_holdout_evaluation_and_isolation() -> None:
    """Test one-time holdout evaluation on small subsample."""
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, _ = split_dataset(df, save_manifest=False)

    pipeline = create_logistic_regression_pipeline(max_iter=300)
    result, refitted = evaluate_final_model_on_holdout(
        pipeline=pipeline,
        X_train=X_train.iloc[:200],
        y_train=y_train.iloc[:200],
        X_test=X_test.iloc[:50],
        y_test=y_test.iloc[:50],
        model_name="TestLR",
        threshold=0.5,
    )

    assert isinstance(result, HoldoutEvaluationResult)
    assert result.model_name == "TestLR"
    assert result.threshold == 0.5
    assert len(result.feature_names) == 47
    assert result.feature_importances is not None
    assert len(result.feature_importances) == 47
    assert "average_precision" in result.metrics
    assert "roc" in result.curves
    assert "pr" in result.curves
    assert result.confusion_matrix["total_samples"] == 50


def test_model_artifact_save_and_reload() -> None:
    """Test saving model artifact bundle with metadata and re-loading it identically."""
    df = load_raw_dataset()
    X_train, X_test, y_train, _, _ = split_dataset(df, save_manifest=False)

    pipeline = create_logistic_regression_pipeline(max_iter=300)
    pipeline.fit(X_train.iloc[:100], y_train.iloc[:100])
    expected_preds = pipeline.predict_proba(X_test.iloc[:20])

    metadata = {
        "dataset_sha256": "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
        "model_name": "TestLR",
        "feature_count": 47,
        "cv_metrics_mean": {"average_precision": 0.65, "threshold": 0.5},
        "cv_metrics_std": {"average_precision": 0.02, "threshold": 0.0},
        "holdout_metrics": {"average_precision": 0.64, "threshold": 0.5},
        "decision_threshold": 0.5,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test_model.joblib"
        saved_path = save_model_artifact(pipeline, metadata, file_path=tmp_file)
        assert saved_path.exists()

        bundle = load_model_artifact(saved_path)
        assert "pipeline" in bundle
        assert "metadata" in bundle
        assert bundle["metadata"]["model_name"] == "TestLR"
        assert bundle["metadata"]["feature_count"] == 47
        assert "environment" in bundle["metadata"]
        assert "threshold" not in bundle["metadata"]["cv_metrics_mean"]
        assert "threshold" not in bundle["metadata"]["cv_metrics_std"]
        assert "threshold" not in bundle["metadata"]["holdout_metrics"]
        assert bundle["metadata"]["decision_threshold"] == 0.5

        loaded_pipeline = bundle["pipeline"]
        loaded_preds = loaded_pipeline.predict_proba(X_test.iloc[:20])
        np.testing.assert_allclose(expected_preds, loaded_preds)


def test_experiment_json_serialization() -> None:
    """Test serializing experiment results to JSON."""
    data = {
        "experiment": "Phase4_Benchmark",
        "primary_metric": "average_precision",
        "scores": [0.65, 0.67, 0.68],
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_file = save_experiment_json(data, "test_exp.json", output_dir=tmp_dir)
        assert saved_file.exists()
        assert "Phase4_Benchmark" in saved_file.read_text(encoding="utf-8")


def test_controlled_hyperparameter_tuning_small() -> None:
    """Test controlled hyperparameter tuning with a lightweight 2-candidate grid."""
    df = load_raw_dataset()
    X_train, _, y_train, _, _ = split_dataset(df, save_manifest=False)

    pipeline = create_logistic_regression_pipeline(max_iter=200)
    param_grid = {"classifier__C": [0.1, 1.0]}

    res, best_pipe = tune_pipeline_hyperparameters(
        pipeline=pipeline,
        param_grid=param_grid,
        X_train=X_train.iloc[:150],
        y_train=y_train.iloc[:150],
        model_name="MiniTuningLR",
        n_splits=2,
    )

    assert isinstance(res, TuningResult)
    assert res.n_candidates_evaluated == 2
    assert "C" in res.best_params
    assert res.best_score > 0.0
    assert hasattr(best_pipe, "predict_proba")

