"""Modeling package for PredictLab: candidate models, cross-validation, and selection."""

from predictlab.models.baselines import create_dummy_baseline
from predictlab.models.cross_validation import (
    CrossValidationResult,
    cv_results_to_dataframe,
    evaluate_pipeline_cv,
)
from predictlab.models.estimators import (
    build_model_pipeline,
    create_hist_gradient_boosting_pipeline,
    create_logistic_regression_pipeline,
    create_random_forest_pipeline,
    get_candidate_models,
)
from predictlab.models.evaluation import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_confusion_matrix_summary,
    compute_roc_pr_curves,
)
from predictlab.models.experiments import (
    run_feature_ablation_study,
    run_model_benchmarking,
)
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
from predictlab.models.tuning import (
    TuningResult,
    tune_hist_gradient_boosting,
    tune_logistic_regression,
    tune_pipeline_hyperparameters,
)

__all__ = [
    "ClassificationMetrics",
    "CrossValidationResult",
    "HoldoutEvaluationResult",
    "TuningResult",
    "build_model_pipeline",
    "compute_classification_metrics",
    "compute_confusion_matrix_summary",
    "compute_roc_pr_curves",
    "create_dummy_baseline",
    "create_hist_gradient_boosting_pipeline",
    "create_logistic_regression_pipeline",
    "create_random_forest_pipeline",
    "cv_results_to_dataframe",
    "evaluate_final_model_on_holdout",
    "evaluate_pipeline_cv",
    "get_candidate_models",
    "load_model_artifact",
    "run_feature_ablation_study",
    "run_model_benchmarking",
    "save_experiment_json",
    "save_model_artifact",
    "select_best_candidate",
    "tune_hist_gradient_boosting",
    "tune_logistic_regression",
    "tune_pipeline_hyperparameters",
]

