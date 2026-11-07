"""Execution script running the complete Phase 4 modeling, ablation, tuning, and evaluation workflow."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from predictlab.config import get_settings
from predictlab.data.load import load_raw_dataset
from predictlab.logging_config import get_logger
from predictlab.models.cross_validation import evaluate_pipeline_cv
from predictlab.models.estimators import (
    create_hist_gradient_boosting_pipeline,
    get_candidate_models,
)
from predictlab.models.experiments import (
    run_feature_ablation_study,
    run_model_benchmarking,
)
from predictlab.models.persistence import (
    save_experiment_json,
    save_model_artifact,
)
from predictlab.models.selection import (
    evaluate_final_model_on_holdout,
    select_best_candidate,
)
from predictlab.models.tuning import tune_pipeline_hyperparameters
from predictlab.preprocessing.split import split_dataset

logger = get_logger(__name__)

# Configure matplotlib aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
    }
)


def main() -> None:
    """Run full Phase 4 experimental pipeline."""
    settings = get_settings()
    exp_dir = settings.reports_dir / "experiments"
    fig_dir = settings.reports_dir / "figures"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(" PREDICTLAB PHASE 4: MODEL EXPLORATION & CONTROLLED BENCHMARKING")
    print("=================================================================")

    # 1. Load canonical dataset and stratified partitions
    print("\n[Step 1/7] Loading canonical dataset and verified 80/20 train/test split...")
    df_raw = load_raw_dataset()
    X_train, X_test, y_train, y_test, manifest = split_dataset(
        df_raw,
        test_size=settings.test_size,
        random_state=settings.random_state,
        save_manifest=False,
    )
    print(f"  Training partition: {X_train.shape[0]} samples, {X_train.shape[1]} predictors")
    print(f"  Holdout test partition: {X_test.shape[0]} samples (STRICTLY ISOLATED)")
    print(f"  Training churn rate: {y_train.mean() * 100:.2f}% ({y_train.sum()} churned)")

    # 2. Candidate Model Benchmarking (5-fold stratified CV on X_train only)
    print("\n[Step 2/7] Executing 5-fold Stratified CV across all candidate model families...")
    candidate_models = get_candidate_models(random_state=settings.random_state)
    bench_results, df_bench = run_model_benchmarking(
        X_train=X_train,
        y_train=y_train,
        candidate_models=candidate_models,
        n_splits=5,
        random_state=settings.random_state,
    )

    print("\n--- Cross-Validation Benchmark Results (Ranked by PR-AUC) ---")
    display_cols = ["Model", "average_precision", "roc_auc", "f1", "balanced_accuracy", "log_loss", "mean_fit_time_s"]
    print(df_bench[display_cols].to_string(index=False))

    # Persist benchmark artifacts
    df_bench.to_csv(exp_dir / "model_benchmarks_summary.csv", index=False)
    bench_json = [r.model_dump() for r in bench_results]
    save_experiment_json(bench_json, "model_benchmarks.json")

    # Plot Candidate PR-AUC comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models_sorted = df_bench.sort_values(by="average_precision_mean", ascending=True)
    colors = ["#2b5c8f" if "Hist" in m else "#4682b4" if "Forest" in m else "#5c92c2" if "Logistic" in m else "#999999" for m in models_sorted["Model"]]
    y_pos = np.arange(len(models_sorted))
    ax.barh(y_pos, models_sorted["average_precision_mean"], xerr=models_sorted["average_precision_std"], align="center", color=colors, capsize=4, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_sorted["Model"], fontsize=10)
    ax.set_xlabel("Mean Average Precision / PR-AUC (5-Fold CV)", fontsize=11, fontweight="bold")
    ax.set_title("PredictLab Phase 4: Candidate Model Benchmark (5-Fold Stratified CV)", fontsize=13, fontweight="bold", pad=12)
    ax.axvline(x=y_train.mean(), color="#d9534f", linestyle="--", linewidth=1.2, label=f"Dummy Baseline Prior ({y_train.mean():.3f})")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(fig_dir / "17_candidate_models_pr_auc_comparison.png", dpi=300)
    plt.close(fig)
    print("  Saved figure: 17_candidate_models_pr_auc_comparison.png")

    # 3. Experimental Feature Ablation Study
    print("\n[Step 3/7] Running controlled feature ablation study (47 baseline vs 52 experimental)...")
    ablation_results, df_ablation = run_feature_ablation_study(
        X_train=X_train,
        y_train=y_train,
        n_splits=5,
        random_state=settings.random_state,
    )

    print("\n--- Feature Ablation Results (Relative to 47-Feature Baseline) ---")
    print(df_ablation.to_string(index=False))

    df_ablation.to_csv(exp_dir / "feature_ablation_summary.csv", index=False)
    save_experiment_json([r.model_dump() for r in ablation_results], "feature_ablation_results.json")

    # Plot Ablation Deltas
    fig, ax = plt.subplots(figsize=(10, 5))
    ab_labels = df_ablation["Configuration"].tolist()
    delta_pr = df_ablation["Δ PR-AUC"].tolist()
    colors = ["#999999" if d == 0 else "#2e7d32" if d > 0 else "#c62828" for d in delta_pr]
    bars = ax.barh(ab_labels, delta_pr, color=colors, alpha=0.85)
    ax.axvline(x=0, color="#333333", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Δ PR-AUC Relative to 47-Feature Baseline", fontsize=11, fontweight="bold")
    ax.set_title("PredictLab Phase 4: Feature Ablation Impact on HistGradientBoosting (5-Fold CV)", fontsize=12, fontweight="bold", pad=12)
    for bar in bars:
        val = bar.get_width()
        offset = 0.0003 if val >= 0 else -0.0018
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.4f}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "18_feature_ablation_delta_comparison.png", dpi=300)
    plt.close(fig)
    print("  Saved figure: 18_feature_ablation_delta_comparison.png")

    # 4. Controlled Hyperparameter Tuning
    print("\n[Step 4/7] Performing controlled hyperparameter search for HistGradientBoosting on training partition...")
    base_pipeline = create_hist_gradient_boosting_pipeline(random_state=settings.random_state)
    param_grid = {
        "classifier__learning_rate": [0.03, 0.05, 0.08],
        "classifier__max_iter": [100, 150, 200],
        "classifier__max_leaf_nodes": [15, 31],
        "classifier__min_samples_leaf": [20, 40],
        "classifier__l2_regularization": [0.1, 1.0, 5.0],
    }
    tuning_res, tuned_pipeline = tune_pipeline_hyperparameters(
        pipeline=base_pipeline,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        model_name="Hist Gradient Boosting",
        scoring="average_precision",
        n_splits=5,
        search_type="grid",
        random_state=settings.random_state,
    )
    print(f"  Best Tuning PR-AUC: {tuning_res.best_score:.4f}")
    print(f"  Best Parameters: {tuning_res.best_params}")
    save_experiment_json(tuning_res.model_dump(), "hyperparameter_tuning_results.json")

    # Also evaluate tuned pipeline via 5-fold CV to capture all full metric statistics
    tuned_cv = evaluate_pipeline_cv(
        pipeline=tuned_pipeline,
        X=X_train,
        y=y_train,
        model_name="Hist Gradient Boosting (Tuned)",
        n_splits=5,
        random_state=settings.random_state,
    )
    save_experiment_json(tuned_cv.model_dump(), "tuned_model_cv_results.json")

    # 5. Model Selection
    print("\n[Step 5/7] Selecting final candidate model based on cross-validation criteria...")
    all_cv_candidates = bench_results + [tuned_cv]
    best_candidate, rationale = select_best_candidate(all_cv_candidates, primary_metric="average_precision")
    print(f"  Selected: {best_candidate.model_name}")
    print(f"  Rationale: {rationale}")
    save_experiment_json(
        {"selected_model": best_candidate.model_name, "rationale": rationale, "metrics": best_candidate.mean_metrics},
        "model_selection_rationale.json",
    )

    # 6. One-Time Holdout Evaluation
    print("\n[Step 6/7] Refitting selected model on full X_train and evaluating ONCE on holdout test partition...")
    holdout_res, refitted_final_pipeline = evaluate_final_model_on_holdout(
        pipeline=tuned_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name=best_candidate.model_name,
        threshold=0.5,
    )

    print("\n=======================================================")
    print(" ONE-TIME HOLDOUT TEST EVALUATION RESULTS (N = 1,409)")
    print("=======================================================")
    for m, val in holdout_res.metrics.items():
        print(f"  {m:20s}: {val:.4f}")

    print("\nConfusion Matrix:")
    cm = holdout_res.confusion_matrix
    print(f"  True Negatives:  {cm['true_negatives']:4d} | False Positives: {cm['false_positives']:4d}")
    print(f"  False Negatives: {cm['false_negatives']:4d} | True Positives:  {cm['true_positives']:4d}")

    save_experiment_json(holdout_res.model_dump(), "holdout_evaluation.json")

    # Plot Holdout ROC and PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curve
    roc_data = holdout_res.curves["roc"]
    ax1.plot(roc_data["fpr"], roc_data["tpr"], color="#1f77b4", linewidth=2.0, label=f"Tuned HGB (AUC = {roc_data['roc_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.2, label="Random Guess (AUC = 0.5000)")
    ax1.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (Holdout Test)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)

    # PR curve
    pr_data = holdout_res.curves["pr"]
    ax2.plot(pr_data["recall"], pr_data["precision"], color="#2ca02c", linewidth=2.0, label=f"Tuned HGB (PR-AUC = {pr_data['average_precision']:.4f})")
    ax2.axhline(y=pr_data["baseline_prevalence"], color="#d9534f", linestyle="--", linewidth=1.2, label=f"Prior Churn Rate ({pr_data['baseline_prevalence']:.4f})")
    ax2.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title("Precision-Recall Curve (Holdout Test)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(fig_dir / "19_holdout_roc_pr_curves.png", dpi=300)
    plt.close(fig)
    print("  Saved figure: 19_holdout_roc_pr_curves.png")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_matrix = np.array([[cm["true_negatives"], cm["false_positives"]], [cm["false_negatives"], cm["true_positives"]]])
    sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], annot_kws={"size": 13, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title("Holdout Test Confusion Matrix (Threshold = 0.5)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(fig_dir / "20_holdout_confusion_matrix.png", dpi=300)
    plt.close(fig)
    print("  Saved figure: 20_holdout_confusion_matrix.png")

    # Plot Feature Importances Top 20 (Impurity-based / Permutation proxy)
    if holdout_res.feature_importances:
        fig, ax = plt.subplots(figsize=(10, 8))
        sorted_imp = sorted(holdout_res.feature_importances.items(), key=lambda x: x[1], reverse=True)[:20]
        f_names, f_vals = zip(*sorted_imp, strict=True)
        y_pos = np.arange(len(f_names))
        clean_names = [f.replace("num__", "").replace("cat__", "") for f in f_names]
        ax.barh(y_pos, f_vals, align="center", color="#3b6998", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(clean_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Relative Feature Importance (Sum = 1.0)", fontsize=11, fontweight="bold")
        ax.set_title("Top 20 Features in Selected Production Candidate (HistGradientBoosting)", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        fig.savefig(fig_dir / "21_feature_importances_top20.png", dpi=300)
        plt.close(fig)
        print("  Saved figure: 21_feature_importances_top20.png")

    # 7. Persist Final Serialized Model Bundle
    print("\n[Step 7/7] Serializing final model artifact bundle to artifacts/model.joblib...")
    metadata = {
        "dataset_sha256": manifest.raw_data_sha256,
        "train_test_split_seed": settings.random_state,
        "split_provenance": "Canonical Phase 3 stratified holdout partition (80/20 train/test split, random_state=42, stratify=Churn) recorded in data/splits/split_manifest.json",
        "model_family": "HistGradientBoostingClassifier",
        "model_name": best_candidate.model_name,
        "preprocessing_pipeline": "baseline_tree",
        "feature_count": len(holdout_res.feature_names),
        "feature_names": holdout_res.feature_names,
        "hyperparameters": tuning_res.best_params,
        "cv_strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "cv_metrics_mean": best_candidate.mean_metrics,
        "cv_metrics_std": best_candidate.std_metrics,
        "holdout_metrics": holdout_res.metrics,
        "decision_threshold": 0.5,
        "evaluation_notes": (
            "cv_metrics_mean and cv_metrics_std represent training-only 5-fold cross-validation. "
            "holdout_metrics represent single independent evaluation on protected holdout test set (N=1,409). "
            "Output scores are model-estimated class probabilities, not calibrated probabilities. "
            "Formal calibration analysis is deferred to Phase 5."
        ),
    }
    artifact_path = save_model_artifact(refitted_final_pipeline, metadata)
    print(f"  Artifact successfully persisted: {artifact_path.resolve()}")
    print("\n=======================================================")
    print(" PHASE 4 EXPERIMENTATION WORKFLOW COMPLETED SUCCESSFULLY")
    print("=======================================================")


if __name__ == "__main__":
    main()
