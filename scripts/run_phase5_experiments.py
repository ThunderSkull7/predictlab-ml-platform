"""PredictLab — Phase 5: Calibration, Threshold Analysis, Explainability & Inference Experiment Pipeline.

Executes leakage-free nested probability calibration cross-validation, cost-sensitive threshold tuning,
global and local SHAP explainability, risk tiering, and serializes the production Phase 5 artifact.
"""

import shutil

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictlab.calibration.cross_validation import run_leakage_free_calibration_cv
from predictlab.calibration.selection import (
    fit_final_calibrated_pipeline,
    select_best_calibration_method,
)
from predictlab.config import get_settings
from predictlab.data.load import load_raw_dataset
from predictlab.explainability.global_importance import compute_training_permutation_importance
from predictlab.explainability.local_attribution import TreeShapExplainer
from predictlab.inference.engine import InferenceEngine
from predictlab.logging_config import get_logger
from predictlab.models.persistence import (
    load_model_artifact,
    save_experiment_json,
    save_model_artifact,
)
from predictlab.preprocessing.split import split_dataset
from predictlab.thresholding.costs import (
    CostMatrix,
    evaluate_cost_sensitive_thresholds,
    run_cost_ratio_sensitivity,
)
from predictlab.thresholding.metrics import evaluate_threshold_grid
from predictlab.thresholding.optimization import evaluate_threshold_policy_nested
from predictlab.thresholding.risk_tiers import RiskTierer

logger = get_logger(__name__)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    }
)


def main() -> None:
    settings = get_settings()
    fig_dir = settings.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = settings.reports_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = settings.artifacts_dir

    print("=======================================================")
    print(" PREDICTLAB — PHASE 5 EXPERIMENTATION PIPELINE")
    print("=======================================================")

    # 1. Load Data & Verify Holdout Protection
    print("\n[Step 1/8] Loading dataset and verifying strict holdout test isolation...")
    df_raw = load_raw_dataset()
    X_train, X_test, y_train, y_test, manifest = split_dataset(
        df_raw, test_size=0.20, random_state=settings.random_state, save_manifest=False
    )
    print(f"  Training partition: N = {len(X_train)} samples")
    print(f"  Holdout test partition: N = {len(X_test)} samples (STRICTLY PROTECTED)")
    print(f"  Dataset SHA-256: {manifest.raw_data_sha256[:16]}...")

    # 2. Historical Phase 4 Preservation
    print("\n[Step 2/8] Preserving historical Phase 4 model artifact...")
    phase4_path = artifacts_dir / "model.joblib"
    phase4_backup = artifacts_dir / "model_phase4.joblib"
    if phase4_path.exists():
        shutil.copy2(phase4_path, phase4_backup)
        print(f"  Historical Phase 4 model safely preserved at: {phase4_backup.name}")
        phase4_bundle = load_model_artifact(phase4_path)
        base_pipeline = phase4_bundle["pipeline"]
    else:
        raise FileNotFoundError(f"Base Phase 4 model not found at {phase4_path.resolve()}")

    # 3. Leakage-Free Calibration Cross-Validation
    print("\n[Step 3/8] Running leakage-free calibration cross-validation (5 outer folds, 5 inner folds)...")
    calib_results = run_leakage_free_calibration_cv(
        base_pipeline=base_pipeline,
        X_train=X_train,
        y_train=y_train,
        n_outer_splits=5,
        inner_cv=5,
        random_state=settings.random_state,
    )

    # Serialize calibration results to experiment JSON
    calib_json_payload = {}
    for name, (eval_res, _) in calib_results.items():
        calib_json_payload[name] = eval_res.model_dump()
    save_experiment_json(calib_json_payload, "calibration_cv_results.json")

    # Select best calibration method
    selected_calib_method, calib_rationale, calib_summary = select_best_calibration_method(
        {k: v[0] for k, v in calib_results.items()}
    )
    print(f"  Selected Calibration Method: '{selected_calib_method}'")
    print(f"  Selection Rationale: {calib_rationale}")
    save_experiment_json(
        {"selected_method": selected_calib_method, "rationale": calib_rationale, "summary": calib_summary},
        "calibration_selection_rationale.json",
    )

    # Extract chosen out-of-fold probabilities for downstream threshold analysis
    chosen_oof_probs = calib_results[selected_calib_method][1]

    # Plot Calibration Reliability Curves & Histograms
    print("  Generating Figure 22: Reliability Diagrams and Probability Distributions...")
    fig, (ax_curve, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"uncalibrated": "#7f7f7f", "sigmoid": "#1f77b4", "isotonic": "#2ca02c"}
    labels = {
        "uncalibrated": f"Uncalibrated (Brier={calib_summary['uncalibrated']['brier_score']:.4f}, ECE={calib_summary['uncalibrated']['ece_uniform']:.4f})",
        "sigmoid": f"Sigmoid / Platt (Brier={calib_summary['sigmoid']['brier_score']:.4f}, ECE={calib_summary['sigmoid']['ece_uniform']:.4f})",
        "isotonic": f"Isotonic (Brier={calib_summary['isotonic']['brier_score']:.4f}, ECE={calib_summary['isotonic']['ece_uniform']:.4f})",
    }

    ax_curve.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1.5, label="Perfect Calibration (y = x)")

    for name in ["uncalibrated", "sigmoid", "isotonic"]:
        res = calib_results[name][0]
        curve = res.reliability_curve_uniform
        ax_curve.plot(
            curve["mean_predicted_value"],
            curve["fraction_of_positives"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=colors[name],
            label=labels[name],
        )

    ax_curve.set_xlabel("Mean Predicted Probability (Out-of-Fold)", fontsize=11, fontweight="bold")
    ax_curve.set_ylabel("Observed Churn Rate (Fraction of Positives)", fontsize=11, fontweight="bold")
    ax_curve.set_title("Reliability Diagrams (5-Fold Out-of-Fold Calibration)", fontsize=12, fontweight="bold")
    ax_curve.legend(loc="upper left", frameon=True, fontsize=10)
    ax_curve.set_xlim(-0.02, 1.02)
    ax_curve.set_ylim(-0.02, 1.02)

    # Plot Probability Distribution
    bins = np.linspace(0, 1, 25)
    for name in ["uncalibrated", "sigmoid", "isotonic"]:
        probs = calib_results[name][1]
        ax_hist.hist(probs, bins=bins, histtype="step", linewidth=2.0, color=colors[name], label=name.capitalize())

    ax_hist.set_xlabel("Predicted Probability", fontsize=11, fontweight="bold")
    ax_hist.set_ylabel("Sample Count (N = 5,634)", fontsize=11, fontweight="bold")
    ax_hist.set_title("Out-of-Fold Predicted Probability Distributions", fontsize=12, fontweight="bold")
    ax_hist.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / "22_calibration_curves_comparison.png", dpi=300)
    plt.close(fig)

    # Plot Brier Score Decomposition & ECE comparison
    print("  Generating Figure 23: Brier Decomposition and ECE...")
    fig, (ax_brier, ax_ece) = plt.subplots(1, 2, figsize=(13, 5))

    names = ["Uncalibrated", "Sigmoid (Platt)", "Isotonic"]
    keys = ["uncalibrated", "sigmoid", "isotonic"]

    reliabilities = [calib_results[k][0].overall_metrics["brier_reliability"] * 1000 for k in keys]
    resolutions = [calib_results[k][0].overall_metrics["brier_resolution"] * 1000 for k in keys]

    x = np.arange(len(names))
    width = 0.35

    ax_brier.bar(x - width / 2, reliabilities, width, label="Reliability × 10³ (Lower is Better)", color="#d9534f", alpha=0.85)
    ax_brier.bar(x + width / 2, resolutions, width, label="Resolution × 10³ (Higher is Better)", color="#5cb85c", alpha=0.85)
    ax_brier.set_xticks(x)
    ax_brier.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax_brier.set_ylabel("Metric Value (× 10³)", fontsize=11, fontweight="bold")
    ax_brier.set_title("Brier Score Decomposition Components", fontsize=12, fontweight="bold")
    ax_brier.legend(loc="upper right", frameon=True)

    eces_unif = [calib_results[k][0].mean_metrics["ece_uniform"] for k in keys]
    eces_quant = [calib_results[k][0].mean_metrics["ece_quantile"] for k in keys]

    ax_ece.bar(x - width / 2, eces_unif, width, label="ECE (Equal-Width, 10 Bins)", color="#3b6998", alpha=0.85)
    ax_ece.bar(x + width / 2, eces_quant, width, label="ECE (Equal-Frequency, 10 Bins)", color="#f0ad4e", alpha=0.85)
    ax_ece.set_xticks(x)
    ax_ece.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax_ece.set_ylabel("Expected Calibration Error", fontsize=11, fontweight="bold")
    ax_ece.set_title("Expected Calibration Error (Binning Sensitivity)", fontsize=12, fontweight="bold")
    ax_ece.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(fig_dir / "23_brier_decomposition_and_ece.png", dpi=300)
    plt.close(fig)

    # 4. Decision Threshold Analysis
    print("\n[Step 4/8] Executing decision threshold analysis on calibrated out-of-fold predictions...")
    y_train_arr = np.asarray(y_train).astype(int)
    threshold_grid = np.linspace(0.10, 0.90, 81)
    df_thresh = evaluate_threshold_grid(y_train_arr, chosen_oof_probs, thresholds=threshold_grid)
    df_thresh.to_csv(exp_dir / "threshold_evaluation_grid.csv", index=False)

    # Cost Matrix and Sensitivity Analysis
    base_cost_matrix = CostMatrix(cost_fp=1.0, cost_fn=5.0)  # 5:1 loss ratio
    df_thresh_cost = evaluate_cost_sensitive_thresholds(df_thresh, base_cost_matrix)
    df_thresh_cost.to_csv(exp_dir / "threshold_cost_analysis.csv", index=False)

    df_cost_sensitivity = run_cost_ratio_sensitivity(df_thresh)
    df_cost_sensitivity.to_csv(exp_dir / "cost_ratio_sensitivity.csv", index=False)
    print("  Cost Ratio Sensitivity Analysis across hypothetical ratios (CFN : CFP):")
    for _, row in df_cost_sensitivity.iterrows():
        print(f"    Ratio {row['cost_ratio_fn_to_fp']:>6s} -> Optimal Threshold: {row['optimal_threshold']:.2f} | Recall: {row['recall_at_optimal']:.2f} | Savings vs tau=0.5: {row['savings_pct_vs_default']:.1f}%")

    # Nested Threshold Policy Evaluation
    policy_res = evaluate_threshold_policy_nested(
        y_true=y_train_arr,
        y_prob=chosen_oof_probs,
        objective="cost",
        cost_matrix=base_cost_matrix,
        n_splits=5,
        random_state=settings.random_state,
    )
    save_experiment_json(policy_res.model_dump(), "threshold_policy_evaluation.json")

    # Plot Threshold Curves
    print("  Generating Figure 24: Threshold Trade-off Curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision", color="#1f77b4", linewidth=2.0)
    ax.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall (Sensitivity)", color="#d9534f", linewidth=2.0)
    ax.plot(df_thresh["threshold"], df_thresh["f1"], label="F1-Score", color="#2ca02c", linewidth=2.0)
    ax.plot(df_thresh["threshold"], df_thresh["f2"], label="F2-Score (Recall Weighted)", color="#9467bd", linewidth=1.8, linestyle=":")
    ax.plot(df_thresh["threshold"], df_thresh["balanced_accuracy"], label="Balanced Accuracy", color="#ff7f0e", linewidth=1.8, linestyle="--")

    opt_t = policy_res.selected_threshold
    ax.axvline(x=0.50, color="#666666", linestyle="--", linewidth=1.2, label="Default Threshold (0.50)")
    ax.axvline(x=opt_t, color="#d9534f", linestyle="-.", linewidth=1.5, label=f"Cost-Optimal Threshold ({opt_t:.2f} for 5:1 Ratio)")

    ax.set_xlabel("Decision Threshold (τ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_title("Classification Performance vs. Decision Threshold (Training Out-of-Fold)", fontsize=12, fontweight="bold")
    ax.legend(loc="center right", frameon=True, fontsize=10)
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "24_threshold_precision_recall_f1.png", dpi=300)
    plt.close(fig)

    # Plot Cost Curve & Sensitivity
    print("  Generating Figure 25: Expected Cost Curve & Sensitivity...")
    fig, (ax_cost, ax_sens) = plt.subplots(1, 2, figsize=(14, 5))

    ax_cost.plot(df_thresh_cost["threshold"], df_thresh_cost["cost_per_customer"], color="#c9302c", linewidth=2.2, label="Cost per Customer (CFN=5, CFP=1)")
    ax_cost.axvline(x=0.50, color="#666666", linestyle="--", linewidth=1.2, label="Default (0.50)")
    ax_cost.axvline(x=opt_t, color="#2ca02c", linestyle="-.", linewidth=1.5, label=f"Cost-Minimizing (τ* = {opt_t:.2f})")
    ax_cost.set_xlabel("Decision Threshold (τ)", fontsize=11, fontweight="bold")
    ax_cost.set_ylabel("Expected Cost per Customer (Normalized)", fontsize=11, fontweight="bold")
    ax_cost.set_title("Expected Cost Curve (Asymmetric 5:1 Loss Ratio)", fontsize=12, fontweight="bold")
    ax_cost.legend(loc="upper right", frameon=True)

    ratios = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    opt_thresholds = df_cost_sensitivity["optimal_threshold"].tolist()
    ax_sens.plot(ratios, opt_thresholds, marker="s", markersize=7, linewidth=2.0, color="#3b6998")
    ax_sens.set_xlabel("Hypothetical Cost Ratio (Cost FN / Cost FP)", fontsize=11, fontweight="bold")
    ax_sens.set_ylabel("Optimal Decision Threshold (τ*)", fontsize=11, fontweight="bold")
    ax_sens.set_title("Threshold Sensitivity across Cost Ratios", fontsize=12, fontweight="bold")
    for r, th in zip(ratios, opt_thresholds, strict=True):
        ax_sens.annotate(f"{th:.2f}", (r, th), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(fig_dir / "25_cost_curve_and_sensitivity.png", dpi=300)
    plt.close(fig)

    # 5. Customer Churn Risk Tiers
    print("\n[Step 5/8] Implementing customer churn risk tiers...")
    risk_tierer = RiskTierer()
    df_tiers = risk_tierer.compute_tier_summary(chosen_oof_probs, y_true=y_train_arr)
    df_tiers.to_csv(exp_dir / "risk_tier_summary.csv", index=False)
    print("  Risk Tier Statistical Profile:")
    for _, r in df_tiers.iterrows():
        print(f"    Tier: {r['risk_tier']:<6s} | Prob Range: {r['probability_range']:<14s} | Accounts: {r['account_count']:4d} ({r['percent_of_cohort']:5.1f}%) | Observed Churn: {r['observed_churn_rate']:.4f}")

    # Plot Risk Tier Cohorts
    print("  Generating Figure 26: Risk Tier Distribution...")
    fig, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(13, 5))

    tier_colors = ["#5cb85c", "#f0ad4e", "#d9534f"]
    ax_t1.bar(df_tiers["risk_tier"], df_tiers["account_count"], color=tier_colors, alpha=0.85, edgecolor="#555555")
    for idx, count in enumerate(df_tiers["account_count"]):
        pct = df_tiers["percent_of_cohort"].iloc[idx]
        ax_t1.text(idx, count + 60, f"{count}\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    ax_t1.set_ylabel("Subscriber Count", fontsize=11, fontweight="bold")
    ax_t1.set_title("Customer Cohort Size by Risk Tier (N = 5,634)", fontsize=12, fontweight="bold")
    ax_t1.set_ylim(0, max(df_tiers["account_count"]) * 1.2)

    ax_t2.bar(df_tiers["risk_tier"], df_tiers["observed_churn_rate"] * 100, color=tier_colors, alpha=0.85, edgecolor="#555555")
    for idx, rate in enumerate(df_tiers["observed_churn_rate"]):
        ax_t2.text(idx, (rate * 100) + 2, f"{rate * 100:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax_t2.set_ylabel("Observed Churn Prevalence (%)", fontsize=11, fontweight="bold")
    ax_t2.set_title("Observed Churn Rate by Risk Tier", fontsize=12, fontweight="bold")
    ax_t2.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(fig_dir / "26_risk_tier_distribution.png", dpi=300)
    plt.close(fig)

    # 6. Global & Local Explainability
    print("\n[Step 6/8] Evaluating global permutation importance and local TreeSHAP explanations...")
    perm_res, df_perm = compute_training_permutation_importance(
        pipeline=base_pipeline.fit(X_train, y_train_arr),
        X_train=X_train,
        y_train=y_train_arr,
        scoring="average_precision",
        n_repeats=10,
        random_state=settings.random_state,
    )
    df_perm.to_csv(exp_dir / "permutation_importance_detailed.csv", index=False)
    save_experiment_json(perm_res.model_dump(), "permutation_importance_summary.json")

    # Fit final calibrated production model on full training data
    print("\n[Step 7/8] Fitting final production calibrated pipeline on full training data (N = 5,634)...")
    final_calibrated_pipeline = fit_final_calibrated_pipeline(
        base_pipeline=base_pipeline,
        selected_method=selected_calib_method,
        X_train=X_train,
        y_train=y_train_arr,
        inner_cv=5,
    )

    # Initialize SHAP explainer on final calibrated model
    explainer = TreeShapExplainer(final_calibrated_pipeline, feature_names=perm_res.feature_names)

    # Plot SHAP Global Feature Importance
    print("  Generating Figure 27: SHAP Global Feature Importance Summary...")
    fig, ax = plt.subplots(figsize=(10, 8))
    top15 = df_perm.iloc[:15]
    y_pos = np.arange(len(top15))
    ax.barh(y_pos, top15["importance_mean"], xerr=top15["importance_std"], align="center", color="#3b6998", alpha=0.85, capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15["clean_feature"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance: Mean Δ PR-AUC (± Std)", fontsize=11, fontweight="bold")
    ax.set_title("Top 15 Predictive Features (Permutation Importance on Training Partition)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "27_shap_global_summary.png", dpi=300)
    plt.close(fig)

    # Local Explanation on Representative Customers
    # Sample a high-risk churner and a low-risk retained customer
    churner_idx = np.where((chosen_oof_probs > 0.70) & (y_train_arr == 1))[0][0]
    retained_idx = np.where((chosen_oof_probs < 0.15) & (y_train_arr == 0))[0][0]

    sample_churner = X_train.iloc[churner_idx]
    sample_retained = X_train.iloc[retained_idx]

    expl_churner = explainer.explain_record(sample_churner, threshold=opt_t)
    expl_retained = explainer.explain_record(sample_retained, threshold=opt_t)

    save_experiment_json(
        {"sample_churner_explanation": expl_churner.model_dump(), "sample_retained_explanation": expl_retained.model_dump()},
        "local_explanations_examples.json",
    )

    # Plot Local Waterfall Attribution for Churner
    print("  Generating Figure 28: Local Waterfall Attribution Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    drivers = expl_churner.top_churn_drivers + expl_churner.top_retention_drivers
    drivers = sorted(drivers, key=lambda d: abs(d.shap_value), reverse=True)[:8]

    d_names = [d.clean_feature for d in drivers]
    d_vals = [d.shap_value for d in drivers]
    d_colors = ["#d9534f" if v > 0 else "#5cb85c" for v in d_vals]

    y_pos = np.arange(len(drivers))
    ax.barh(y_pos, d_vals, color=d_colors, alpha=0.85, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d_names, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Shapley Attribution Value (Log-Odds Contribution)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Local Feature Attribution (Customer Prediction: P(Churn)={expl_churner.model_estimated_probability:.2f}, {expl_churner.predicted_label})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "28_shap_local_waterfall.png", dpi=300)
    plt.close(fig)

    # 7. Persist Production Phase 5 Artifact
    print("\n[Step 8/8] Serializing production Phase 5 model artifact...")
    phase5_artifact_path = artifacts_dir / "model_phase5_calibrated.joblib"

    metadata = {
        "model_version": "5.0.0",
        "model_name": f"Hist Gradient Boosting ({selected_calib_method.capitalize()} Calibrated)",
        "dataset_sha256": manifest.raw_data_sha256,
        "split_provenance": "Canonical Phase 3 stratified holdout partition (80/20 train/test split, random_state=42, stratify=Churn) recorded in data/splits/split_manifest.json",
        "selected_estimator": "HistGradientBoostingClassifier",
        "hyperparameters": phase4_bundle["metadata"]["hyperparameters"],
        "preprocessing_identity": "baseline_tree (cleaning + OptionalServiceCount + column transformation)",
        "feature_count": 47,
        "feature_names": perm_res.feature_names,
        "calibration_method": selected_calib_method,
        "calibration_cv_config": "Nested 5-fold outer StratifiedKFold x 5-fold inner CalibratedClassifierCV",
        "calibration_metrics": calib_results[selected_calib_method][0].mean_metrics,
        "decision_threshold": opt_t,
        "default_threshold": 0.50,
        "risk_tier_boundaries": [t.model_dump() for t in risk_tierer.definitions],
        "training_and_validation_methodology": (
            "Outer Stratified 5-Fold Cross-Validation on training partition only. "
            "Holdout test partition (N=1,409) remained strictly protected and untouched during calibration, "
            "threshold tuning, and feature explanations."
        ),
        "environment": phase4_bundle["metadata"]["environment"],
        "known_limitations": (
            "1. Dataset is a cross-sectional snapshot; predictions represent retrospective risk ranking, not temporal event forecasting. "
            "2. Probabilities are empirically calibrated using Platt/Isotonic methods on training data, but perfect calibration cannot be claimed. "
            "3. Permutation importances and local SHAP values measure model reliance and margin attribution, not causal intervention effects."
        ),
    }

    save_model_artifact(final_calibrated_pipeline, metadata, file_path=phase5_artifact_path)
    print(f"  Phase 5 Calibrated Artifact successfully saved: {phase5_artifact_path.resolve()}")

    # 8. Test Production Inference Engine
    print("\nVerifying Inference Engine with serialized Phase 5 artifact...")
    engine = InferenceEngine(model_artifact_path=phase5_artifact_path, decision_threshold=opt_t)
    sample_dict = sample_churner.to_dict()
    pred_res = engine.predict_single(sample_dict, include_explanation=True)
    print(f"  Single Prediction Test -> Prob: {pred_res.estimated_probability:.4f} | Class: {pred_res.predicted_class} ({pred_res.predicted_label}) | Tier: {pred_res.risk_tier}")

    # Test Batch Inference
    batch_records = [sample_churner.to_dict(), sample_retained.to_dict()]
    batch_res = engine.predict_batch(batch_records)
    print(f"  Batch Prediction Test  -> Total: {batch_res.total_records} | Success: {batch_res.successful_predictions} | Failed: {batch_res.failed_predictions}")

    print("\n=======================================================")
    print(" PHASE 5 EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY")
    print("=======================================================")


if __name__ == "__main__":
    main()

