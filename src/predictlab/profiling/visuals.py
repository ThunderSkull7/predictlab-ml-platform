"""Publication-grade visualization generation module for exploratory data analysis."""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from predictlab.logging_config import get_logger
from predictlab.profiling.insights import (
    compute_categorical_associations,
    compute_correlation_matrix,
    compute_segment_churn_rates,
)

logger = get_logger(__name__)

# Standard styling configuration
STYLE_CONFIG = {
    "palette_binary": ["#1E3A8A", "#EF4444"],  # Retained (Dark Blue), Churned (Red/Coral)
    "palette_multi": ["#2563EB", "#0D9488", "#F59E0B", "#DC2626", "#8B5CF6"],
    "dpi": 300,
}


def _apply_plot_style() -> None:
    """Apply consistent styling across matplotlib figures."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }
    )


def generate_all_eda_figures(df: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """Generate the complete set of 16 structured, publication-grade EDA figures.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or loaded Telco dataset.
    output_dir : Union[Path, str]
        Directory path where images will be saved.

    Returns
    -------
    list[Path]
        List of generated image file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()

    # Pre-parse numeric TotalCharges cleanly for plotting
    df_plot = df.copy()
    df_plot["TotalCharges_num"] = pd.to_numeric(
        df_plot["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
    )

    created_files: list[Path] = []

    # 1. Target Class Distribution
    p1 = out_path / "01_target_distribution.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df_plot["Churn"].value_counts()
    pcts = df_plot["Churn"].value_counts(normalize=True) * 100
    bars = ax.bar(["Retained (No)", "Churned (Yes)"], counts.values, color=["#1E3A8A", "#EF4444"], width=0.5)
    for bar, pct in zip(bars, pcts.values, strict=False):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 80,
            f"{int(yval):,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title("Ground-Truth Customer Churn Distribution (Target Variable)")
    ax.set_ylabel("Customer Count")
    ax.set_ylim(0, max(counts.values) * 1.15)
    fig.savefig(p1, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p1)

    # 2. Numerical Distributions
    p2 = out_path / "02_numeric_distributions.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    num_cols = [("tenure", "Tenure (Months)"), ("MonthlyCharges", "Monthly Charges ($)"), ("TotalCharges_num", "Total Charges ($)")]
    for ax, (col, label) in zip(axes, num_cols, strict=False):
        sns.histplot(df_plot[col].dropna(), kde=True, ax=ax, color="#2563EB", bins=30)
        ax.set_title(f"Distribution of {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
    fig.savefig(p2, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p2)

    # 3. Tenure vs Churn
    p3 = out_path / "03_tenure_vs_churn.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(data=df_plot, x="tenure", hue="Churn", common_norm=False, fill=True, palette={"No": "#1E3A8A", "Yes": "#EF4444"}, ax=ax1)
    ax1.set_title("Tenure Density Conditioned on Churn")
    ax1.set_xlabel("Tenure (Months)")
    # Tenure buckets churn rate
    df_plot["tenure_group"] = pd.cut(
        df_plot["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"],
    )
    t_rate = compute_segment_churn_rates(df_plot, group_cols="tenure_group").sort_values(by="tenure_group")
    bars = ax2.bar(t_rate["tenure_group"], t_rate["churn_rate_pct"], color="#EF4444", width=0.55)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax2.set_title("Observed Churn Rate by Tenure Cohort")
    ax2.set_ylabel("Churn Rate (%)")
    ax2.set_xlabel("Tenure Cohort")
    ax2.set_ylim(0, 60)
    fig.savefig(p3, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p3)

    # 4. Monthly Charges vs Churn
    p4 = out_path / "04_monthly_charges_vs_churn.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(
        data=df_plot,
        x="Churn",
        y="MonthlyCharges",
        hue="Churn",
        palette={"No": "#1E3A8A", "Yes": "#EF4444"},
        ax=ax1,
        width=0.4,
        legend=False,
    )
    ax1.set_title("Monthly Charges Distribution by Churn Status")
    ax1.set_ylabel("Monthly Charges ($)")
    sns.kdeplot(data=df_plot, x="MonthlyCharges", hue="Churn", common_norm=False, fill=True, palette={"No": "#1E3A8A", "Yes": "#EF4444"}, ax=ax2)
    ax2.set_title("Monthly Charges Density Profile")
    ax2.set_xlabel("Monthly Charges ($)")
    fig.savefig(p4, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p4)

    # 5. Total Charges vs Churn
    p5 = out_path / "05_total_charges_vs_churn.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(
        data=df_plot.dropna(subset=["TotalCharges_num"]),
        x="Churn",
        y="TotalCharges_num",
        hue="Churn",
        palette={"No": "#1E3A8A", "Yes": "#EF4444"},
        ax=ax1,
        width=0.4,
        legend=False,
    )
    ax1.set_title("Total Charges Distribution by Churn Status")
    ax1.set_ylabel("Total Cumulative Charges ($)")
    sns.kdeplot(data=df_plot.dropna(subset=["TotalCharges_num"]), x="TotalCharges_num", hue="Churn", common_norm=False, fill=True, palette={"No": "#1E3A8A", "Yes": "#EF4444"}, ax=ax2)
    ax2.set_title("Total Cumulative Charges Density Profile")
    ax2.set_xlabel("Total Cumulative Charges ($)")
    fig.savefig(p5, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p5)

    # 6. Churn by Contract
    p6 = out_path / "06_churn_by_contract.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    c_rates = compute_segment_churn_rates(df_plot, group_cols="Contract").sort_values(by="total_customers", ascending=False)
    ax1.bar(c_rates["Contract"], c_rates["total_customers"], color="#1E3A8A", width=0.5)
    for idx, (_x, y) in enumerate(zip(c_rates["Contract"], c_rates["total_customers"], strict=False)):
        ax1.text(idx, y + 50, f"{y:,}", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Total Customers by Contract Type")
    ax1.set_ylabel("Customer Count")
    ax1.set_ylim(0, max(c_rates["total_customers"]) * 1.15)
    bars2 = ax2.bar(c_rates["Contract"], c_rates["churn_rate_pct"], color="#EF4444", width=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax2.set_title("Observed Churn Rate by Contract Type")
    ax2.set_ylabel("Churn Rate (%)")
    ax2.set_ylim(0, 50)
    fig.savefig(p6, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p6)

    # 7. Churn by Payment Method
    p7 = out_path / "07_churn_by_payment_method.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    pm_rates = compute_segment_churn_rates(df_plot, group_cols="PaymentMethod")
    bars = ax.barh(pm_rates["PaymentMethod"], pm_rates["churn_rate_pct"], color="#DC2626", height=0.55)
    ax.axvline(26.54, color="gray", linestyle="--", label="Overall Baseline Churn Rate (26.5%)")
    for bar in bars:
        xval = bar.get_width()
        ax.text(xval + 1, bar.get_y() + bar.get_height() / 2, f"{xval:.1f}%", va="center", fontweight="bold")
    ax.set_title("Observed Churn Rate by Payment Method")
    ax.set_xlabel("Churn Rate (%)")
    ax.set_xlim(0, 55)
    ax.legend(loc="lower right")
    fig.savefig(p7, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p7)

    # 8. Churn by Internet Service
    p8 = out_path / "08_churn_by_internet_service.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    is_rates = compute_segment_churn_rates(df_plot, group_cols="InternetService")
    ax1.bar(is_rates["InternetService"], is_rates["total_customers"], color="#2563EB", width=0.5)
    ax1.set_title("Customer Volume by Internet Service")
    ax1.set_ylabel("Customer Count")
    for idx, (_x, y) in enumerate(zip(is_rates["InternetService"], is_rates["total_customers"], strict=False)):
        ax1.text(idx, y + 40, f"{y:,}", ha="center", va="bottom", fontweight="bold")
    bars = ax2.bar(is_rates["InternetService"], is_rates["churn_rate_pct"], color="#EF4444", width=0.5)
    ax2.axhline(26.54, color="gray", linestyle="--", label="Baseline (26.5%)")
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax2.set_title("Observed Churn Rate by Internet Service")
    ax2.set_ylabel("Churn Rate (%)")
    ax2.set_ylim(0, 50)
    ax2.legend()
    fig.savefig(p8, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p8)

    # 9. Churn by Tech Support
    p9 = out_path / "09_churn_by_tech_support.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ts_rates = compute_segment_churn_rates(df_plot, group_cols="TechSupport")
    bars = ax.bar(ts_rates["TechSupport"], ts_rates["churn_rate_pct"], color=["#EF4444", "#3B82F6", "#10B981"], width=0.5)
    ax.axhline(26.54, color="gray", linestyle="--", label="Baseline (26.5%)")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Observed Churn Rate by Tech Support Status")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 50)
    ax.legend()
    fig.savefig(p9, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p9)

    # 10. Churn by Online Security
    p10 = out_path / "10_churn_by_online_security.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    os_rates = compute_segment_churn_rates(df_plot, group_cols="OnlineSecurity")
    bars = ax.bar(os_rates["OnlineSecurity"], os_rates["churn_rate_pct"], color=["#EF4444", "#3B82F6", "#10B981"], width=0.5)
    ax.axhline(26.54, color="gray", linestyle="--", label="Baseline (26.5%)")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Observed Churn Rate by Online Security Status")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 50)
    ax.legend()
    fig.savefig(p10, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p10)

    # 11. Churn by Demographics
    p11 = out_path / "11_churn_by_demographics.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    demo_features = [("SeniorCitizen", "Senior Citizen (0=No, 1=Yes)"), ("Partner", "Partner Status"), ("Dependents", "Dependents Status")]
    for ax, (col, title) in zip(axes, demo_features, strict=False):
        d_rates = compute_segment_churn_rates(df_plot, group_cols=col)
        bars = ax.bar([str(x) for x in d_rates[col]], d_rates["churn_rate_pct"], color=["#3B82F6", "#EF4444"], width=0.5)
        ax.axhline(26.54, color="gray", linestyle="--", alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
        ax.set_title(title)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, 50)
    fig.savefig(p11, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p11)

    # 12. Multivariate: Internet Service and Tech Support Interaction
    p12 = out_path / "12_multivariate_internet_and_support.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    is_ts_rates = compute_segment_churn_rates(df_plot, group_cols=["InternetService", "TechSupport"])
    # Pivot for clustered bar chart
    pivot_rates = is_ts_rates.pivot(index="InternetService", columns="TechSupport", values="churn_rate_pct")
    pivot_rates.plot(kind="bar", ax=ax, colormap="viridis", width=0.7)
    ax.axhline(26.54, color="red", linestyle="--", label="Baseline Churn Rate (26.5%)")
    ax.set_title("Multivariate Interaction: Internet Service & Tech Support on Churn Rate")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Internet Service Type")
    ax.set_ylim(0, 60)
    ax.legend(title="Tech Support Status")
    fig.savefig(p12, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p12)

    # 13. Multivariate: Contract and Tenure Cohort Interaction
    p13 = out_path / "13_multivariate_contract_and_tenure.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ct_rates = compute_segment_churn_rates(df_plot, group_cols=["Contract", "tenure_group"])
    ct_pivot = ct_rates.pivot(index="tenure_group", columns="Contract", values="churn_rate_pct")
    ct_pivot.plot(kind="line", marker="o", linewidth=2.5, ax=ax, color=["#EF4444", "#3B82F6", "#10B981"])
    ax.axhline(26.54, color="gray", linestyle="--", label="Overall Baseline (26.5%)")
    ax.set_title("Cross-Sectional Churn Prevalence by Contract Across Tenure Cohorts")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Tenure Cohort")
    ax.set_ylim(0, 65)
    ax.legend(title="Contract Type")
    fig.savefig(p13, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p13)

    # 14. TotalCharges vs Tenure Scatter Investigation (Highlighting the 11 Anomalies)
    p14 = out_path / "14_total_charges_tenure_investigation.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_mask = df_plot["TotalCharges_num"].notna()
    # Plot sample of normal points
    ax.scatter(df_plot.loc[valid_mask, "tenure"], df_plot.loc[valid_mask, "TotalCharges_num"], alpha=0.2, color="#2563EB", s=15, label="Standard Accounts (Valid Charges)")
    # Highlight the 11 whitespace records at tenure==0
    ws_mask = df_plot["TotalCharges_num"].isna()
    ax.scatter(df_plot.loc[ws_mask, "tenure"], [0] * int(ws_mask.sum()), color="#DC2626", s=120, marker="X", zorder=5, label="11 Whitespace Anomalies at Tenure=0 (Proposed = $0.00)")
    ax.set_title("Total Charges vs. Tenure Relationship & Whitespace Anomaly Audit")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Total Cumulative Charges ($)")
    ax.legend(loc="upper left")
    fig.savefig(p14, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p14)

    # 15. Numeric Correlation Matrix Heatmap
    p15 = out_path / "15_numeric_correlation_matrix.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    s_corr = compute_correlation_matrix(df_plot, num_cols=["tenure", "MonthlyCharges", "TotalCharges_num"], method="spearman")
    p_corr = compute_correlation_matrix(df_plot, num_cols=["tenure", "MonthlyCharges", "TotalCharges_num"], method="pearson")
    sns.heatmap(s_corr, annot=True, cmap="Blues", fmt=".2f", vmin=-1, vmax=1, ax=ax1, cbar=False)
    ax1.set_title("Spearman Rank Correlation Matrix")
    sns.heatmap(p_corr, annot=True, cmap="Blues", fmt=".2f", vmin=-1, vmax=1, ax=ax2)
    ax2.set_title("Pearson Linear Correlation Matrix")
    fig.savefig(p15, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p15)

    # 16. Categorical Associations (Cramér's V Ranking)
    p16 = out_path / "16_categorical_associations_cramers_v.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    all_cats = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    ]
    assoc_df = compute_categorical_associations(df_plot, cat_cols=all_cats).sort_values(by="cramers_v", ascending=True)
    bars = ax.barh(assoc_df["feature"], assoc_df["cramers_v"], color="#1E3A8A", height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Strength of Categorical Association with Churn (Bias-Corrected Cramér's V)")
    ax.set_xlabel("Bias-Corrected Cramér's V [0.0 - 1.0]")
    ax.set_xlim(0, max(assoc_df["cramers_v"]) * 1.15)
    fig.savefig(p16, dpi=STYLE_CONFIG["dpi"])
    plt.close(fig)
    created_files.append(p16)

    logger.info("Successfully generated %d figures in %s", len(created_files), out_path)
    return created_files
