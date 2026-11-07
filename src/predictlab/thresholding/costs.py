"""Cost-sensitive classification framework and sensitivity analysis across hypothetical cost ratios."""

import pandas as pd
from pydantic import BaseModel, Field


class CostMatrix(BaseModel):
    """Configurable cost matrix specifying asymmetric costs across all four classification outcomes.

    Explicit mathematical formula:
    TotalCost = C_FP * FP + C_FN * FN + C_TP * TP + C_TN * TN

    Methodological & Governance Warnings:
    1. Numerical cost values (e.g. C_FP=$100, C_FN=$500, C_TP=$100, C_TN=$0) represent
       hypothetical scenario parameters for sensitivity exploration, NOT empirically validated
       or accounting-audited business economics. No operational enterprise retention data
       have been independently established.
    2. Any optimal threshold derived from this matrix depends directly on the chosen ratio
       and assumptions, and must be treated as conditional and illustrative.
    """

    cost_fp: float = Field(
        default=100.0,
        ge=0.0,
        description="Cost incurred per False Positive (e.g. outreach or concession to a retained customer)",
    )
    cost_fn: float = Field(
        default=500.0,
        ge=0.0,
        description="Cost incurred per False Negative (e.g. unmitigated lost customer lifetime value)",
    )
    cost_tp: float = Field(
        default=100.0,
        ge=0.0,
        description="Cost incurred per True Positive (e.g. retention concession delivered to an at-risk subscriber)",
    )
    cost_tn: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost incurred per True Negative (uncontacted retained subscriber; typically 0.0)",
    )
    scenario_name: str = Field(
        default="comprehensive_incentive",
        description="Descriptive identifier for the cost scenario",
    )

    def compute_total_cost(self, tp: int, fp: int, tn: int, fn: int) -> float:
        """Calculate total net cost: TotalCost = C_FP*FP + C_FN*FN + C_TP*TP + C_TN*TN."""
        return float(
            (fp * self.cost_fp)
            + (fn * self.cost_fn)
            + (tp * self.cost_tp)
            + (tn * self.cost_tn)
        )

    @classmethod
    def comprehensive_incentive(
        cls,
        cost_fp: float = 100.0,
        cost_fn: float = 500.0,
        cost_tp: float = 100.0,
        cost_tn: float = 0.0,
    ) -> "CostMatrix":
        """Scenario where every intervention (TP and FP) incurs concession cost."""
        return cls(
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            cost_tp=cost_tp,
            cost_tn=cost_tn,
            scenario_name="comprehensive_incentive",
        )

    @classmethod
    def misclassification_penalty_only(
        cls,
        cost_fp: float = 100.0,
        cost_fn: float = 500.0,
    ) -> "CostMatrix":
        """Scenario penalizing only classification errors (FP and FN) with C_TP=0 and C_TN=0."""
        return cls(
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            cost_tp=0.0,
            cost_tn=0.0,
            scenario_name="misclassification_penalty_only",
        )


def evaluate_cost_sensitive_thresholds(
    df_metrics: pd.DataFrame,
    cost_matrix: CostMatrix,
) -> pd.DataFrame:
    """Augment threshold metrics table with expected cost and net benefit per customer.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Threshold grid containing 'true_positives', 'false_positives', 'true_negatives', 'false_negatives'.
    cost_matrix : CostMatrix
        Configured cost matrix.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'total_cost', 'cost_per_customer', 'cost_savings_vs_default'.
    """
    df = df_metrics.copy()
    n_total = (
        df["true_positives"].iloc[0]
        + df["false_positives"].iloc[0]
        + df["true_negatives"].iloc[0]
        + df["false_negatives"].iloc[0]
    )

    costs = []
    for _, row in df.iterrows():
        total_c = cost_matrix.compute_total_cost(
            tp=int(row["true_positives"]),
            fp=int(row["false_positives"]),
            tn=int(row["true_negatives"]),
            fn=int(row["false_negatives"]),
        )
        costs.append(total_c)

    df["total_cost"] = [round(c, 2) for c in costs]
    df["cost_per_customer"] = [round(c / n_total, 4) for c in costs]

    # Reference cost at default threshold 0.50
    default_row = df.iloc[(df["threshold"] - 0.50).abs().argsort()[:1]]
    default_cost = float(default_row["total_cost"].iloc[0])

    df["cost_savings_vs_default"] = [round(default_cost - c, 2) for c in df["total_cost"]]
    df["savings_pct_vs_default"] = [
        round((default_cost - c) / default_cost * 100, 2) if default_cost > 0 else 0.0
        for c in df["total_cost"]
    ]
    return df


def run_cost_ratio_sensitivity(
    df_metrics: pd.DataFrame,
    cost_ratios: list[float] | None = None,
    cost_fp: float = 100.0,
    cost_tp: float = 100.0,
    cost_tn: float = 0.0,
    scenario_label: str = "comprehensive_incentive",
) -> pd.DataFrame:
    """Perform sensitivity analysis across clearly labeled hypothetical cost ratios (C_FN / C_FP).

    Methodological Warning:
    Cost ratios (e.g., C_FN:C_FP from 0.2:1 to 10:1) represent hypothetical scenario parameters,
    NOT empirical enterprise financial facts. The derived optimal thresholds illustrate sensitivity
    to business trade-offs; they do NOT establish validated commercial retention policies.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Threshold grid metrics table.
    cost_ratios : list of float, optional
        Hypothetical cost ratios C_FN / C_FP. Defaults to [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0].
    cost_fp : float, default=100.0
        Base hypothetical cost per False Positive.
    cost_tp : float, default=100.0
        Hypothetical cost per True Positive. Set to 0.0 for misclassification-penalty-only scenarios.
    cost_tn : float, default=0.0
        Hypothetical cost per True Negative.
    scenario_label : str, default='comprehensive_incentive'
        Descriptive scenario tag.

    Returns
    -------
    pd.DataFrame
        Table documenting optimal threshold, recall, precision, and savings across ratios.
    """
    if cost_ratios is None:
        cost_ratios = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

    records = []
    for ratio in cost_ratios:
        cost_fn = float(cost_fp * ratio)
        cm = CostMatrix(
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            cost_tp=cost_tp,
            cost_tn=cost_tn,
            scenario_name=f"{scenario_label}_{ratio:.1f}_to_1",
        )
        df_cost = evaluate_cost_sensitive_thresholds(df_metrics, cm)
        best_row = df_cost.sort_values(by="total_cost", ascending=True).iloc[0]

        default_row = df_cost.iloc[(df_cost["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]
        cost_delta = float(default_row["total_cost"] - best_row["total_cost"])
        pct_savings = (cost_delta / default_row["total_cost"] * 100) if default_row["total_cost"] > 0 else 0.0

        records.append(
            {
                "cost_ratio_fn_to_fp": f"{ratio:.1f}:1",
                "cost_fn": cost_fn,
                "cost_fp": cost_fp,
                "cost_tp": cost_tp,
                "optimal_threshold": float(best_row["threshold"]),
                "recall_at_optimal": float(best_row["recall"]),
                "precision_at_optimal": float(best_row["precision"]),
                "f1_at_optimal": float(best_row["f1"]),
                "min_total_cost": float(best_row["total_cost"]),
                "min_cost_per_customer": float(best_row["cost_per_customer"]),
                "default_total_cost": float(default_row["total_cost"]),
                "default_cost_per_customer": float(default_row["cost_per_customer"]),
                "cost_savings_dollars": round(cost_delta, 2),
                "savings_pct_vs_default": round(pct_savings, 2),
                "scenario_name": scenario_label,
            }
        )

    return pd.DataFrame(records)

