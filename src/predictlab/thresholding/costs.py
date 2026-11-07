"""Cost-sensitive classification framework and sensitivity analysis across hypothetical cost ratios."""

import pandas as pd
from pydantic import BaseModel, Field


class CostMatrix(BaseModel):
    """Configurable cost matrix specifying asymmetric misclassification costs and intervention benefits.

    Methodological Warning:
    Specific numerical costs depend on company-specific financial realities (customer lifetime value,
    retention discount costs, outreach labor). Costs provided here serve as configurable parameters
    or hypothetical scenarios, not universally validated empirical facts.
    """

    cost_fp: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost incurred per False Positive (e.g., outreach cost or discount to retained customer)",
    )
    cost_fn: float = Field(
        default=5.0,
        ge=0.0,
        description="Cost incurred per False Negative (e.g., lost gross margin / customer lifetime value)",
    )
    benefit_tp: float = Field(
        default=0.0,
        ge=0.0,
        description="Net benefit gained per successfully retained True Positive",
    )
    benefit_tn: float = Field(
        default=0.0,
        ge=0.0,
        description="Benefit gained for correctly identifying retained customer (typically 0.0)",
    )

    def compute_total_cost(self, tp: int, fp: int, tn: int, fn: int) -> float:
        """Calculate total net cost under this cost matrix."""
        cost = (fp * self.cost_fp) + (fn * self.cost_fn)
        benefit = (tp * self.benefit_tp) + (tn * self.benefit_tn)
        return float(cost - benefit)


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
    return df


def run_cost_ratio_sensitivity(
    df_metrics: pd.DataFrame,
    cost_ratios: list[float] | None = None,
) -> pd.DataFrame:
    """Perform sensitivity analysis across clearly labeled hypothetical cost ratios (C_FN / C_FP).

    Fixes cost_fp = 1.0 (normalized unit cost of outreach/incentive) and tests variable
    costs of missed churners (cost_fn).

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Threshold grid metrics table.
    cost_ratios : list of float, optional
        Hypothetical cost ratios C_FN / C_FP. Defaults to [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0].

    Returns
    -------
    pd.DataFrame
        Table documenting optimal threshold, recall, precision, and savings across ratios.
    """
    if cost_ratios is None:
        cost_ratios = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

    records = []
    for ratio in cost_ratios:
        cm = CostMatrix(cost_fp=1.0, cost_fn=float(ratio))
        df_cost = evaluate_cost_sensitive_thresholds(df_metrics, cm)
        best_row = df_cost.sort_values(by="total_cost", ascending=True).iloc[0]

        default_row = df_cost.iloc[(df_cost["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]
        cost_delta = float(default_row["total_cost"] - best_row["total_cost"])
        pct_savings = (cost_delta / default_row["total_cost"] * 100) if default_row["total_cost"] > 0 else 0.0

        records.append(
            {
                "cost_ratio_fn_to_fp": f"{ratio:.1f}:1",
                "optimal_threshold": float(best_row["threshold"]),
                "recall_at_optimal": float(best_row["recall"]),
                "precision_at_optimal": float(best_row["precision"]),
                "f1_at_optimal": float(best_row["f1"]),
                "min_cost_per_customer": float(best_row["cost_per_customer"]),
                "default_cost_per_customer": float(default_row["cost_per_customer"]),
                "savings_pct_vs_default": round(pct_savings, 2),
            }
        )

    return pd.DataFrame(records)

