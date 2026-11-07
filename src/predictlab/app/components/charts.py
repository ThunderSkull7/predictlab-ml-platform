"""Interactive visualization components using Altair for PredictLab dashboard."""

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


def render_shap_attribution_chart(explanation_dict: dict[str, Any], max_features: int = 10) -> None:
    """Render horizontal bar chart of local TreeSHAP feature attributions on the base decision margin."""
    if not explanation_dict or "all_attributions" not in explanation_dict:
        st.info("Local feature attribution is not available for this record.")
        return

    attributions = explanation_dict.get("all_attributions", [])
    if not attributions:
        st.info("No feature attributions generated.")
        return

    # Sort attributions by absolute magnitude
    sorted_attr = sorted(attributions, key=lambda x: abs(x["shap_value"]), reverse=True)[:max_features]
    df = pd.DataFrame(
        [
            {
                "Feature": item["clean_feature"],
                "Contribution": item["shap_value"],
                "Impact": "Increases Churn Risk" if item["shap_value"] > 0 else "Decreases Churn Risk",
                "Value": str(item["feature_value"]),
            }
            for item in sorted_attr
        ]
    )

    # Sort descending for display
    df = df.iloc[::-1]

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Contribution:Q",
                title="SHAP Value (Contribution to Log-Odds Base Decision Margin)",
                axis=alt.Axis(grid=True, gridColor="#334155"),
            ),
            y=alt.Y(
                "Feature:N",
                sort=None,
                title="Predictor Feature",
                axis=alt.Axis(labelLimit=250),
            ),
            color=alt.Color(
                "Impact:N",
                scale=alt.Scale(
                    domain=["Increases Churn Risk", "Decreases Churn Risk"],
                    range=["#F43F5E", "#38BDF8"],
                ),
                legend=alt.Legend(title="Direction of Impact", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Feature:N", title="Feature"),
                alt.Tooltip("Contribution:Q", format="+.4f", title="SHAP Value"),
                alt.Tooltip("Impact:N", title="Impact"),
            ],
        )
        .properties(height=320)
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, use_container_width=True)


def render_risk_distribution_chart(df_risk_tiers: pd.DataFrame) -> None:
    """Render interactive risk tier distribution bar chart."""
    if df_risk_tiers.empty:
        return

    df = df_risk_tiers.copy()

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X(
                "risk_tier:N",
                title="Risk Tier Queue",
                sort=["Low", "Medium", "High"],
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "account_count:Q",
                title="Customer Accounts",
                axis=alt.Axis(grid=True, gridColor="#334155"),
            ),
            color=alt.Color(
                "risk_tier:N",
                scale=alt.Scale(
                    domain=["Low", "Medium", "High"],
                    range=["#38BDF8", "#FBBF24", "#F43F5E"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("risk_tier:N", title="Risk Tier"),
                alt.Tooltip("probability_range:N", title="Probability Range"),
                alt.Tooltip("account_count:Q", format=",.0f", title="Accounts"),
                alt.Tooltip("percent_of_cohort:Q", format=".1f", title="Cohort %"),
                alt.Tooltip("observed_churn_rate:Q", format=".1%", title="Observed Churn Rate"),
            ],
        )
        .properties(height=260)
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, use_container_width=True)


def render_batch_risk_donut_chart(tier_counts: dict[str, int]) -> None:
    """Render donut chart for batch prediction risk tier breakdown."""
    data = [
        {"Tier": tier, "Count": count}
        for tier, count in tier_counts.items()
        if count > 0
    ]
    if not data:
        return

    df = pd.DataFrame(data)

    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=45, cornerRadius=3)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color(
                "Tier:N",
                scale=alt.Scale(
                    domain=["Low", "Medium", "High"],
                    range=["#38BDF8", "#FBBF24", "#F43F5E"],
                ),
                legend=alt.Legend(title="Risk Tier", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Tier:N", title="Risk Tier"),
                alt.Tooltip("Count:Q", format=",.0f", title="Customers"),
            ],
        )
        .properties(height=220)
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, use_container_width=True)


def render_permutation_importance_chart(df_importance: pd.DataFrame, top_n: int = 15) -> None:
    """Render horizontal bar chart of cross-validated permutation importance."""
    if df_importance.empty:
        st.info("Permutation importance data is not available.")
        return

    df = df_importance.head(top_n).copy()
    # Format clean feature names if available
    df["display_name"] = df["feature"].str.replace("cat__", "").str.replace("num__", "").str.replace("_", " ")

    # Invert for vertical display order
    df = df.iloc[::-1]

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, color="#3B82F6")
        .encode(
            x=alt.X(
                "importance_mean:Q",
                title="Mean Drop in Average Precision (Cross-Validated)",
                axis=alt.Axis(grid=True, gridColor="#334155"),
            ),
            y=alt.Y(
                "display_name:N",
                sort=None,
                title="Feature",
                axis=alt.Axis(labelLimit=250),
            ),
            tooltip=[
                alt.Tooltip("display_name:N", title="Feature"),
                alt.Tooltip("importance_mean:Q", format=".4f", title="Mean Drop (AP)"),
                alt.Tooltip("importance_std:Q", format=".4f", title="Std Dev"),
                alt.Tooltip("rank:Q", title="Rank"),
            ],
        )
        .properties(height=380)
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, use_container_width=True)


def render_contract_churn_chart() -> None:
    """Render Phase 2 contract duration vs churn rate relationship."""
    df = pd.DataFrame(
        [
            {"Contract": "Month-to-month", "ChurnRate": 0.427, "Accounts": 3875},
            {"Contract": "One year", "ChurnRate": 0.113, "Accounts": 1473},
            {"Contract": "Two year", "ChurnRate": 0.028, "Accounts": 1695},
        ]
    )

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Contract:N", title="Contract Term", sort=["Month-to-month", "One year", "Two year"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("ChurnRate:Q", title="Observed Churn Rate", axis=alt.Axis(format=".0%", grid=True, gridColor="#334155")),
            color=alt.Color(
                "ChurnRate:Q",
                scale=alt.Scale(scheme="redyellowblue", reverse=True),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Contract:N", title="Contract"),
                alt.Tooltip("ChurnRate:Q", format=".1%", title="Churn Rate"),
                alt.Tooltip("Accounts:Q", format=",.0f", title="Total Customers"),
            ],
        )
        .properties(height=260)
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, use_container_width=True)
