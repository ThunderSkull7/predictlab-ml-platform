"""Page 4: Model Insights & Explainability for PredictLab Platform."""

import streamlit as st

from predictlab.app.components.badges import render_decision_badge, render_risk_tier_badge
from predictlab.app.components.cards import render_callout, render_metric_card
from predictlab.app.components.charts import render_permutation_importance_chart
from predictlab.app.services.data_service import (
    load_permutation_importance_data,
    load_risk_tier_summary,
)
from predictlab.app.services.inference_service import get_inference_engine
from predictlab.app.utils.formatting import format_probability


def render_explainability_page() -> None:
    """Render global permutation feature importance, TreeSHAP methodology, and risk tier calibration."""
    st.title("Model Insights & Explainability Architecture")
    st.markdown(
        """
        Understand the statistical mechanisms governing PredictLab predictions.
        This section communicates **global feature sensitivity**, the **local TreeSHAP attribution framework**,
        and the mathematical rationale behind **calibrated operational risk tiers**.
        """
    )

    try:
        engine = get_inference_engine()
        threshold = engine.decision_threshold
    except Exception:
        threshold = 0.19

    # 1. Global Feature Importance (Cross-Validated Permutation Importance)
    st.markdown('<div class="section-header"><h3>1. Global Predictive Importance (Permutation Importance)</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
        Global feature importance was evaluated via **5-fold cross-validation on held-out validation folds**
        using the primary model-selection metric: **Average Precision (PR-AUC)**.
        In each fold, predictor columns were permuted across 5 randomized repeats, measuring the resultant degradation in predictive score.
        """
    )

    df_importance = load_permutation_importance_data()
    top_n = st.slider("Select Number of Top Features to Display:", min_value=5, max_value=25, value=12, step=1)

    render_permutation_importance_chart(df_importance, top_n=top_n)

    render_callout(
        "<strong>Scientific Meaning & Collinearity Limitations:</strong><br>"
        "1. <strong>Measures Predictive Sensitivity:</strong> Permutation importance quantifies how heavily the model relies on a feature's statistical signal to distinguish churners. It <em>does not establish real-world causality</em>.<br>"
        "2. <strong>Collinear Dilution:</strong> When multiple predictors share statistical information (e.g. <code>tenure</code>, <code>TotalCharges</code>, and <code>MonthlyCharges</code>), permuting one predictor allows the tree ensemble to partially compensate using collinear counterparts, potentially understating individual importance values.<br>"
        "3. <strong>Precomputed Integrity:</strong> Global importance is read from precomputed validated artifacts and is never re-calculated on the fly during user sessions.",
        title="Methodological Disclosures on Global Importance",
        variant="warning",
    )

    # 2. Local Explainability Framework (TreeSHAP)
    st.markdown('<div class="section-header"><h3>2. Local Explainability Framework (TreeSHAP)</h3></div>', unsafe_allow_html=True)
    st.markdown("##### Local explanation of the model's averaged base decision margin")
    st.markdown(
        """
        PredictLab uses **TreeSHAP** (Lundberg et al., 2020) to explain individual customer predictions across
        the 5-fold ensemble of `HistGradientBoostingClassifier` estimators.
        """
    )

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        render_metric_card("Explained Target", "Decision Margin", "Log-odds space across 5 estimators", "neutral")
    with mcol2:
        render_metric_card("Ensemble Size", "5 Base Trees", "Averaged Shapley attributions", "neutral")
    with mcol3:
        render_metric_card("Additive Efficiency", "2.2 × 10⁻¹⁶", "Exact machine precision additivity", "positive")

    st.markdown(
        """
        #### Theoretical Properties of PredictLab Local Explanations
        1. **Margin Space vs Probability Space**:
           The local explanation operates strictly in the **averaged base model decision margin** (log-odds space before Sigmoid scaling).
           Because the subsequent Sigmoid (Platt) transformation is non-linear ($p = \\sigma(Af + B)$), raw SHAP values **must not be interpreted as percentage increases in churn probability**.
        2. **Exact Additivity Guarantee**:
           For every scored account, the sum of all 47 feature SHAP values plus the base expected value equals the ensemble output margin exactly:
           $$\\text{Margin}(\\mathbf{x}) = \\phi_0 + \\sum_{j=1}^{47} \\phi_j(\\mathbf{x})$$
           This mathematical identity was verified across tests to within machine epsilon ($\\le 2.22 \\times 10^{-16}$).
        3. **Non-Causal Interpretation**:
           SHAP values quantify statistical credit assignment within the observational tree models. They **do not prove causal mechanisms** or indicate that changing a customer's attribute will alter their actual retention behavior.
        """
    )

    # 3. Risk Tier Cohort Summary & Decoupling Simulator
    st.markdown('<div class="section-header"><h3>3. Operational Risk Tier Cohort Summary</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
        Calibrated probabilities are partitioned into three contiguous operational risk tiers to support workflow queue management.

        **Population Reference:** The cohort shares and observed churn rates below are derived strictly from the **retrospective observational training cohort ($N = 5,634$)** under 5-fold out-of-fold cross-validation.
        """
    )

    render_callout(
        "<strong>Methodological Distinction on Probability Calibration:</strong> Grouping scores into operational risk tiers "
        "is an operational categorization for queue management; <em>tier grouping itself does not establish or create probability calibration</em>. "
        "Probability calibration was independently evaluated and established using Sigmoid (Platt) scaling across proper scoring rules (Brier score and Expected Calibration Error). "
        "The observed churn rates represent historical cohort frequencies within the retrospective dataset, not guaranteed event rates for individual accounts.",
        title="Calibration vs Tier Grouping Boundary",
        variant="info",
    )

    df_risk = load_risk_tier_summary()
    if not df_risk.empty:
        st.dataframe(
            df_risk[
                [
                    "risk_tier",
                    "probability_range",
                    "account_count",
                    "percent_of_cohort",
                    "mean_model_probability",
                    "observed_churn_rate",
                    "description",
                ]
            ].rename(
                columns={
                    "risk_tier": "Risk Tier",
                    "probability_range": "Probability Range",
                    "account_count": "Cohort Accounts",
                    "percent_of_cohort": "Cohort Share (%)",
                    "mean_model_probability": "Mean Model Prob",
                    "observed_churn_rate": "Observed Churn Rate",
                    "description": "Operational Intervention Meaning",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Interactive Risk Tier & Threshold Policy Simulator")
    st.caption("Simulate how any probability score is categorized across operational queues and binary action policies.")

    sim_col1, sim_col2 = st.columns([2, 2])
    with sim_col1:
        sim_prob = st.slider("Simulated Customer Churn Probability:", min_value=0.00, max_value=1.00, value=0.19, step=0.01)
        sim_thresh = st.slider("Operating Action Threshold (τ):", min_value=0.05, max_value=0.95, value=float(threshold), step=0.01)

    with sim_col2:
        # Determine tier
        if sim_prob < 0.20:
            sim_tier = "Low"
        elif sim_prob < 0.50:
            sim_tier = "Medium"
        else:
            sim_tier = "High"

        sim_action = sim_prob >= sim_thresh

        st.markdown("**Simulation Output:**")
        st.markdown(f"- **Simulated Probability:** `{sim_prob:.2f}` ({format_probability(sim_prob)})")
        st.markdown("- **Operational Queue:**")
        render_risk_tier_badge(sim_tier)
        st.markdown("- **Binary Retention Action:**")
        render_decision_badge(
            sim_action,
            "Flagged by Current Policy" if sim_action else "Not Flagged by Current Policy",
        )

        if sim_action and sim_tier == "Low":
            st.warning(
                f"**Boundary Demonstration:** Probability `{sim_prob:.2f}` triggers binary outreach action (`≥ {sim_thresh:.2f}`), "
                "yet belongs to the **Low Risk Tier** (`< 0.20`). This demonstrates why decision policy and risk ranking are decoupled."
            )
