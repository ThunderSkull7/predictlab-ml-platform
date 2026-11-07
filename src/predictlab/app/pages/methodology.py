"""Page 5: Methodology & Scientific Governance for PredictLab Platform."""

import streamlit as st

from predictlab.app.components.cards import render_callout


def render_methodology_page() -> None:
    """Render comprehensive methodological documentation, limitations, and governance standards."""
    st.title("Methodology, System Architecture & Scientific Governance")
    st.markdown(
        """
        Transparent scientific documentation is a core requirement of the PredictLab engineering philosophy.
        This document outlines the data lineage, preprocessing safeguards, model development lifecycle,
        calibration protocols, and explicit methodological boundaries of the platform.
        """
    )

    # 1. Problem Formulation & Scope
    st.markdown('<div class="section-header"><h3>1. Problem Formulation & Scope Boundaries</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Problem Type**: Retrospective, cross-sectional binary classification and customer churn risk ranking.
        - **Observational Nature**: The IBM Telco Customer Churn dataset represents an observational cross-sectional snapshot of customer accounts at a single fixed observation point.
        - **Explicit Scope Limitations**:
          - This system does **not** perform forward-looking longitudinal event forecasting.
          - It is **not** a survival analysis or time-to-event model.
          - Churn outcomes indicate whether a subscriber had terminated service as of the snapshot date, not when they will churn in the future.
          - Temporal generalization across changing market regimes or macroeconomic cycles has not been established.
        """
    )

    # 2. Dataset Lineage & Integrity
    st.markdown('<div class="section-header"><h3>2. Dataset Lineage & Quality Audit</h3></div>', unsafe_allow_html=True)
    st.markdown(
        r"""
        - **Dataset Source**: IBM Sample Data Sets (Telco Customer Churn).
        - **Sample Size**: 7,043 unique customer accounts.
        - **Target Variable**: `Churn` (binary: 'Yes' / 'No').
          - Churned subscribers: 1,869 ($26.54\%$).
          - Retained subscribers: 5,174 ($73.46\%$).
        - **Input Predictors**: Exactly 19 raw domain predictors spanning demographic attributes, subscribed telephony and internet services, contract terms, payment methods, and billing figures.
        - **Identified Data Quality Anomalies**:
          - `TotalCharges` contained 11 whitespace strings (`" "`) corresponding strictly to subscribers with `tenure == 0` (brand-new accounts who had not yet received their first monthly bill).
          - During ingestion and preprocessing, these whitespace strings are mapped to missing representations and imputed with `0.0` or median within cross-validation folds without data leakage.
        - **Dataset Immutability**:
          - The raw dataset is cryptographically hashed: `SHA-256: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
          - The canonical train/test split manifest locks $5,634$ training samples ($80\%$) and $1,409$ holdout test samples ($20\%$).
        """
    )

    # 3. Leakage-Free Preprocessing Architecture
    st.markdown('<div class="section-header"><h3>3. Leakage-Free Preprocessing Pipeline</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Identifier Exclusion**: `customerID` is strictly isolated during ingestion and never exposed to the preprocessing pipeline or estimator.
        - **Baseline 47 Features**:
          - Numerical predictors (`tenure`, `MonthlyCharges`, `TotalCharges`) are imputed with training-fold medians.
          - Categorical predictors (16 features) are encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
        - **Experimental Pipeline Audit & ChargesRatio Correction**:
          - An experimental 52-feature pipeline was investigated in Phase 3–3.2 with engineered features (`ChargesRatio`, `TotalServicesCount`, `StreamingBundle`, `SeniorLivingAlone`).
          - Phase 3.1 identified a mathematical singularity: `ChargesRatio = TotalCharges / (tenure * MonthlyCharges)` is undefined ($0/0$) when `tenure == 0`. The implementation was corrected to assign median imputation with a boolean indicator `ChargesRatio_is_defined`.
          - Rigorous Phase 4 feature ablation revealed that experimental features yielded no statistically significant improvement in Average Precision ($0.6684$ baseline vs $0.6689$ experimental, $p = 0.94$). Under Occam's razor, the **default 47-feature baseline** was selected for production inference.
        """
    )

    # 4. Model Exploration & Calibration Protocol
    st.markdown('<div class="section-header"><h3>4. Model Selection & Probability Calibration</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Cross-Validation Framework**: 5-fold Stratified Cross-Validation strictly inside the training partition.
        - **Primary Selection Metric**: **Average Precision (PR-AUC)** was selected as the authoritative primary metric due to class imbalance ($26.54\\%$ minority prevalence), avoiding the well-documented optimism of ROC-AUC.
        - **Candidate Architecture Comparison**:
          - Logistic Regression (L1/L2): Average Precision $\\approx 0.648$
          - Random Forest (Balanced/Tuned): Average Precision $\\approx 0.638$
          - HistGradientBoosting (Tuned): **Average Precision = 0.6684 ± 0.0218** (Selected Winner)
        - **Probability Calibration**:
          - Tree ensemble outputs raw decision margins that do not represent true empirical probabilities.
          - Evaluated uncalibrated estimates, Isotonic regression, and Sigmoid (Platt) scaling across 5x5 nested cross-validation.
          - **Sigmoid Scaling** was selected: achieved lowest Brier score ($0.1335$), superior log loss ($0.4116$), and stable reliability curves without the step-function overfitting characteristic of Isotonic regression on small validation cohorts.
        """
    )

    # 5. Cost-Sensitive Optimization & Historical Audits
    st.markdown('<div class="section-header"><h3>5. Decision Threshold Modeling & Historical Audit Reconciliations</h3></div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **Financial Cost Matrix**:
          $$\\text{Total Cost} = C_{\\text{FP}} \\cdot \\text{FP} + C_{\\text{FN}} \\cdot \\text{FN} + C_{\\text{TP}} \\cdot \\text{TP} + C_{\\text{TN}} \\cdot \\text{TN}$$
          Under standard telecommunications retention assumptions ($C_{\\text{FN}} = \\$500, C_{\\text{FP}} = \\$100, C_{\\text{TP}} = \\$100, C_{\\text{TN}} = \\$0$):
          - Default threshold $\\tau = 0.50$: Total cost is $\\$467,000$ (missing $697$ churners).
          - Cost-optimal threshold $\\tau^* = 0.19$: Total cost drops to **$\\$370,100$**, capturing $1,305$ churners and delivering **$\\$96,900$ in net savings ($20.75\\%$)**.
        - **Historical Audit Reconciliations**:
          - **Nested Threshold Optimism**: A 5-fold nested cross-validation policy evaluation confirmed minimal selection optimism across folds (mean fold threshold $\\tau = 0.1980 \\pm 0.0098$, optimism delta = $-0.0012$).
          - **Historical 21-Observation Discrepancy**: Phase 5.1.2 reconciled an empirical 21-observation discrepancy on the negative class between the original Phase 5 narrative draft ($\text{FP}=410, \text{TN}=3,729$) and the reproduced uncalibrated out-of-fold predictions ($\text{FP}=389, \text{TN}=3,750$). Exhaustive audits confirmed that positive class counts ($\text{TP}=782, \text{FN}=713$) matched identically, and that the 21-case shift existed strictly as an unverified manual draft artifact.
          - The **Sigmoid-calibrated out-of-fold results** ($\text{TP}=798, \text{FP}=387, \text{FN}=697, \text{TN}=3,752$) serve as the sole authoritative and fully reproducible evaluation basis.
        """
    )

    # 6. Explicit Governance & Limitations
    st.markdown('<div class="section-header"><h3>6. Explicit Governance Disclosures & Limitations</h3></div>', unsafe_allow_html=True)
    render_callout(
        "<strong>Portfolio Governance Summary:</strong><br>"
        "1. <strong>No Production Performance Claims:</strong> This application demonstrates machine learning engineering best practices on an open benchmark dataset. It does not represent an actively deployed commercial production system.<br>"
        "2. <strong>Hypothetical Economics:</strong> The retention costs ($500 false negative penalty, $100 concession cost) are hypothetical scenario parameters designed to demonstrate asymmetric decision theory, not audited financial accounting figures.<br>"
        "3. <strong>Strict Holdout Isolation:</strong> The canonical holdout partition (N = 1,409) was not used for application-level threshold selection or interactive feature exploration.<br>"
        "4. <strong>No Causal Guarantees:</strong> Model predictions and SHAP attributions measure observational correlations, not real-world intervention efficacy.",
        title="Ethical & Methodological Transparency",
        variant="warning",
    )
