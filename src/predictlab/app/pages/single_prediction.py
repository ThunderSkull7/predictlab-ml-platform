"""Page 2: Single Customer Prediction Interface for PredictLab Platform."""

from typing import Any

import streamlit as st

from predictlab.app.components.badges import render_decision_badge, render_risk_tier_badge
from predictlab.app.components.cards import render_callout, render_metric_card
from predictlab.app.components.charts import render_shap_attribution_chart
from predictlab.app.services.data_service import get_sample_customer_presets
from predictlab.app.services.inference_service import get_inference_engine, run_single_prediction
from predictlab.app.utils.formatting import format_probability
from predictlab.inference.schemas import InferenceValidationError


def render_single_prediction_page() -> None:
    """Render the single-customer inference form, prediction scoring, and local explainability."""
    st.title("Single Customer Prediction & Risk Scoring")
    st.markdown(
        """
        Score an individual customer account through the validated **PredictLab Inference Pipeline**.
        The engine applies verified schema validation, automated baseline transformations (47 features),
        5-fold Sigmoid calibration, cost-sensitive threshold classification, and local TreeSHAP attribution.
        """
    )

    try:
        engine = get_inference_engine()
    except Exception as err:
        st.error(f"Inference Engine failed to initialize: {err}")
        return

    # 1. Preset Profile Selection
    st.markdown('<div class="section-header"><h3>1. Customer Account Attributes</h3></div>', unsafe_allow_html=True)

    presets = get_sample_customer_presets()
    preset_names = ["Custom Manual Entry"] + list(presets.keys())
    selected_preset = st.selectbox(
        "Load Representative Customer Profile (Optional Preset):",
        options=preset_names,
        index=0,
        help="Select a pre-configured profile to auto-populate the input fields below.",
    )

    default_values: dict[str, Any] = presets.get(selected_preset, presets["High Risk (Short Tenure, Fiber, Month-to-Month)"]) if selected_preset != "Custom Manual Entry" else {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 6,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 479.10,
    }

    # Input Form containing the exact 19 raw predictors
    with st.form(key="single_customer_form"):
        st.subheader("Demographics & Account Tenancy")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            gender = st.selectbox("Gender", options=["Female", "Male"], index=0 if default_values["gender"] == "Female" else 1)
        with dcol2:
            senior = st.selectbox("Senior Citizen", options=[0, 1], index=int(default_values["SeniorCitizen"]), format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        with dcol3:
            partner = st.selectbox("Partner", options=["Yes", "No"], index=0 if default_values["Partner"] == "Yes" else 1)
        with dcol4:
            dependents = st.selectbox("Dependents", options=["Yes", "No"], index=0 if default_values["Dependents"] == "Yes" else 1)

        st.subheader("Telephony & Internet Services")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            phone_service = st.selectbox("Phone Service", options=["Yes", "No"], index=0 if default_values["PhoneService"] == "Yes" else 1)
        with scol2:
            mult_lines_idx = ["No", "Yes", "No phone service"].index(default_values["MultipleLines"]) if default_values["MultipleLines"] in ["No", "Yes", "No phone service"] else 0
            multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes", "No phone service"], index=mult_lines_idx)
        with scol3:
            inet_idx = ["DSL", "Fiber optic", "No"].index(default_values["InternetService"]) if default_values["InternetService"] in ["DSL", "Fiber optic", "No"] else 0
            internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"], index=inet_idx)

        st.subheader("Value-Added Digital Services")
        vcol1, vcol2, vcol3, vcol4, vcol5, vcol6 = st.columns(6)
        opt_choices = ["No", "Yes", "No internet service"]

        def get_opt_idx(val: str) -> int:
            return opt_choices.index(val) if val in opt_choices else 0

        with vcol1:
            online_security = st.selectbox("Online Security", options=opt_choices, index=get_opt_idx(default_values["OnlineSecurity"]))
        with vcol2:
            online_backup = st.selectbox("Online Backup", options=opt_choices, index=get_opt_idx(default_values["OnlineBackup"]))
        with vcol3:
            device_protection = st.selectbox("Device Protection", options=opt_choices, index=get_opt_idx(default_values["DeviceProtection"]))
        with vcol4:
            tech_support = st.selectbox("Tech Support", options=opt_choices, index=get_opt_idx(default_values["TechSupport"]))
        with vcol5:
            streaming_tv = st.selectbox("Streaming TV", options=opt_choices, index=get_opt_idx(default_values["StreamingTV"]))
        with vcol6:
            streaming_movies = st.selectbox("Streaming Movies", options=opt_choices, index=get_opt_idx(default_values["StreamingMovies"]))

        st.subheader("Contract Terms & Billing")
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            contract_idx = ["Month-to-month", "One year", "Two year"].index(default_values["Contract"]) if default_values["Contract"] in ["Month-to-month", "One year", "Two year"] else 0
            contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"], index=contract_idx)
        with bcol2:
            paperless = st.selectbox("Paperless Billing", options=["Yes", "No"], index=0 if default_values["PaperlessBilling"] == "Yes" else 1)
        with bcol3:
            pay_options = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            pay_idx = pay_options.index(default_values["PaymentMethod"]) if default_values["PaymentMethod"] in pay_options else 0
            payment_method = st.selectbox("Payment Method", options=pay_options, index=pay_idx)

        st.subheader("Financial & Tenancy Metrics")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            tenure = st.number_input(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=120,
                value=int(default_values["tenure"]),
                step=1,
                help="Duration in months as an active subscriber (0 for brand-new accounts).",
            )
        with mcol2:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=300.0,
                value=float(default_values["MonthlyCharges"]),
                step=0.50,
                format="%.2f",
                help="Current recurring monthly bill in USD.",
            )
        with mcol3:
            tot_val = default_values["TotalCharges"]
            if isinstance(tot_val, str) or tot_val == 0.0:
                calc_tot = 0.0 if tenure == 0 else round(float(monthly_charges) * max(tenure, 1), 2)
            else:
                calc_tot = float(tot_val)
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=20000.0,
                value=calc_tot,
                step=5.0,
                format="%.2f",
                help="Cumulative gross charges billed to date. Enter 0.0 for zero-tenure new subscribers.",
            )

        st.markdown('<div class="section-header"><h3>2. Operating Decision Threshold Configuration</h3></div>', unsafe_allow_html=True)
        tcol1, tcol2 = st.columns([2, 3])
        with tcol1:
            applied_threshold = st.slider(
                "Classification Decision Threshold (τ):",
                min_value=0.05,
                max_value=0.95,
                value=float(engine.decision_threshold),
                step=0.01,
                help="Threshold governing the binary decision. The cost-optimal operating threshold is τ* = 0.19.",
            )
        with tcol2:
            st.caption(
                f"**Current Threshold:** `τ = {applied_threshold:.2f}` (Cost-optimal default: `0.19`). "
                "Subscribers with estimated churn probability ≥ threshold are flagged for proactive retention outreach."
            )

        submit_btn = st.form_submit_button(
            label="⚡ Run Production Inference & Local Explainability",
            type="primary",
            use_container_width=True,
        )

    # 2. Prediction Execution & Output Rendering
    if submit_btn:
        # Build raw predictor payload
        record_payload = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": " " if (tenure == 0 and total_charges == 0.0) else float(total_charges),
        }

        try:
            with st.spinner("Executing calibrated inference and TreeSHAP attributions..."):
                res = run_single_prediction(
                    engine=engine,
                    record=record_payload,
                    threshold=applied_threshold,
                    include_explanation=True,
                )
        except InferenceValidationError as val_err:
            st.error(f"Input Validation Error: {val_err}")
            return
        except Exception as pred_err:
            st.error(f"Inference Execution Error: {pred_err}")
            return

        st.markdown('<div class="section-header"><h3>3. Inference & Risk Classification Results</h3></div>', unsafe_allow_html=True)

        # KPI Output Tiles
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            render_metric_card(
                "Estimated Churn Probability",
                format_probability(res.estimated_probability, decimals=1),
                f"Model: {res.calibration_method.title()}-Calibrated",
                "warning" if res.estimated_probability >= applied_threshold else "positive",
            )
        with pcol2:
            render_metric_card(
                "Binary Decision",
                "Flagged by Current Policy" if res.predicted_class == 1 else "Not Flagged by Current Policy",
                f"Threshold τ = {res.applied_threshold:.2f}",
                "warning" if res.predicted_class == 1 else "positive",
            )
        with pcol3:
            render_metric_card(
                "Operational Risk Tier",
                f"{res.risk_tier} Risk",
                "Workflow Queue Category",
                "warning" if res.risk_tier == "High" else ("neutral" if res.risk_tier == "Medium" else "positive"),
            )
        with pcol4:
            render_metric_card(
                "Model Pipeline",
                res.model_name.split("(")[0].strip(),
                f"Version {res.model_version}",
                "neutral",
            )

        # Status and Badges
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Operational Queue:**")
            render_risk_tier_badge(res.risk_tier or "Low")
        with b2:
            st.markdown("**Decision Status:**")
            render_decision_badge(
                res.predicted_class == 1,
                "Flagged by Current Policy" if res.predicted_class == 1 else "Not Flagged by Current Policy",
            )

        # Crucial Methodological Distinction Notice
        st.markdown("---")
        st.markdown("#### Operational Distinction: Decision Threshold vs Risk Tiers")
        st.markdown(
            f"""
            This prediction demonstrates why the **Operating Decision Threshold** and **Risk Tiers** are intentionally decoupled:
            - **Decision Threshold ($\tau = {res.applied_threshold:.2f}$)** determines model-based classification:
              {'<span style="color:#F97316; font-weight:bold;">Flagged by Current Policy</span>' if res.predicted_class == 1 else '<span style="color:#10B981; font-weight:bold;">Not Flagged by Current Policy</span>'}.
            - **Risk Tier ({res.risk_tier} Risk)** partitions accounts into broad lifecycle workflows:
              Low `[0.00, 0.20)`, Medium `[0.20, 0.50)`, High `[0.50, 1.00]`.
            """
            + (
                f"> **Important Boundary Case:** Notice that a probability of `{res.estimated_probability:.2f}` produces an active classification flag (since `{res.estimated_probability:.2f} ≥ {res.applied_threshold:.2f}`), while remaining in the **Low Risk Tier** (`< 0.20`). The two systems serve distinct operational functions and must not be merged."
                if (res.predicted_class == 1 and res.risk_tier == "Low")
                else ""
            ),
            unsafe_allow_html=True,
        )

        render_callout(
            "<strong>Decision Policy Disclaimer:</strong> Classifications as <em>'Flagged by Current Policy'</em> or "
            "<em>'Not Flagged by Current Policy'</em> represent model-based classifications under the currently selected operating threshold (τ). "
            "These are statistical categorizations within a retrospective cross-sectional model and are <strong>not guarantees of future customer behavior</strong>, "
            "nor do they constitute a clinical or commercial recommendation that intervention is unnecessary for unflagged accounts.",
            title="Policy Classification & Intervention Disclaimer",
            variant="warning",
        )

        render_callout(
            "<strong>Probabilistic Interpretation:</strong> The model-estimated probability of "
            f"<strong>{format_probability(res.estimated_probability)}</strong> reflects statistical churn resemblance "
            "evaluated against historical observational patterns in the IBM Telco snapshot. "
            "It represents an estimated risk index, <strong>not a guaranteed individual future outcome</strong>.",
            title="Statistical Interpretation Note",
            variant="info",
        )

        # 4. Local Explainability Section
        st.markdown('<div class="section-header"><h3>4. Local Feature Attribution (TreeSHAP)</h3></div>', unsafe_allow_html=True)
        st.markdown("##### Local explanation of the model's averaged base decision margin")
        st.caption(
            "TreeSHAP attributes feature contributions in log-odds decision margin space across the calibrated ensemble. "
            "Red bars push the customer toward churn; blue bars push toward retention."
        )

        if res.explanation is not None:
            render_shap_attribution_chart(res.explanation, max_features=10)

            # Top Drivers Summary
            ecol1, ecol2 = st.columns(2)
            with ecol1:
                st.markdown("**Top Factors Increasing Churn Margin:**")
                top_churn = res.explanation.get("top_churn_drivers", [])
                if top_churn:
                    for d in top_churn[:4]:
                        st.markdown(f"- **{d['clean_feature']}** (`{d['feature_value']}`): `+{d['shap_value']:.3f}` log-odds")
                else:
                    st.caption("No strong positive churn margin contributors.")
            with ecol2:
                st.markdown("**Top Factors Decreasing Churn Margin (Retention):**")
                top_ret = res.explanation.get("top_retention_drivers", [])
                if top_ret:
                    for d in top_ret[:4]:
                        st.markdown(f"- **{d['clean_feature']}** (`{d['feature_value']}`): `{d['shap_value']:.3f}` log-odds")
                else:
                    st.caption("No strong retention margin contributors.")

            render_callout(
                "<strong>Scientific Limitation of SHAP:</strong> The attributions above explain the <em>averaged base model decision margin</em> (log-odds), "
                "not direct percentage contributions to the final calibrated probability. "
                "Feature attributions measure model sensitivity in observational data; they <strong>do not establish real-world causality</strong> "
                "or prove that altering a feature will prevent customer churn.",
                title="Explainability Boundary",
                variant="warning",
            )
        else:
            st.info("Local TreeSHAP feature attributions were not computed or are unavailable for this record.")
