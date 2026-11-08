"""Page 3: Batch Customer Analysis Interface for PredictLab Platform."""

import io
from typing import Any

import pandas as pd
import streamlit as st

from predictlab.app.components.cards import render_callout, render_metric_card
from predictlab.app.components.charts import render_batch_risk_donut_chart
from predictlab.app.services.data_service import generate_sample_batch_csv
from predictlab.app.services.inference_service import (
    get_inference_engine,
    run_batch_prediction,
    validate_and_parse_batch_csv,
)
from predictlab.app.utils.formatting import format_probability


def render_batch_analysis_page() -> None:
    """Render the batch customer CSV upload, validation, and vectorized scoring interface."""
    st.title("Batch Customer Analysis & Portfolio Risk Scoring")
    st.markdown(
        """
        Upload customer cohort CSV files for **high-throughput, vectorized inference scoring**.
        The engine performs schema validation, captures invalid records with explicit error diagnostics,
        preserves exact input ordering, and assigns calibrated probabilities, binary classifications, and risk tiers.
        """
    )

    try:
        engine = get_inference_engine()
    except Exception as err:
        st.error(f"Inference Engine failed to initialize: {err}")
        return

    # 1. Download Sample Batch CSV
    st.markdown('<div class="section-header"><h3>1. Data Upload & Template</h3></div>', unsafe_allow_html=True)
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        st.caption(
            "Upload a CSV containing customer accounts. Required columns: the 19 raw predictors. "
            "`customerID` is captured for record tracking and stripped before scoring. Ground truth `Churn` is excluded if present."
        )
    with tcol2:
        sample_csv = generate_sample_batch_csv(include_invalid=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            st.download_button(
                label="📥 Download CSV",
                data=sample_csv,
                file_name="predictlab_sample_batch.csv",
                mime="text/csv",
                help="Download sample CSV containing 5 valid customer records and 1 invalid record.",
                use_container_width=True,
            )
        with bc2:
            load_sample = st.button(
                "🚀 Score Sample Cohort",
                help="Directly score the 6 sample customer records without uploading a file.",
                use_container_width=True,
            )

    # Public Demonstration Privacy & Confidentiality Notice
    st.warning(
        "🔒 **Public Demonstration Privacy Notice:** PredictLab is an unauthenticated public portfolio demonstration. "
        "**Do not upload real customer personal, financial, or confidential information.** "
        "Uploaded records are processed strictly in temporary server memory and are never persisted to disk or databases. "
        "Please use synthetic, benchmark, or anonymized cohort data only."
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Cohort CSV File:",
        type=["csv"],
        help="Upload a CSV file (maximum 10,000 records recommended for interactive analysis).",
    )

    # Threshold configuration
    thresh_col1, thresh_col2 = st.columns([2, 3])
    with thresh_col1:
        batch_threshold = st.slider(
            "Batch Decision Threshold (τ):",
            min_value=0.05,
            max_value=0.95,
            value=float(engine.decision_threshold),
            step=0.01,
            help="Threshold applied for binary classification in this batch.",
        )
    with thresh_col2:
        st.caption(
            f"**Applied Decision Threshold:** `τ = {batch_threshold:.2f}` (Cost-optimal default: `0.19`). "
            "Accounts with model-estimated churn probability ≥ threshold will be classified as 'Flagged by Current Policy'."
        )

    # Resolve batch input source
    batch_data = None
    batch_name = ""
    if uploaded_file is not None:
        batch_data = uploaded_file
        batch_name = uploaded_file.name
        st.session_state["batch_source"] = "uploaded"
    elif load_sample or st.session_state.get("batch_source") == "sample" or "batch_source" not in st.session_state:
        batch_data = io.StringIO(sample_csv)
        batch_name = "predictlab_sample_batch.csv (Demo Cohort)"
        st.session_state["batch_source"] = "sample"

    if batch_data is None:
        st.info("Upload a customer CSV above or click 'Score Sample Cohort' to get started.")
        return

    # 2. Parse & Validate CSV via service
    df_raw, parse_err = validate_and_parse_batch_csv(batch_data)
    if parse_err is not None or df_raw is None:
        st.error(parse_err)
        return

    n_raw = len(df_raw)
    st.success(f"Cohort loaded successfully: **{batch_name}** ({n_raw:,} records)")

    # 3. Execute Vectorized Batch Inference
    with st.spinner(f"Scoring {n_raw:,} records via vectorized calibrated pipeline..."):
        records_list = df_raw.to_dict(orient="records")
        batch_output = run_batch_prediction(
            engine=engine,
            records=records_list,
            threshold=batch_threshold,
        )

    # 4. Display Processing & Quality Metrics
    st.markdown('<div class="section-header"><h3>2. Batch Quality & Processing Summary</h3></div>', unsafe_allow_html=True)

    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        render_metric_card("Total Submitted", f"{batch_output.total_records:,}", "Input Rows", "neutral")
    with qcol2:
        render_metric_card("Successfully Scored", f"{batch_output.successful_predictions:,}", f"{batch_output.successful_predictions / batch_output.total_records * 100:.1f}% Yield", "positive")
    with qcol3:
        val_status = "positive" if batch_output.failed_predictions == 0 else "warning"
        render_metric_card("Rejected Records", f"{batch_output.failed_predictions:,}", "Validation Errors", val_status)
    with qcol4:
        render_metric_card("Error Handling Mode", batch_output.error_handling_mode.title(), "Non-terminating per-record", "neutral")

    # If there were rejected records, show error diagnostics table
    if batch_output.failed_predictions > 0:
        st.markdown("#### Validation Rejections Diagnostic Log")
        rejected_rows = [
            {
                "Row Index": r.record_index + 1,
                "Customer ID": r.customer_id or "N/A",
                "Validation Error Diagnostic": r.error_message,
            }
            for r in batch_output.predictions
            if not r.is_valid
        ]
        df_rejected = pd.DataFrame(rejected_rows)
        st.dataframe(df_rejected, use_container_width=True, hide_index=True)

        buf_rej = io.StringIO()
        df_rejected.to_csv(buf_rej, index=False)
        st.download_button(
            label="📥 Download Validation Error Log (CSV)",
            data=buf_rej.getvalue(),
            file_name="predictlab_validation_errors.csv",
            mime="text/csv",
            use_container_width=False,
        )

    if batch_output.successful_predictions == 0:
        st.error("No valid customer records could be processed. Please review the error log above.")
        return

    # 5. Extract Scored Results
    valid_records = [r for r in batch_output.predictions if r.is_valid and r.prediction is not None]
    results_data: list[dict[str, Any]] = []
    tier_counts = {"Low": 0, "Medium": 0, "High": 0}
    churn_count = 0
    retained_count = 0
    prob_sum = 0.0

    for r in valid_records:
        pred = r.prediction
        assert pred is not None
        is_churn = pred.predicted_class == 1
        if is_churn:
            churn_count += 1
        else:
            retained_count += 1

        prob_sum += pred.estimated_probability
        tier = pred.risk_tier or "Low"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        results_data.append(
            {
                "Row Index": r.record_index + 1,
                "CustomerID": r.customer_id if r.customer_id is not None else f"REC-{r.record_index+1:04d}",
                "Estimated Churn Probability": pred.estimated_probability,
                "Binary Decision": "Flagged by Current Policy" if is_churn else "Not Flagged by Current Policy",
                "Risk Tier": tier,
                "Applied Threshold": pred.applied_threshold,
                "Model Version": pred.model_version,
            }
        )

    df_results = pd.DataFrame(results_data)
    n_valid = len(df_results)
    avg_prob = prob_sum / n_valid if n_valid > 0 else 0.0

    # 6. Portfolio Prediction Insights
    st.markdown('<div class="section-header"><h3>3. Portfolio Churn & Risk Distribution</h3></div>', unsafe_allow_html=True)

    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    with rcol1:
        render_metric_card("Flagged by Current Policy", f"{churn_count:,}", f"{churn_count / n_valid * 100:.1f}% of Scored", "warning")
    with rcol2:
        render_metric_card("Not Flagged by Policy", f"{retained_count:,}", f"{retained_count / n_valid * 100:.1f}% of Scored", "positive")
    with rcol3:
        render_metric_card("Mean Churn Probability", format_probability(avg_prob, decimals=1), "Cohort Average", "neutral")
    with rcol4:
        render_metric_card("High Risk Accounts", f"{tier_counts.get('High', 0):,}", f"{tier_counts.get('High', 0) / n_valid * 100:.1f}% of Cohort", "warning")

    # Donut Chart and Breakdown
    dcol_chart, dcol_table = st.columns([1, 1])
    with dcol_chart:
        st.subheader("Risk Tier Queue Breakdown")
        render_batch_risk_donut_chart(tier_counts)
    with dcol_table:
        st.subheader("Queue Summary Table")
        df_queue_summary = pd.DataFrame(
            [
                {"Risk Tier": "Low Risk [0.00, 0.20)", "Accounts": tier_counts.get("Low", 0), "Share": f"{tier_counts.get('Low', 0) / n_valid * 100:.1f}%", "Intervention Policy": "Baseline monitoring; standard operational service"},
                {"Risk Tier": "Medium Risk [0.20, 0.50)", "Accounts": tier_counts.get("Medium", 0), "Share": f"{tier_counts.get('Medium', 0) / n_valid * 100:.1f}%", "Intervention Policy": "Proactive satisfaction check; contract renewal review"},
                {"Risk Tier": "High Risk [0.50, 1.00]", "Accounts": tier_counts.get("High", 0), "Share": f"{tier_counts.get('High', 0) / n_valid * 100:.1f}%", "Intervention Policy": "Priority retention incentive; outbound specialist outreach"},
            ]
        )
        st.dataframe(df_queue_summary, use_container_width=True, hide_index=True)

    # 7. Interactive Results Table & Filtering
    st.markdown('<div class="section-header"><h3>4. Detailed Scored Predictions Table</h3></div>', unsafe_allow_html=True)

    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        tier_filter = st.multiselect(
            "Filter by Risk Tier:",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
        )
    with fcol2:
        search_query = st.text_input("Search Customer ID:", placeholder="e.g. DEMO-001 or 1234")

    # Filter dataframe
    df_filtered = df_results[df_results["Risk Tier"].isin(tier_filter)]
    if search_query:
        df_filtered = df_filtered[df_filtered["CustomerID"].astype(str).str.contains(search_query, case=False, na=False)]

    st.dataframe(
        df_filtered.style.format(
            {
                "Estimated Churn Probability": "{:.4f}",
                "Applied Threshold": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Displaying {len(df_filtered):,} of {n_valid:,} scored accounts.")

    # 8. Download Scored Predictions
    buf_out = io.StringIO()
    df_results.to_csv(buf_out, index=False)
    st.download_button(
        label="📥 Download Complete Scored Predictions (CSV)",
        data=buf_out.getvalue(),
        file_name="predictlab_batch_scored_predictions.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    render_callout(
        "<strong>Session Privacy & Governance Guarantee:</strong> Uploaded customer records are processed entirely in memory "
        "and are <strong>never written to disk or persisted beyond this interactive session</strong>. "
        "Model predictions reflect statistical classification against observational snapshot features, not deterministic future outcomes.",
        title="Data Governance Assurance",
        variant="info",
    )
