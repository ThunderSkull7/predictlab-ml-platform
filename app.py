"""PredictLab — Production-Quality Streamlit Application Entry Point.

Retrospective customer churn classification and risk ranking using the IBM Telco Customer Churn dataset.
"""

import sys
from pathlib import Path

# Ensure 'src' directory is in Python path for portable execution across all platforms
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st  # noqa: E402

from predictlab.app.utils.styling import inject_custom_css  # noqa: E402

# 1. Page Configuration
st.set_page_config(
    page_title="PredictLab — Predictive Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Global Styling Injection
inject_custom_css()

# 3. Sidebar Header & Metadata Branding
with st.sidebar:
    st.markdown("## 📈 PredictLab")
    st.caption("AI/ML Predictive Analytics Platform")
    st.markdown(
        """
        <div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 10px 12px; margin-bottom: 14px;">
            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">System Status</div>
            <div style="font-size: 0.88rem; color: #34D399; font-weight: 600; margin-top: 2px;">● Model Active (Calibrated)</div>
            <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 4px;">HistGradientBoosting + Sigmoid</div>
            <div style="font-size: 0.78rem; color: #38BDF8; margin-top: 2px;">Operating Threshold: τ* = 0.19</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

# 4. Multipage Navigation Setup using Streamlit Native st.navigation
pages = [
    st.Page(
        "app_pages/overview.py",
        title="Overview Dashboard",
        icon="📊",
        default=True,
    ),
    st.Page(
        "app_pages/single_prediction.py",
        title="Single Customer Prediction",
        icon="👤",
    ),
    st.Page(
        "app_pages/batch_analysis.py",
        title="Batch Customer Analysis",
        icon="📁",
    ),
    st.Page(
        "app_pages/explainability.py",
        title="Model Insights & Explainability",
        icon="🔍",
    ),
    st.Page(
        "app_pages/methodology.py",
        title="Methodology & Governance",
        icon="📖",
    ),
]

nav = st.navigation(
    {
        "Analytics Platform": pages,
    }
)

# 5. Sidebar Footer
with st.sidebar:
    st.markdown("---")
    st.caption("PredictLab Portfolio Project • Phase 6")
    st.caption("Dataset: IBM Telco Customer Churn (7,043 records)")
    st.caption("Cross-sectional retrospective classification")

# 6. Run Navigation
nav.run()
