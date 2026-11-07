"""Status and risk badges for analytical outputs."""

import streamlit as st

from predictlab.app.utils.formatting import format_risk_tier_badge, format_status_pill


def render_risk_tier_badge(tier: str) -> None:
    """Render HTML styled risk badge."""
    st.markdown(format_risk_tier_badge(tier), unsafe_allow_html=True)


def render_decision_badge(is_churn: bool, label: str | None = None) -> None:
    """Render HTML styled binary decision pill."""
    st.markdown(format_status_pill(is_churn, label), unsafe_allow_html=True)
