"""KPI and informational card components."""

from typing import Literal

import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_type: Literal["positive", "neutral", "warning"] = "neutral",
) -> None:
    """Render a styled KPI metric card."""
    delta_html = ""
    if delta is not None:
        delta_class = f"delta-{delta_type}"
        delta_html = f'<div class="metric-card-delta {delta_class}">{delta}</div>'

    html = f"""
    <div class="metric-card">
        <div class="metric-card-label">{label}</div>
        <div class="metric-card-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_callout(
    text: str,
    title: str | None = None,
    variant: Literal["info", "warning"] = "info",
) -> None:
    """Render an analytical callout banner."""
    css_class = "warning-card" if variant == "warning" else "info-card"
    title_html = f"<strong>{title}</strong><br>" if title else ""
    st.markdown(f'<div class="{css_class}">{title_html}{text}</div>', unsafe_allow_html=True)
