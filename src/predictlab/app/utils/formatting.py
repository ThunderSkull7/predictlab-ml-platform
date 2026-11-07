"""Data presentation formatting functions for PredictLab UI."""


def format_probability(val: float, decimals: int = 1) -> str:
    """Format float probability into percentage string."""
    return f"{val * 100:.{decimals}f}%"


def format_currency(val: float, decimals: int = 0) -> str:
    """Format dollar currency amount."""
    if decimals == 0:
        return f"${val:,.0f}"
    return f"${val:,.{decimals}f}"


def format_risk_tier_badge(tier: str) -> str:
    """Return HTML string for styled risk tier badge."""
    tier_lower = tier.lower() if tier else "low"
    css_class = f"risk-badge risk-{tier_lower}"
    return f'<span class="{css_class}">{tier} Risk</span>'


def format_status_pill(is_churn: bool, label: str | None = None) -> str:
    """Return HTML string for styled binary decision status pill."""
    if is_churn:
        lbl = label or "Flagged for Retention Outreach"
        return f'<span class="status-pill status-churn">{lbl}</span>'
    lbl = label or "Retained (No Intervention)"
    return f'<span class="status-pill status-retained">{lbl}</span>'
