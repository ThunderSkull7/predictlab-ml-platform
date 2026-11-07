"""Unit tests for PredictLab presentation components and formatting helpers."""

from predictlab.app.utils.formatting import (
    format_currency,
    format_probability,
    format_risk_tier_badge,
    format_status_pill,
)


def test_format_probability() -> None:
    """Test probability formatting."""
    assert format_probability(0.6636) == "66.4%"
    assert format_probability(0.6636, decimals=2) == "66.36%"
    assert format_probability(0.0) == "0.0%"
    assert format_probability(1.0) == "100.0%"


def test_format_currency() -> None:
    """Test currency formatting."""
    assert format_currency(96900.0) == "$96,900"
    assert format_currency(65.69, decimals=2) == "$65.69"
    assert format_currency(0.0) == "$0"


def test_format_risk_tier_badge() -> None:
    """Test HTML badge formatting for risk tiers."""
    low_badge = format_risk_tier_badge("Low")
    assert "risk-low" in low_badge
    assert "Low Risk" in low_badge

    med_badge = format_risk_tier_badge("Medium")
    assert "risk-medium" in med_badge
    assert "Medium Risk" in med_badge

    high_badge = format_risk_tier_badge("High")
    assert "risk-high" in high_badge
    assert "High Risk" in high_badge


def test_format_status_pill() -> None:
    """Test HTML status pill formatting for binary predictions."""
    pill_churn = format_status_pill(is_churn=True)
    assert "status-churn" in pill_churn
    assert "Flagged for Retention Outreach" in pill_churn

    pill_retain = format_status_pill(is_churn=False)
    assert "status-retained" in pill_retain
    assert "Retained (No Intervention)" in pill_retain
