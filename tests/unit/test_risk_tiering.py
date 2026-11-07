"""Unit tests for customer risk-tiering definitions, assignment, and statistical profiling."""

import numpy as np
import pytest

from predictlab.thresholding.risk_tiers import RiskTierer


def test_default_risk_tierer_initialization() -> None:
    """Test default 3-tier RiskTierer initialization and validation."""
    tierer = RiskTierer()
    defs = tierer.definitions
    assert len(defs) == 3
    assert defs[0].name == "Low"
    assert defs[1].name == "Medium"
    assert defs[2].name == "High"
    assert defs[0].min_prob == 0.0
    assert defs[-1].max_prob == 1.0


def test_custom_risk_tierer_validation_gap_raises() -> None:
    """Test that a gap between risk tiers raises ValueError."""
    with pytest.raises(ValueError, match="Tier gap or overlap detected"):
        RiskTierer(
            tiers=[
                ("Low", 0.0, 0.30, "Low tier"),
                ("High", 0.40, 1.0, "High tier"),  # Gap between 0.30 and 0.40
            ]
        )


def test_custom_risk_tierer_validation_endpoint_raises() -> None:
    """Test that tiers not starting at 0.0 raise ValueError."""
    with pytest.raises(ValueError, match="Lowest tier must start at 0.0"):
        RiskTierer(
            tiers=[
                ("Low", 0.10, 0.50, "Low tier"),
                ("High", 0.50, 1.0, "High tier"),
            ]
        )


def test_tier_assignment_single_and_array() -> None:
    """Test assigning individual probabilities and arrays to tiers."""
    tierer = RiskTierer()

    assert tierer.assign_tier(0.05) == "Low"
    assert tierer.assign_tier(0.20) == "Medium"
    assert tierer.assign_tier(0.499) == "Medium"
    assert tierer.assign_tier(0.50) == "High"
    assert tierer.assign_tier(1.0) == "High"

    assigned = tierer.assign_tiers([0.1, 0.3, 0.8])
    assert assigned == ["Low", "Medium", "High"]


def test_risk_tier_statistical_summary() -> None:
    """Test cohort statistical summary generation across risk tiers."""
    tierer = RiskTierer()
    probs = np.array([0.05, 0.15, 0.25, 0.45, 0.65, 0.85])
    labels = np.array([0, 0, 0, 1, 1, 1])

    df = tierer.compute_tier_summary(probs, y_true=labels)
    assert len(df) == 3
    assert "risk_tier" in df.columns
    assert "account_count" in df.columns
    assert "percent_of_cohort" in df.columns
    assert "observed_churn_rate" in df.columns

    # Low tier (2 samples, 0 churners -> 0.0 observed churn)
    low_row = df[df["risk_tier"] == "Low"].iloc[0]
    assert low_row["account_count"] == 2
    assert low_row["observed_churn_rate"] == 0.0

    # High tier (2 samples, 2 churners -> 1.0 observed churn)
    high_row = df[df["risk_tier"] == "High"].iloc[0]
    assert high_row["account_count"] == 2
    assert high_row["observed_churn_rate"] == 1.0
