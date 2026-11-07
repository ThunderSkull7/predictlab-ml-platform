"""Customer churn risk tiering utility and provisional business segmentation."""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class RiskTierDefinition(BaseModel):
    """Specification of a single risk tier segment."""

    name: str = Field(description="Tier identifier (e.g. 'Low', 'Medium', 'High')")
    min_prob: float = Field(ge=0.0, le=1.0, description="Inclusive or exclusive lower probability bound")
    max_prob: float = Field(ge=0.0, le=1.0, description="Upper probability bound")
    description: str = Field(description="Provisional operational or intervention meaning")


class RiskTierer:
    """Assigns subscriber probability scores into risk tiers for operational prioritization.

    Methodological & Governance Warnings:
    1. Risk ranking is distinct from binary classification thresholding.
    2. Boundaries defined here represent provisional, configurable operational buckets. They must not
       be presented as universally validated commercial retention policies.
    3. A risk score does NOT predict future churn with longitudinal temporal certainty; it ranks accounts
       by statistical resemblance to churned subscribers in the cross-sectional dataset.
    """

    def __init__(
        self,
        tiers: list[tuple[str, float, float, str]] | None = None,
    ) -> None:
        """Initialize risk tier definitions.

        Parameters
        ----------
        tiers : list of (name, min_prob, max_prob, description), optional
            Custom tier definitions. Defaults to 3 standard tiers:
            - Low Risk: [0.00, 0.20) - Baseline monitoring, no outreach discount.
            - Medium Risk: [0.20, 0.50) - Value communication, proactive service review.
            - High Risk: [0.50, 1.00] - Priority retention incentive, executive outreach.
        """
        if tiers is None:
            self._tier_defs = [
                RiskTierDefinition(
                    name="Low",
                    min_prob=0.0,
                    max_prob=0.20,
                    description="Accounts with low statistical churn indicators; routine operational engagement.",
                ),
                RiskTierDefinition(
                    name="Medium",
                    min_prob=0.20,
                    max_prob=0.50,
                    description="Elevated attrition indicators; proactive satisfaction checks and contract reviews.",
                ),
                RiskTierDefinition(
                    name="High",
                    min_prob=0.50,
                    max_prob=1.0,
                    description="Substantial attrition indicators; eligible for retention discount or intervention.",
                ),
            ]
        else:
            self._tier_defs = [
                RiskTierDefinition(name=t[0], min_prob=t[1], max_prob=t[2], description=t[3])
                for t in tiers
            ]

        self._validate_tiers()

    def _validate_tiers(self) -> None:
        """Verify tier boundaries form contiguous partition of [0.0, 1.0]."""
        sorted_tiers = sorted(self._tier_defs, key=lambda t: t.min_prob)
        if sorted_tiers[0].min_prob > 1e-6:
            raise ValueError(f"Lowest tier must start at 0.0, got {sorted_tiers[0].min_prob}")
        if sorted_tiers[-1].max_prob < 1.0 - 1e-6:
            raise ValueError(f"Highest tier must end at 1.0, got {sorted_tiers[-1].max_prob}")

        for i in range(len(sorted_tiers) - 1):
            if abs(sorted_tiers[i].max_prob - sorted_tiers[i + 1].min_prob) > 1e-4:
                raise ValueError(
                    f"Tier gap or overlap detected between '{sorted_tiers[i].name}' (max={sorted_tiers[i].max_prob}) "
                    f"and '{sorted_tiers[i + 1].name}' (min={sorted_tiers[i + 1].min_prob})"
                )

    @property
    def definitions(self) -> list[RiskTierDefinition]:
        """Return list of risk tier definitions."""
        return list(self._tier_defs)

    def assign_tier(self, probability: float) -> str:
        """Assign single probability score to its corresponding risk tier."""
        p = float(np.clip(probability, 0.0, 1.0))
        for t in self._tier_defs:
            if t.name == self._tier_defs[-1].name:
                # Include 1.0 in highest tier
                if t.min_prob <= p <= t.max_prob:
                    return t.name
            else:
                if t.min_prob <= p < t.max_prob:
                    return t.name
        return self._tier_defs[-1].name

    def assign_tiers(self, probabilities: np.ndarray | pd.Series | list[float]) -> list[str]:
        """Assign an array of probability scores to their corresponding risk tiers."""
        arr = np.asarray(probabilities).astype(float)
        return [self.assign_tier(p) for p in arr]

    def compute_tier_summary(
        self,
        probabilities: np.ndarray | pd.Series | list[float],
        y_true: np.ndarray | pd.Series | list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate statistical cohort profiling table across risk tiers."""
        probs = np.asarray(probabilities).astype(float)
        tiers = self.assign_tiers(probs)
        n_total = len(probs)

        df = pd.DataFrame({"probability": probs, "risk_tier": tiers})
        if y_true is not None:
            df["actual_churn"] = np.asarray(y_true).astype(int)

        records: list[dict[str, Any]] = []
        for t in self._tier_defs:
            subset = df[df["risk_tier"] == t.name]
            count = len(subset)
            pct_cohort = (count / n_total * 100) if n_total > 0 else 0.0
            mean_prob = float(subset["probability"].mean()) if count > 0 else 0.0

            rec: dict[str, Any] = {
                "risk_tier": t.name,
                "probability_range": f"[{t.min_prob:.2f}, {t.max_prob:.2f}" + ("]" if t == self._tier_defs[-1] else ")"),
                "account_count": count,
                "percent_of_cohort": round(pct_cohort, 2),
                "mean_model_probability": round(mean_prob, 4),
            }

            if y_true is not None:
                obs_churn = float(subset["actual_churn"].mean()) if count > 0 else 0.0
                rec["observed_churn_rate"] = round(obs_churn, 4)
                rec["actual_churned_accounts"] = int(subset["actual_churn"].sum())

            rec["description"] = t.description
            records.append(rec)

        return pd.DataFrame(records)

