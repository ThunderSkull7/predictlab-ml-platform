# Problem Definition: Customer Churn Prediction

## 1. Executive Summary & Business Problem

Customer attrition ("churn") represents the permanent or extended loss of a subscriber or customer to competitors or natural disengagement. In subscription-based telecommunications, customer acquisition costs typically exceed retention costs by 4x to 7x. Losing existing subscribers erodes recurring revenue, depresses customer lifetime value (LTV), and impairs profitability.

When a customer cancels their subscription, the loss is irreversible for that billing cycle. Organizations that rely purely on reactive retention—such as countering cancellation requests at the call center—experience low win-back rates and heavy discount cannibalization.

PredictLab addresses this challenge through **proactive risk identification**, modeling subscriber behavior to estimate churn probability prior to contract expiration or cancellation triggers.

## 2. Why Churn Prediction is Useful

1. **Targeted Retention Interventions**: Customer retention budgets are finite. Rather than indiscriminately issuing discounts to healthy accounts, marketing and retention teams can focus interventions (e.g., proactive loyalty incentives, tech support outreach, plan restructuring) on accounts at genuine risk.
2. **Prioritization by Expected Financial Impact**: By pairing predicted churn probability with customer monthly charges ($P(\text{churn}) \times \text{MonthlyCharges}$), businesses can rank-order at-risk accounts by expected revenue loss.
3. **Operational Root-Cause Diagnostics**: Model feature attributions allow customer success representatives to identify underlying pain points (e.g., month-to-month contracts without support bundles, payment method frictions).

## 3. Machine Learning Objective

The machine learning task is formulated as a **supervised binary probabilistic classification problem**:

$$\hat{y}_i = P(Y_i = 1 \mid \mathbf{x}_i)$$

Where:
- $\mathbf{x}_i$ is the vector of observable customer features for customer $i$ at the evaluation point.
- $Y_i \in \{0, 1\}$ represents the observed retention outcome over the target evaluation window.
- $\hat{y}_i \in [0.0, 1.0]$ is the calibrated posterior probability that customer $i$ will churn.

> [!IMPORTANT]
> The primary output of PredictLab is a **well-calibrated probability score** alongside decile risk bands, not merely a thresholded binary label. Binary classification thresholds ($\tau = 0.5$) discard uncertainty and prevent cost-sensitive business trade-offs.

## 4. Formal Problem Specification

| Dimension | Specification |
|---|---|
| **Unit of Prediction** | Individual customer subscription account (identified by `customerID`). |
| **Prediction Point** | At the billing cycle close, before renewal or cancellation actions. |
| **Prediction Target** | `Churn` (categorical: `"Yes"` vs `"No"`). |
| **Positive Class ($Y=1$)** | `"Yes"`: The customer terminated their contract/service. |
| **Negative Class ($Y=0$)** | `"No"`: The customer maintained an active subscription. |
| **Model Output** | Calibrated probability $\hat{p} \in [0, 1]$, risk tier (`Low`, `Medium`, `High`, `Critical`), and top contributing risk factors. |
| **Optimization Metrics** | Area Under the Precision-Recall Curve (PR-AUC), Brier Score, ROC-AUC, and Expected Value Optimization under asymmetric costs. |

## 5. Intended Users & Stakeholders

- **Retention Marketing Managers**: To design targeted loyalty offers and promotional campaigns for cohorts in the upper risk deciles.
- **Customer Success / Support Specialists**: To review customer risk profiles during inbound or outbound support calls and apply proactive service remediation.
- **Executive Leadership / RevOps**: To monitor portfolio-level churn velocity, aggregate revenue-at-risk, and assess retention program effectiveness.

## 6. Expected Application Use Cases

1. **Proactive Campaign Targeting**: Weekly/monthly batch scoring to identify high-risk customer cohorts for automated email campaigns or loyalty reward credits.
2. **Interactive Customer Health Inspection**: An interactive dashboard (Streamlit UI in Phase 6) enabling customer service teams to look up an account, view its predicted churn probability, understand key risk drivers, and simulate retention scenarios ("What if contract length is extended to 1 year?").
3. **Cohort Risk Segmentation**: Aggregating accounts by tenure, contract type, and service bundles to pinpoint systemic service vulnerabilities (e.g., fiber optic customers lacking tech support).

## 7. Assumptions & Limitations

### Probabilistic Nature & Absence of Certainty
- Predictions represent statistical likelihoods, not certain futures. A customer with $\hat{p} = 0.85$ has an 85% estimated likelihood of churn under historical behavior; individual decisions remain subject to external events.

### Correlation vs. Causation
- The machine learning model identifies statistical associations, not causal mechanisms. For example, customers on month-to-month contracts exhibit high churn rates, but contract type may simply be a proxy for lower initial commitment rather than the root cause. Interventions must be designed with domain judgment.

### Temporal Horizon & Snapshot Limitations
- The IBM Telco dataset is a cross-sectional snapshot containing aggregated features (`tenure`, `MonthlyCharges`, `TotalCharges`) rather than longitudinal event streams. The exact observation window prior to churn is fixed by dataset construction.

### Data Stationarity
- The model assumes the relationship between customer characteristics and churn remains stationary over time. Macroeconomic shifts, competitive pricing changes, or service disruptions may cause concept drift requiring model recalibration.
