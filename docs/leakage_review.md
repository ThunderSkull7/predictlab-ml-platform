# Data Leakage & Feature Review: Telco Churn Dataset

## 1. Executive Summary

Data leakage occurs when information from outside the training dataset or information unavailable at prediction time is inadvertently used to train a machine learning model. Leakage causes unrealistically optimistic offline validation metrics and severe degradation during production deployment.

This review systematically audits every feature in the IBM Telco Customer Churn dataset to evaluate:
1. **Target Leakage**: Features that reveal or approximate the outcome variable because they are recorded or updated as a consequence of the event.
2. **Temporal Leakage**: Features that incorporate information from future timestamps relative to the prediction cutoff point.
3. **Identifier Overfitting / Memorization**: High-cardinality keys that encode sample identity rather than generalizable behavioral patterns.

---

## 2. Feature-by-Feature Leakage Audit

### A. Identifier: `customerID`
- **Assessment**: High Risk of memorization / pseudo-leakage.
- **Audit Findings**: `customerID` contains 7,043 unique alphanumeric strings (100% unique).
- **Decision**: **Mandatory exclusion from model feature matrix**.
- **Justification**: Customer IDs have no generalized predictive value across future subscribers. Any split where IDs are encoded numerically or memorized by deep trees would create spurious associations and severe overfitting.

### B. Financial Aggregates: `TotalCharges` & `MonthlyCharges`
- **Assessment**: Low Leakage Risk, High Preprocessing Importance.
- **Audit Findings**:
  - `MonthlyCharges` is continuous float representing the current contracted rate.
  - `TotalCharges` represents historical cumulative billed revenue up to the observation point.
  - **Whitespace Anomaly**: Exactly 11 rows contain `" "` (whitespace string) in `TotalCharges`. All 11 records have `tenure == 0`. These represent brand-new subscribers who have enrolled but not yet received their first monthly bill.
- **Leakage Question**: Does `TotalCharges` leak churn?
  - *Analysis*: If `TotalCharges` were frozen at the moment of cancellation while retained customers kept accumulating charges indefinitely into the future, comparing `TotalCharges` across uneven future windows would introduce temporal leakage. However, in this cross-sectional snapshot, `TotalCharges` corresponds to tenure at the observation point ($\approx \text{tenure} \times \overline{\text{MonthlyCharges}}$).
- **Decision**: Retain for modeling; during Phase 3 preprocessing, parse whitespace as `0.0` (reflecting zero historical charges billed at `tenure == 0`) or evaluate median imputation.

### C. Account Duration: `tenure`
- **Assessment**: No Leakage Risk (Valid Predictive Feature).
- **Audit Findings**: Discrete integer between 0 and 72 months.
- **Analysis**: Tenure is known at the instant of prediction. Churn likelihood is strongly tied to subscriber lifecycle (hazard rate is highest in months 1–6). Retaining `tenure` does not violate temporal ordering as long as the evaluation point is fixed.

### D. Contractual Terms: `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Assessment**: No Leakage Risk (Valid Predictive Features).
- **Audit Findings**:
  - `Contract` (`Month-to-month`, `One year`, `Two year`).
  - Correlation with churn is pronounced (Month-to-month customers churn at ~42.7%, while Two-year customers churn at ~2.8%).
- **Leakage Question**: Does `Contract` constitute target leakage?
  - *Analysis*: No. The contract structure is agreed upon *in advance* of service provision. A customer on a month-to-month contract possesses the contractual freedom to leave at any billing cycle, making it a powerful operational risk factor, not an artifact created by cancellation.
- **Decision**: Retain as core predictive features.

### E. Service Subscription Bundles: `PhoneService`, `InternetService`, Add-ons
- **Features**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Assessment**: No Leakage Risk.
- **Analysis**: These indicate active service configurations as of the prediction cutoff. If an add-on were automatically cancelled two weeks before contract termination and recorded as "No", that would introduce leakage. However, in this dataset, these represent baseline service tiers.
- **Decision**: Retain all service bundle indicators.

### F. Demographic Features: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Assessment**: No Leakage Risk (Exogenous Customer Attributes).
- **Analysis**: Demographics are collected at account origination and remain static or slow-changing.
- **Decision**: Retain for exploration. In Phase 2 and Phase 4, fairness and disparate impact metrics will be audited to ensure model fairness across demographic subgroups.

---

## 3. Potential Post-Event Features (Expanded Dataset Comparison)

Some later variants of the Telco churn dataset (e.g., IBM Cognos / Kaggle Telco Churn v2) contain columns such as:
- `Churn Category` (e.g., "Competitor", "Price", "Dissatisfaction")
- `Churn Reason` (e.g., "Competitor offered higher download speeds")
- `Customer Status` (`"Stayed"`, `"Churned"`, `"Joined"`)

> [!CAUTION]
> Features like `Churn Reason` or `Customer Status` are post-hoc exit survey artifacts that exist **only because churn already occurred**. Including them in training would yield trivial 100% accuracy and complete operational failure.
>
> In our verified dataset (`Telco-Customer-Churn.csv`), **none of these post-event fields are present**. The only outcome indicator is the binary target `Churn`.

---

## 4. Formal Prediction-Time Assumptions

1. **Prediction Cutoff**: The model scores accounts at the end of each billing cycle, prior to invoicing and contract renewal windows.
2. **Feature Availability**: All 19 candidate features (demographics, services, tenure, billing configurations, charges) are fully accessible in billing/CRM systems at the cutoff time.
3. **Partitioning Guardrails**:
   - Out-of-fold and train-test splits must fit all transformers, encoders, and scalers **strictly on the training partition**.
   - No target statistics or aggregate statistics from the validation/test partition may bleed into the training pipeline.
