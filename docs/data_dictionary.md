# Data Dictionary: IBM Telco Customer Churn

## 1. Dataset Overview & Provenance Audit

- **Source Asset**: IBM Telco Customer Churn Sample Dataset (`Telco-Customer-Churn.csv`)
- **Host Repository**: `IBM/telco-customer-churn-on-icp4d`
- **Direct Source URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Retrieval Date**: `2026-11-07`
- **Repository License**: Apache License 2.0 (governing code pattern software).
- **Data Terms & Provenance Assessment**: The repository does not include a separate open-data license (e.g., CDLA, CC0, CC-BY) explicitly defining dataset redistribution terms. IBM hosts the file as sample demonstration data for a tutorial rather than publishing an official open data asset. Redistribution rights cannot be conclusively established from upstream metadata alone. PredictLab maintains its code under the MIT License and manages the dataset as an external sample asset with a reproducible download procedure.
- **SHA-256 Checksum**: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Total Records**: 7,043 rows
- **Total Features**: 21 columns (1 Identifier, 19 Features, 1 Target)
- **Target Feature**: `Churn` (Binary: `"Yes"` vs `"No"`)
- **Missing Values**: 11 records with whitespace strings (`" "`) in `TotalCharges` (coinciding strictly with brand-new subscribers where `tenure == 0`). Zero nulls across all other columns.
- **Duplicate Rows**: 0 complete row duplicates.
- **Duplicate Identifiers**: 0 duplicate `customerID` values.

---

## 2. Feature Definitions & Metadata

| Column Name | Raw Type | Semantic Type | Allowable / Observed Values | Missing Count | Description |
|---|---|---|---|---|---|
| `customerID` | `object` | Identifier | 7,043 unique alphanumeric strings (e.g., `"7590-VHVEG"`) | 0 | Unique customer account identifier. **Must be excluded from feature modeling** to prevent memorization. |
| `gender` | `object` | Categorical (Binary) | `["Female", "Male"]` | 0 | Customer demographic gender. |
| `SeniorCitizen` | `int64` | Categorical (Binary) | `[0, 1]` | 0 | Binary indicator denoting whether the customer is 65 years or older (1 = Yes, 0 = No). |
| `Partner` | `object` | Categorical (Binary) | `["Yes", "No"]` | 0 | Indicates whether the customer has a domestic partner or spouse. |
| `Dependents` | `object` | Categorical (Binary) | `["Yes", "No"]` | 0 | Indicates whether the customer has financial dependents (children, elderly relatives). |
| `tenure` | `int64` | Numeric (Discrete) | `0` to `72` (months) | 0 | Number of continuous months the customer has subscribed to the company. |
| `PhoneService` | `object` | Categorical (Binary) | `["Yes", "No"]` | 0 | Indicates whether the customer subscribes to landline telephone service. |
| `MultipleLines` | `object` | Categorical (Ternary) | `["No", "Yes", "No phone service"]` | 0 | Indicates whether customer subscribes to multiple telephone lines. |
| `InternetService` | `object` | Categorical (Nominal) | `["DSL", "Fiber optic", "No"]` | 0 | Primary internet service protocol provider. |
| `OnlineSecurity` | `object` | Categorical (Ternary) | `["No", "Yes", "No internet service"]` | 0 | Indicates whether the customer subscribes to online security threat protection add-on. |
| `OnlineBackup` | `object` | Categorical (Ternary) | `["No", "Yes", "No internet service"]` | 0 | Indicates whether customer subscribes to offsite cloud backup service. |
| `DeviceProtection` | `object` | Categorical (Ternary) | `["No", "Yes", "No internet service"]` | 0 | Indicates whether customer subscribes to hardware/device insurance protection. |
| `TechSupport` | `object` | Categorical (Ternary) | `["No", "Yes", "No internet service"]` | 0 | Indicates whether customer has priority technical support assistance. |
| `StreamingTV` | `object` | Categorical (Ternary) | `["No", "Yes", "No internet service"]` | 0 | Indicates whether customer uses internet connection to stream television programming. |
| `StreamingMovies` | `object` | Categorical (Ternary) | `["No", "Yes", "No internet service"]` | 0 | Indicates whether customer uses internet connection to stream movies. |
| `Contract` | `object` | Categorical (Nominal) | `["Month-to-month", "One year", "Two year"]` | 0 | Customer's contractual commitment duration. |
| `PaperlessBilling` | `object` | Categorical (Binary) | `["Yes", "No"]` | 0 | Indicates whether customer opted into digital paperless invoicing. |
| `PaymentMethod` | `object` | Categorical (Nominal) | `["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]` | 0 | Invoicing settlement mechanism. |
| `MonthlyCharges` | `float64` | Numeric (Continuous) | `18.25` to `118.75` (USD) | 0 | Current monthly bill amount charged to the subscriber. |
| `TotalCharges` | `object`* | Numeric (Continuous) | `" "` (11 rows) or numeric string `18.80` to `8684.80` | 11 whitespace | Cumulative lifetime charges billed to customer. (*Stored as object in raw CSV due to 11 whitespace strings). |
| `Churn` | `object` | Target (Binary) | `["No", "Yes"]` | 0 | Historical ground-truth status: customer discontinued service (`"Yes"`) or stayed (`"No"`). Documented in IBM materials as "Customers who left within the last month". |

---

## 3. Preprocessing Recommendation: `TotalCharges` Whitespace Handling & Feature Directives

- **Audit Findings**: Exactly 11 rows contain `" "` in `TotalCharges`. All 11 records have `tenure == 0` and `Churn == "No"` (10 on Two-year contracts, 1 on One-year contract).
- **Integrity Status**: The raw CSV file in `data/raw/` remains completely unmodified.
- **Empirical Proof**: For all 613 accounts with `tenure == 1`, $\text{TotalCharges} == \text{MonthlyCharges}$ ($\Delta = 0.00$), proving that TotalCharges represents accumulated completed billing cycles.
- **Phase 2 Conclusion**: At enrollment (`tenure == 0`), accumulated billed revenue is operationally $0.00. In Phase 3, the preprocessing pipeline will convert whitespace to `NaN`, validate that missingness coincides exclusively with `tenure == 0` (raising an error or warning if missingness occurs for positive tenure), and impute `0.0`.
- **Encoding & Feature Directives**:
  - `Contract`: One-Hot Encoding is the default treatment. Ordinal encoding may be evaluated only as an alternative experiment with documented linear spacing assumptions.
  - `OptionalServiceCount`: Renamed from `SupportBundleCount`, counting active optional services across 6 internet add-ons (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`), scoring `"No internet service"` as 0.
  - `ChargesRatio`: Retained strictly as an experimental candidate; requires explicit zero-tenure handling rather than an arbitrary epsilon.

---

## 4. Phase 2 Verified Distribution Statistics

### Numerical Distribution Profiles

| Feature | Count | Mean | Std | Min | 25% | Median | 75% | Max | IQR | Skewness | Churned Median | Retained Median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `tenure` | 7,043 | 32.37 | 24.56 | 0.00 | 9.00 | 29.00 | 55.00 | 72.00 | 46.00 | +0.24 | 10.00 | 38.00 |
| `MonthlyCharges` | 7,043 | 64.76 | 30.09 | 18.25 | 35.50 | 70.35 | 89.85 | 118.75 | 54.35 | -0.22 | 79.65 | 64.43 |
| `TotalCharges`* | 7,032 | 2283.30 | 2266.77 | 18.80 | 401.45 | 1397.47 | 3794.74 | 8684.80 | 3393.29 | +0.96 | 701.50 | 1683.60 |

*\* Evaluated across the 7,032 valid numerical rows.*

### Categorical Churn Associations (Ranked by Cramér's V)

| Feature | Primary Levels | Frequency Distribution | Segment Churn Rate (%) | Cramér's $V$ | FDR $p$-Value |
|---|---|---|---|---|---|
| `Contract` | Month-to-month, One year, Two year | 3875 (55%), 1473 (21%), 1695 (24%) | 42.71%, 11.27%, 2.83% | **0.408** | $< 10^{-200}$ |
| `OnlineSecurity` | No, Yes, No internet | 3498 (50%), 2019 (29%), 1526 (22%) | 41.77%, 14.61%, 7.40% | **0.347** | $< 10^{-180}$ |
| `TechSupport` | No, Yes, No internet | 3473 (49%), 2044 (29%), 1526 (22%) | 41.65%, 15.17%, 7.40% | **0.344** | $< 10^{-180}$ |
| `InternetService` | Fiber optic, DSL, No | 3096 (44%), 2421 (34%), 1526 (22%) | 41.89%, 18.96%, 7.40% | **0.320** | $< 10^{-155}$ |
| `PaymentMethod` | Electronic check, Mailed check, Bank auto, Card auto | 2365 (34%), 1612 (23%), 1544 (22%), 1522 (22%) | 45.29%, 19.11%, 16.71%, 15.24% | **0.302** | $< 10^{-135}$ |
