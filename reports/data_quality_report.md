# PredictLab — Data Quality & Integrity Report

## 1. Executive Summary

This report establishes a rigorous empirical audit of the canonical IBM Telco Customer Churn sample dataset (`data/raw/Telco-Customer-Churn.csv`) conducted during Phase 2 of PredictLab.

The dataset comprises **7,043 customer records** and **21 columns** (1 unique identifier, 19 predictive candidate features, and 1 ground-truth target variable).

### Key Audit Highlights
- **Zero Duplicate Rows**: The dataset contains 0 duplicate records across all 21 columns.
- **Identifier Integrity**: The `customerID` attribute contains exactly 7,043 unique alphanumeric strings (100% uniqueness rate, 0 duplicate keys).
- **Missing Value Profile**: Standard pandas `.isna()` checks report zero missing values across all columns.
- **Whitespace Anomaly**: An explicit string-level scan revealed that `TotalCharges` contains exactly **11 records with whitespace-only strings** (`" "`).
- **Anomaly Origin**: All 11 whitespace records strictly coincide with **`tenure == 0`** and **`Churn == "No"`**.
- **Data Preservation**: The raw CSV file in `data/raw/` remains **completely unmodified**.

---

## 2. Structural Quality & Schema Audit

| Metric | Measured Value | Validation Status | Notes |
|---|---|---|---|
| **Total Observations** | 7,043 | Confirmed | Complete cross-sectional cohort |
| **Total Attributes** | 21 | Confirmed | 1 ID, 19 Features, 1 Target |
| **Complete Row Duplicates** | 0 | PASS | No repeated rows |
| **Identifier Duplicates** | 0 | PASS | 7,043 unique `customerID` values |
| **Memory Footprint** | ~5.89 MB | Confirmed | In-memory uncompressed representation |
| **Target Presence** | `Churn` | PASS | Ground-truth outcome present |
| **Target Nulls** | 0 | PASS | 100% complete labels |

### Column-Level Data Type & Quality Summary

| Column Name | Stored Dtype | Semantic Type | Null Count | Whitespace Count | Unique Values | Sample Values |
|---|---|---|---|---|---|---|
| `customerID` | `object` | Identifier | 0 (0.0%) | 0 | 7,043 | `7590-VHVEG`, `5575-GNVDE`, `3668-QPYBK` |
| `gender` | `object` | Binary Categorical | 0 (0.0%) | 0 | 2 | `Female`, `Male` |
| `SeniorCitizen` | `int64` | Binary Categorical | 0 (0.0%) | 0 | 2 | `0`, `1` |
| `Partner` | `object` | Binary Categorical | 0 (0.0%) | 0 | 2 | `Yes`, `No` |
| `Dependents` | `object` | Binary Categorical | 0 (0.0%) | 0 | 2 | `No`, `Yes` |
| `tenure` | `int64` | Discrete Numeric | 0 (0.0%) | 0 | 73 | `1`, `34`, `2`, `45`, `8` |
| `PhoneService` | `object` | Binary Categorical | 0 (0.0%) | 0 | 2 | `No`, `Yes` |
| `MultipleLines` | `object` | Ternary Categorical | 0 (0.0%) | 0 | 3 | `No phone service`, `No`, `Yes` |
| `InternetService` | `object` | Nominal Categorical | 0 (0.0%) | 0 | 3 | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | `object` | Ternary Categorical | 0 (0.0%) | 0 | 3 | `No`, `Yes`, `No internet service` |
| `OnlineBackup` | `object` | Ternary Categorical | 0 (0.0%) | 0 | 3 | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | `object` | Ternary Categorical | 0 (0.0%) | 0 | 3 | `No`, `Yes`, `No internet service` |
| `TechSupport` | `object` | Ternary Categorical | 0 (0.0%) | 0 | 3 | `No`, `Yes`, `No internet service` |
| `StreamingTV` | `object` | Ternary Categorical | 0 (0.0%) | 0 | 3 | `No`, `Yes`, `No internet service` |
| `StreamingMovies` | `object` | Ternary Categorical | 0 (0.0%) | 0 | 3 | `No`, `Yes`, `No internet service` |
| `Contract` | `object` | Nominal Categorical | 0 (0.0%) | 0 | 3 | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | `object` | Binary Categorical | 0 (0.0%) | 0 | 2 | `Yes`, `No` |
| `PaymentMethod` | `object` | Nominal Categorical | 0 (0.0%) | 0 | 4 | `Electronic check`, `Mailed check`, `Bank transfer (automatic)` |
| `MonthlyCharges` | `float64` | Continuous Numeric | 0 (0.0%) | 0 | 1,585 | `29.85`, `56.95`, `53.85`, `42.30` |
| `TotalCharges` | `object`* | Continuous Numeric | 0 (0.0%) | **11** (0.16%) | 6,531 | `29.85`, `1889.5`, `108.15`, `" "` |
| `Churn` | `object` | Binary Target | 0 (0.0%) | 0 | 2 | `No`, `Yes` |

*\* Note: Stored as object in raw CSV due to 11 whitespace entries.*

---

## 3. Mandatory Empirical Investigation: `TotalCharges`

### A. Observations & Profile of the 11 Affected Rows
An exhaustive row-level inspection of the 11 whitespace records revealed the following empirical properties:

| Row Index | `customerID` | `tenure` | `MonthlyCharges` | `TotalCharges` (Raw) | `Contract` | `PaymentMethod` | `Churn` |
|---|---|---|---|---|---|---|---|
| **488** | `4472-LVYGI` | **0** | $52.55 | `" "` | Two year | Bank transfer (automatic) | **No** |
| **753** | `3115-CZMZD` | **0** | $20.25 | `" "` | Two year | Mailed check | **No** |
| **936** | `5709-LVOEQ` | **0** | $80.85 | `" "` | Two year | Mailed check | **No** |
| **1082** | `4367-NUYAO` | **0** | $25.75 | `" "` | Two year | Mailed check | **No** |
| **1340** | `1371-DWPAZ` | **0** | $56.05 | `" "` | Two year | Credit card (automatic) | **No** |
| **3331** | `7644-OMVMY` | **0** | $19.85 | `" "` | Two year | Mailed check | **No** |
| **3826** | `3213-VVOLG` | **0** | $25.35 | `" "` | Two year | Mailed check | **No** |
| **4380** | `2520-SGTTA` | **0** | $20.00 | `" "` | Two year | Mailed check | **No** |
| **5218** | `2923-ARZLG` | **0** | $19.70 | `" "` | One year | Mailed check | **No** |
| **6670** | `4075-WKNIU` | **0** | $73.35 | `" "` | Two year | Mailed check | **No** |
| **6754** | `2775-SEFEE` | **0** | $61.90 | `" "` | Two year | Bank transfer (automatic) | **No** |

### B. Empirical Deductions
1. **Zero Tenure Coincidence**: Exactly 11 customers in the entire dataset have `tenure == 0`. All 11 have whitespace in `TotalCharges`. No other customer has `tenure == 0`, and no other customer has whitespace in `TotalCharges`.
2. **Contract Structure**: 10 accounts have a **Two year** contract; 1 has a **One year** contract. None are on a Month-to-month contract.
3. **Retention Status**: 100% of these customers have **`Churn == "No"`**.
4. **Consistency at `tenure == 1`**:
   - For all 613 customers in the dataset with `tenure == 1`, $\text{TotalCharges} == \text{MonthlyCharges}$ ($\Delta = 0.000$ across all 613 rows).
   - This proves that `TotalCharges` accumulates completed monthly billing cycles.
5. **Physical Meaning**: Subscribers with `tenure == 0` are brand-new signups who have entered into long-term contracts but have not completed their first billing cycle. At tenure zero, cumulative historical billed revenue is mathematically and operationally **$0.00**.

### C. Evaluation of Alternative Treatments

| Treatment Option | Operational Impact | Risk / Bias Assessment | Recommendation Status |
|---|---|---|---|
| **Option 1: Drop 11 Rows** | Removes 11 valid observations (0.16% of dataset). | Discards valuable examples of long-term contract signups; induces slight survivorship selection bias. | **Rejected** |
| **Option 2: Mean / Median Imputation** | Replaces whitespace with overall mean ($2,283.30) or median ($1,397.47). | Severely corrupts data; assigns thousands of dollars of historical billing to accounts that just enrolled (tenure = 0). | **Rejected (Severe Distortion)** |
| **Option 3: Impute with `MonthlyCharges`** | Sets `TotalCharges` = `MonthlyCharges`. | Overstates historical billing; assumes customer has already completed month 1 when `tenure == 0`. | **Rejected** |
| **Option 4: Impute with `0.0`** | Sets `TotalCharges = 0.0` for all `tenure == 0` rows. | Exactly matches business reality ($\text{TotalCharges} = \text{tenure} \times \overline{\text{MonthlyCharges}} = 0$). Perfectly preserves all 7,043 rows. | **Recommended for Phase 3** |

---

## 4. Numerical Outliers & Distribution Integrity

Descriptive statistics were computed across all numerical attributes (with `TotalCharges` whitespace evaluated as NaN for descriptive profiling):

| Attribute | Count | Mean | Std | Min | 25% | Median | 75% | Max | IQR | Skewness |
|---|---|---|---|---|---|---|---|---|---|---|
| `tenure` | 7,043 | 32.37 | 24.56 | 0.00 | 9.00 | 29.00 | 55.00 | 72.00 | 46.00 | +0.24 |
| `MonthlyCharges` | 7,043 | 64.76 | 30.09 | 18.25 | 35.50 | 70.35 | 89.85 | 118.75 | 54.35 | -0.22 |
| `TotalCharges` | 7,032 | 2283.30 | 2266.77 | 18.80 | 401.45 | 1397.47 | 3794.74 | 8684.80 | 3393.29 | +0.96 |

### Outlier Assessment
- **`tenure`**: Range 0 to 72 months (6 years). Distribution is bimodal with peaks at new signups (months 1–6) and long-tenured loyalists (months 65–72). Values represent genuine customer lifecycle stages; no truncation or capping required.
- **`MonthlyCharges`**: Range $18.25 to $118.75. Plausible telecommunications billing range reflecting low-cost landlines to premium fiber optic + entertainment bundles. No unphysical negatives or extreme outliers.
- **`TotalCharges`**: Range $18.80 to $8,684.80. Positive skewness (+0.96) reflects the compound product of tenure and monthly rate. The maximum value ($8,684.80) corresponds to a customer with tenure = 72 months paying ~$118/month ($72 \times 118 \approx \$8,500$), demonstrating complete physical plausibility.

---

## 5. Schema Concerns & Preprocessing Directives for Phase 3

1. **`customerID` Exclusion**: Must be dropped from all modeling matrices to eliminate identifier memorization.
2. **`TotalCharges` Conversion Directive**: The canonical raw CSV remains byte-unmodified. In the Phase 3 preprocessing pipeline, parse whitespace strings (`" "`) to `NaN`, validate that missingness strictly and exclusively coincides with `tenure == 0` (raising a validation error if missing values occur for accounts with `tenure > 0`), and impute `0.0` before casting to `float64`.
3. **`Contract` Encoding Directive**: One-Hot Encoding must be the default baseline treatment. Ordinal encoding may be evaluated only as an alternative experiment, with documented equal-spacing assumptions.
4. **Feature Engineering Clarifications**: Feature engineering plans should implement `OptionalServiceCount` across all 6 optional services (scoring "No internet service" as 0) and treat `ChargesRatio` strictly as an experimental candidate requiring explicit zero-tenure handling rather than an arbitrary epsilon.
5. **Multicollinearity Guardrail**: Strong correlation between `tenure` and `TotalCharges` (Spearman $\rho = 0.889$) requires awareness in linear models (ridge/lasso penalty) while remaining benign for tree-based estimators.
6. **Ternary Category Separation**: Categories like `"No internet service"` and `"No phone service"` represent structural service dependencies and must not be collapsed blindly into `"No"`.
