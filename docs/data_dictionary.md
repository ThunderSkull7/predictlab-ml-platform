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

## 3. Pending Preprocessing Decision: `TotalCharges` Whitespace Handling

- **Audit Findings**: Exactly 11 rows contain `" "` in `TotalCharges`. All 11 records have `tenure == 0` and `Churn == "No"`.
- **Integrity Status**: The raw CSV file in `data/raw/` remains completely unmodified. No silent conversions are applied during Phase 1.
- **Pending Decision**: The proposal to convert these 11 whitespace entries to `0.0` (reflecting zero accumulated billing at initial subscription) or evaluate alternative imputations is formally recorded as a **pending preprocessing decision**. It will be investigated during Phase 2 exploratory data analysis and implemented inside Phase 3 preprocessing pipelines.
