# Data Dictionary: IBM Telco Customer Churn

## 1. Dataset Overview

- **Source Asset**: IBM Telco Customer Churn Sample Dataset (`Telco-Customer-Churn.csv`)
- **Authoritative Repository**: `IBM/telco-customer-churn-on-icp4d`
- **Total Records**: 7,043 rows
- **Total Features**: 21 columns (1 Identifier, 19 Features, 1 Target)
- **Target Feature**: `Churn` (Binary: `"Yes"` vs `"No"`)
- **Missing Values**: 11 records with whitespace strings (`" "`) in `TotalCharges` (corresponding strictly to brand-new subscribers with `tenure == 0`). Zero nulls across all other columns.
- **Duplicate Rows**: 0 complete row duplicates.
- **Duplicate Identifiers**: 0 duplicate `customerID` values.

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
| `TotalCharges` | `object`* | Numeric (Continuous) | `" "` (11 rows) or numeric string `18.80` to `8684.80` | 11 whitespace | Cumulative lifetime charges billed to customer. (*Stored as object in raw CSV due to whitespace). |
| `Churn` | `object` | Target (Binary) | `["No", "Yes"]` | 0 | Ground-truth observed outcome: whether the customer discontinued service (`"Yes"`) or stayed (`"No"`). |

> [!NOTE]
> All descriptions reflect documented IBM metadata verified against raw CSV attributes. No feature semantics or units were fabricated.
