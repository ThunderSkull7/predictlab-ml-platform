# PredictLab — Phase 3 Preprocessing, Feature Engineering & Transformation Report

## 1. Executive Summary

This report documents the architecture, methodological safeguards, and empirical validation of **Phase 3: Leakage-Free Preprocessing, Feature Engineering & Transformation Pipeline** for PredictLab.

Phase 3 transitions the project from exploratory data analysis to production-grade, scikit-learn-compatible data preparation for downstream classification modeling in Phase 4. All transformations adhere strictly to leakage-prevention contracts:
- Training and holdout test partitions are isolated prior to any learned statistical estimation.
- Deterministic data-cleaning rules are cleanly decoupled from learned statistical transformers.
- The canonical raw dataset [`data/raw/Telco-Customer-Churn.csv`](file:///d:/Project/Predictive%20Analytics%20Platform/data/raw/Telco-Customer-Churn.csv) remains 100% byte-unmodified.
- No model training, hyperparameter tuning, cross-validation, or classification metric calculations were executed in this phase.

---

## 2. Dataset Contract & Leakage-Safe Partitioning

### Feature and Target Separation
- **Target Variable**: `Churn`
  - Extracted prior to feature transformation.
  - Mapped deterministically to binary integers: `{"No": 0, "Yes": 1}`.
  - Target is never passed into feature preprocessing transformers.
- **Excluded Identifier**: `customerID`
  - 100% unique string identifier.
  - Strictly excluded from feature matrices $X_{\text{train}}$ and $X_{\text{test}}$ to prevent memorization and false spatial clustering.
  - Stored exclusively in the partition manifest for record tracking and error auditing.

### Reproducible Stratified Holdout Split
- **Holdout Ratio**: 80% Training ($N = 5,634$), 20% Testing ($N = 1,409$).
- **Stratification Variable**: `Churn` (preserving exact class balance).
- **Deterministic Random Seed**: `random_state = 42` (configured centrally in [`Settings`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/config.py#L88)).
- **Class Balance Preservation**:
  - Raw Population: 26.5369% Churn ($N = 1,869 / 7,043$).
  - Training Partition: **26.5353%** Churn ($N = 1,495 / 5,634$).
  - Holdout Test Partition: **26.5436%** Churn ($N = 374 / 1,409$).
  - Class balance is preserved within $0.007\%$ delta across partitions.
- **Customer ID Overlap**: Exactly **0** intersecting customer IDs between training and testing sets ($| \text{IDs}_{\text{train}} \cap \text{IDs}_{\text{test}} | = 0$).
- **Manifest Persistence**: Serialized as a lightweight JSON document to [`data/splits/split_manifest.json`](file:///d:/Project/Predictive%20Analytics%20Platform/data/splits/split_manifest.json) containing partition metadata, checksums, and customer ID lists, avoiding redundant data duplication.

---

## 3. Mandatory Data Quality Requirement: `TotalCharges` Cleaning

### Domain Accounting Context & Justification
- In the canonical dataset, exactly 11 records contain whitespace strings (`" "`) in `TotalCharges`.
- During Phase 2 EDA, empirical auditing proved that:
  1. All 11 whitespace records strictly coincide with $\text{tenure} == 0$ ($Churn = No$).
  2. For all 613 accounts with $\text{tenure} == 1$, $\text{TotalCharges} == \text{MonthlyCharges}$ ($100\%$ exact match, $\Delta = 0.00$).
  3. This demonstrates that `TotalCharges` records cumulative completed monthly billing cycles.
  4. Brand-new accounts at enrollment ($\text{tenure} == 0$) have received zero billing cycles. Their accumulated historical billed revenue is physically and operationally $\$0.00$.
- **Distinction from Universal Rule**: Setting missing total charges to 0.0 is justified here by empirical proof of subscription billing cycles, not as a universal default for telecommunications data.

### Preprocessing Pipeline Implementation
Implemented in [`clean_total_charges`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/preprocessing/cleaning.py#L18) and [`TotalChargesCleaner`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/preprocessing/transformers.py#L25):
1. Detects whitespace-only strings and maps them to `np.nan`.
2. Converts valid numeric strings to `float64`, detecting and raising on malformed non-numeric values.
3. **Strict Validation Guardrail**: Asserts that missing `TotalCharges` values occur *strictly and exclusively* where $\text{tenure} == 0$. If any missing value occurs where $\text{tenure} > 0$, raises an informative [`TotalChargesValidationError`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/preprocessing/cleaning.py#L14).
4. Imputes validated zero-tenure records with `0.0`.
5. Guarantees input DataFrame immutability by operating on an explicit copy.
6. Operates seamlessly when train/test partitioning separates the 11 records (observed distribution: 9 in $X_{\text{train}}$, 2 in $X_{\text{test}}$).

---

## 4. Feature Engineering Architecture

### 4.1 `OptionalServiceCount` (Baseline Feature)
- **Concept**: Quantifies total subscriber ecosystem engagement across optional internet add-on services.
- **Contributing Columns (Exactly 6)**:
  1. `OnlineSecurity`
  2. `OnlineBackup`
  3. `DeviceProtection`
  4. `TechSupport`
  5. `StreamingTV`
  6. `StreamingMovies`
- **Value Mapping**: `"Yes"` $\rightarrow 1$, `"No"` $\rightarrow 0$, `"No internet service"` $\rightarrow 0$.
- **Valid Range**: Discrete integer from $0$ to $6$.
- **Categorical Integrity**: Input columns are validated against allowed categories `{"Yes", "No", "No internet service"}`. Unrecognized strings raise [`InvalidCategoryValueError`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/preprocessing/schema.py#L90) rather than being silently coerced.
- **Baseline Invariance**: All 6 underlying service columns are retained in the baseline pipeline alongside `OptionalServiceCount` to allow non-linear estimators to exploit specific service interactions.

### 4.2 Sub-Aggregations (Optional / Experimental)
- `TechnicalProtectionCount`: Subtotal across `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` (range $0$ to $4$).
- `StreamingEntertainmentCount`: Subtotal across `StreamingTV`, `StreamingMovies` (range $0$ to $2$).
- Excluded from the default baseline to prevent collinear feature duplication. Available via [`create_experimental_preprocessor`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/preprocessing/pipelines.py#L180).

### 4.3 `AutomaticPayment` (Alternative Representation)
- Deterministic binary feature: $1$ if `PaymentMethod` is `"Bank transfer (automatic)"` or `"Credit card (automatic)"`, else $0$.
- Validates that payment method strings match the canonical schema.
- Retained as an alternative experimental feature; excluded from default baseline to avoid collinear redundancy with the 4 one-hot payment method columns.

### 4.4 `ChargesRatio` & `ChargesRatioDefined` (Phase 3.1 Mathematical Validity Correction)
- **Formula**: $\text{TotalCharges} / (\text{tenure} \times \text{MonthlyCharges})$.
- **Status**: **Strictly excluded from default baseline**.
- **Root Cause & Mathematical Validity**:
  - In Phase 3, the initial experimental implementation assigned `ChargesRatio = 1.0` when $\text{tenure} == 0$.
  - This was mathematically invalid: when $\text{tenure} == 0$, both the numerator ($\text{TotalCharges} = 0$) and denominator ($\text{tenure} \times \text{MonthlyCharges} = 0$) evaluate to zero. The expression $0 / 0$ is indeterminate/undefined. Fabricating an artificial value of $1.0$ (or injecting an arbitrary epsilon) falsely implies that newly enrolled customers exhibited historical billing consistency identical to seasoned subscribers.
- **Corrected Zero-Tenure Handling**:
  - When $\text{tenure} == 0$, `ChargesRatio` is strictly represented as **`NaN`** (missing value).
  - Never substitutes $1.0$, an arbitrary epsilon, or fabricated numbers.
  - Inputs are validated: positive-tenure denominators must be positive, and computed ratios must be finite.
- **Companion Binary Indicator: `ChargesRatioDefined`**:
  - A deterministic binary feature derived strictly from predictor columns without target access:
    - `1`: `ChargesRatio` is mathematically defined ($\text{tenure} > 0$ and $\text{MonthlyCharges} > 0$).
    - `0`: `ChargesRatio` is undefined ($\text{tenure} == 0$).
- **Experimental Pipeline Integration**:
  - In [`create_experimental_preprocessor()`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/preprocessing/pipelines.py#L180), `ChargesRatio` and `ChargesRatioDefined` are generated prior to numerical imputation.
  - Undefined zero-tenure ratios remain `NaN` until the numerical imputer (`SimpleImputer(strategy="median")`) inside the column transformer processes them.
  - Imputation statistics (median $\approx 1.0000$) are learned **strictly from training data**.
  - Held-out test data is transformed using the already-fitted training median.
  - `ChargesRatioDefined` remains an explicit separate feature ($0$ for zero tenure, $1$ for positive tenure) and is never overwritten by imputation.

---

## 5. Preprocessing Pipelines & ColumnTransformer Architecture

PredictLab implements two dedicated pipeline factory functions:

### 5.1 Linear-Model-Compatible Pipeline (`create_linear_preprocessor`)
Designed for regularized logistic regression, linear support vector machines, and distance-based estimators:
1. `TotalChargesCleaner`: Deterministic whitespace cleaning, zero-tenure validation, $0.0$ imputation.
2. `OptionalServiceCounter`: Computes `OptionalServiceCount`.
3. `ColumnDropper`: Ensures `customerID` is excluded.
4. `ColumnTransformer`:
   - **Numerical Branch** (4 features: `tenure`, `MonthlyCharges`, `TotalCharges`, `OptionalServiceCount`):
     - `SimpleImputer(strategy="median")`
     - `StandardScaler()` (standardizes to mean $0.0$, variance $1.0$).
   - **Categorical Branch** (16 features: all nominal categoricals including `Contract` and `SeniorCitizen`):
     - `SimpleImputer(strategy="most_frequent")`
     - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

### 5.2 Tree-Model-Compatible Pipeline (`create_tree_preprocessor`)
Designed for decision trees, random forests, gradient boosting, and XGBoost/LightGBM:
1. Same deterministic cleaning and feature engineering steps.
2. `ColumnTransformer`:
   - **Numerical Branch**: `SimpleImputer(strategy="median")` (**NO standard scaling**; natural units preserved: tenure in $[0, 72]$, monthly charges in $[\$18.25, \$118.75]$, total charges in $[\$0, \$8,684.80]$, optional service count in $[0, 6]$).
   - **Categorical Branch**: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

---

## 6. Transformed Feature Dimensions & Schema

Both baseline preprocessor variants transform the 19 raw predictive features into **exactly 47 dense numerical features**:

| Branch | Feature Count | Transformed Features |
|---|---|---|
| **Numerical** | 4 | `num__tenure`, `num__MonthlyCharges`, `num__TotalCharges`, `num__OptionalServiceCount` |
| **Demographics** | 8 | `cat__gender_Female`, `cat__gender_Male`, `cat__SeniorCitizen_0`, `cat__SeniorCitizen_1`, `cat__Partner_No`, `cat__Partner_Yes`, `cat__Dependents_No`, `cat__Dependents_Yes` |
| **Phone Services** | 5 | `cat__PhoneService_No`, `cat__PhoneService_Yes`, `cat__MultipleLines_No`, `cat__MultipleLines_No phone service`, `cat__MultipleLines_Yes` |
| **Internet & Add-ons** | 21 | `cat__InternetService_DSL`, `cat__InternetService_Fiber optic`, `cat__InternetService_No`, plus 3 levels each for `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` (`_No`, `_No internet service`, `_Yes`) |
| **Contract** | 3 | `cat__Contract_Month-to-month`, `cat__Contract_One year`, `cat__Contract_Two year` |
| **Billing & Payment** | 6 | `cat__PaperlessBilling_No`, `cat__PaperlessBilling_Yes`, `cat__PaymentMethod_Bank transfer (automatic)`, `cat__PaymentMethod_Credit card (automatic)`, `cat__PaymentMethod_Electronic check`, `cat__PaymentMethod_Mailed check` |
| **Total Features** | **47** | Complete feature space across all baseline estimators |

- In the experimental pipeline (`create_experimental_preprocessor`), 5 additional features are generated (`num__ChargesRatio`, `num__ChargesRatioDefined`, `num__AutomaticPayment`, `num__TechnicalProtectionCount`, `num__StreamingEntertainmentCount`), yielding **52 features**.

---

## 7. Leakage Prevention Evidence & Empirical Verification

### Invariance Under Corrupted Test Data
A dedicated integration test ([`test_leakage_invariance_extreme_test_values`](file:///d:/Project/Predictive%20Analytics%20Platform/tests/integration/test_preprocessing_pipeline.py#L24)) proves that test-set data cannot contaminate training parameters:
1. The linear pipeline was fitted strictly on $X_{\text{train}}$. Initial scaler parameters were recorded ($\mu_{\text{tenure}} = 32.371, \sigma_{\text{tenure}} = 24.595$).
2. A corrupted holdout test set was injected with extreme unphysical outliers ($\text{tenure} = 99,999$, $\text{MonthlyCharges} = 88,888.0$, $\text{TotalCharges} = 9,999,999.0$).
3. The corrupted test set was transformed through the pipeline.
4. **Result**: The pipeline's stored parameters ($\mu, \sigma$, imputer medians) remained bit-for-bit identical ($0.0$ parameter delta).
5. Transforming $X_{\text{train}}$ before and after the corrupted test transformation yielded identical outputs.

### Summary of Leakage Guardrails
1. `customerID` dropped before transformer fitting.
2. `Churn` target separated before feature transformation.
3. Train/test split occurs before any transformer `fit()`.
4. Imputer statistics computed strictly on training folds.
5. StandardScaler means and variances computed strictly on training folds.
6. OneHotEncoder category vocabularies learned strictly from training folds.
7. Unseen categories at test time handled gracefully via `handle_unknown="ignore"`.
8. Zero oversampling or synthetic sampling performed prior to splitting.

---

## 8. Artifact Serialization

The fitted pipeline is serialized using Joblib:
- File location: `artifacts/preprocessor.joblib`.
- Deserialization verified: A reloaded pipeline applied to $X_{\text{test}}$ yields a bit-identical DataFrame compared to the original in-memory transformation.

---

## 9. Limitations & Phase 4 Recommendations

1. **Cross-Sectional Limitation**: Transformations prepare data for retrospective snapshot classification. They do not convert snapshot observations into longitudinal sequences.
2. **One-Hot Encoding Dimensionality**: 47 features is modest and well within linear model capabilities, but L1/L2 regularization should be applied in Phase 4 to control collinearity between add-on one-hot indicators and `OptionalServiceCount`.
3. **Phase 4 Baseline**: Phase 4 should evaluate dummy baselines, regularized logistic regression, random forests, and gradient boosting using the exact partitions saved in `data/splits/split_manifest.json`.
