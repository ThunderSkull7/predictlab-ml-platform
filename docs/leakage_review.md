# Data Leakage & Feature Review: Telco Churn Dataset

## 1. Executive Summary & Snapshot Context

Data leakage occurs when information outside the training partition or unavailable at prediction time is inadvertently introduced into a machine learning model, producing artificially inflated offline metrics and operational failure.

In the context of the IBM Telco Customer Churn dataset, a fundamental methodological constraint exists:
> **The dataset is a single static, cross-sectional snapshot without timestamps.**

Because observation timestamps ($T_0$) and outcome realization horizons ($\Delta t$) are unrecorded, temporal leakage cannot be audited using chronological train/test splits. Instead, the audit evaluates:
1. **Target Leakage**: Features that reflect consequences of the churn outcome rather than antecedents.
2. **Identifier Memorization**: High-cardinality keys that encode record identity rather than generalizable patterns.
3. **Cross-Sectional Measurement Assumptions**: Assessing whether attributes represent active status prior to or at the point of attrition.

---

## 2. Feature-by-Feature Leakage Audit

### A. Identifier: `customerID`
- **Assessment**: High Risk of memorization / spurious correlation.
- **Audit Findings**: `customerID` contains 7,043 unique alphanumeric strings (100% unique).
- **Decision**: **Mandatory exclusion from model feature matrix**.
- **Justification**: Customer IDs carry no generalizable predictive signal. Including them would allow deep trees or high-capacity models to memorize training samples, causing severe overfitting.

### B. Financial Aggregates: `TotalCharges` & `MonthlyCharges`
- **Assessment**: Low Leakage Risk, Critical Preprocessing Decision.
- **Audit Findings**:
  - `MonthlyCharges` is a continuous float representing the contracted monthly billing rate.
  - `TotalCharges` represents historical cumulative billed revenue up to the observation point.
  - **Whitespace Anomaly**: Exactly 11 rows contain `" "` (whitespace string) in `TotalCharges`. All 11 records have `tenure == 0` and `Churn == "No"`. These represent brand-new subscribers who enrolled but have not yet received their first bill.
- **Leakage Question**: Does `TotalCharges` leak churn?
  - *Analysis*: In this snapshot, `TotalCharges` reflects cumulative historical billings ($\approx \text{tenure} \times \overline{\text{MonthlyCharges}}$). It is not recorded *after* churn; it represents accumulated billing up to account termination or current active state.
- **Decision**: **Pending preprocessing decision**. The proposal to map whitespace to `0.0` (reflecting zero historical charges billed at `tenure == 0`) or evaluate median imputation will be investigated during Phase 2 exploratory data analysis and implemented in Phase 3. The raw CSV is preserved completely unmodified.

### C. Account Duration: `tenure`
- **Assessment**: Valid Cross-Sectional Attribute.
- **Audit Findings**: Discrete integer between 0 and 72 months.
- **Analysis**: Tenure measures account duration up to the observation snapshot. While tenure strongly associates with churn rate (highest attrition observed in early tenure cohorts), it is an active customer state variable, not an outcome consequence. Note that cross-sectional survivorship bias exists: long-tenured accounts in this snapshot have remained active across early tenures without churning. Because the dataset lacks longitudinal event timestamps, it does not support dynamic hazard analysis.

### D. Contractual Terms: `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Assessment**: No Target Leakage (Valid Structural Features).
- **Audit Findings**: Month-to-month subscribers churn at ~42.7%, while two-year contract subscribers churn at ~2.8%.
- **Leakage Analysis**: The contract term is an operational commitment established at enrollment or renewal prior to the churn observation. It represents a valid leading risk factor rather than a consequence of cancellation.

### E. Service Subscription Bundles: `PhoneService`, `InternetService`, Add-ons
- **Features**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Assessment**: No Target Leakage.
- **Analysis**: These represent active service packages in place for the account. There is no evidence in the metadata that add-ons were selectively removed prior to recording churn.

### F. Demographic Features: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Assessment**: No Leakage Risk (Exogenous Customer Attributes).
- **Analysis**: Demographics are recorded at subscription origination and remain invariant to service status. In later phases, fairness audits will evaluate performance parity across these groups.

---

## 3. Post-Event Exit Features (Expanded Dataset Comparison)

Certain expanded variants of the Telco dataset (e.g., IBM Cognos Analytics / Kaggle Telco Churn v2) contain explicit post-cancellation exit survey features:
- `Churn Category` (e.g., "Competitor", "Price", "Dissatisfaction")
- `Churn Reason` (e.g., "Competitor offered higher download speeds")
- `Customer Status` (`"Stayed"`, `"Churned"`, `"Joined"`)

> [!CAUTION]
> Exit survey fields exist **only because churn has already occurred**. Training on post-event fields introduces catastrophic target leakage.
>
> In our verified canonical dataset (`Telco-Customer-Churn.csv`), **none of these post-event fields are present**. The only outcome indicator is the binary target `Churn`.

---

## 4. Retrospective Modeling Assumptions & Guardrails

1. **Retrospective Scope**: The model is evaluated on its ability to classify historical account status and rank relative risk across the snapshot cohort. It cannot be assumed to generalize across future temporal shifts without validation on time-stamped longitudinal data.
2. **Partitioning Guardrails**:
   - Out-of-fold and train-test splits must fit all transformers, encoders, and scalers **strictly on the training partition**.
   - No target statistics or aggregate properties from validation or test partitions may bleed into preprocessing steps.

---

## 5. Phase 2 Exploratory Leakage Verification

The Phase 2 statistical profiling re-audited all 21 columns and confirmed:
1. **Zero Post-Event Exit Artifacts**: No exit survey fields, termination codes, or customer satisfaction surveys conducted post-cancellation are present.
2. **Contract Operational Precedence**: While Month-to-month contracts strongly associate with churn ($\chi^2 = 1184.6, V = 0.408$), contract terms are established at agreement inception, confirming they represent valid pre-event operational risk factors.
3. **TotalCharges Accumulation**: The empirical consistency check (100% match with MonthlyCharges at `tenure == 1`, $\rho = 0.889$ with `tenure`) confirms that `TotalCharges` represents historical billed revenue to date, rather than post-cancellation accumulations.
4. **Conclusion**: Feature sets remain free of target leakage under retrospective snapshot modeling assumptions. Setting `TotalCharges = 0.0` at `tenure == 0` in Phase 3 introduces no outcome information.

---

## 6. Phase 3 Pipeline Leakage Prevention Architecture & Invariance Proof

Phase 3 established formal architectural and programmatic safeguards to guarantee zero leakage:

### A. Preprocessing Boundary Contracts
1. **Identifier Exclusion**: `customerID` is strictly removed from feature matrices $X_{\text{train}}$ and $X_{\text{test}}$ prior to pipeline ingestion. A redundant `ColumnDropper` step in the pipeline provides defense-in-depth against accidental leakage.
2. **Target Isolation**: The target `Churn` is extracted and binary-encoded (`{"No": 0, "Yes": 1}`) before any transformation. The target series $y$ is never passed to feature transformers or column transformers.
3. **No Target-Derived Features**: All engineered features (`OptionalServiceCount`, `AutomaticPayment`, and experimental `ChargesRatio`) are derived purely from exogenous or operational service features, never using target statistics, out-of-fold target encoding, or label-correlated aggregations.
4. **Strict Training-Only Fitting**:
   - `SimpleImputer` medians are calculated exclusively from $X_{\text{train}}$.
   - `StandardScaler` empirical means $\mu$ and standard deviations $\sigma$ are calculated exclusively from $X_{\text{train}}$.
   - `OneHotEncoder` category dictionaries are learned exclusively from $X_{\text{train}}$.
   - Unseen categories in test or operational inference data are handled gracefully via `handle_unknown="ignore"` (assigned all zeros), preventing training refitting or runtime crashes.
5. **No Pre-Split Resampling**: No synthetic oversampling (e.g. SMOTE, ADASYN) or undersampling is applied prior to partitioning.
6. **Deterministic Cleanliness**: Deterministic cleaning operations (converting whitespace to NaN and validating $\text{tenure} == 0$) operate purely record-by-record and learn no global distributional parameters.

### B. Empirical Leakage Invariance Proof
To verify these boundaries programmatically, an automated test ([`test_leakage_invariance_extreme_test_values`](file:///d:/Project/Predictive%20Analytics%20Platform/tests/integration/test_preprocessing_pipeline.py#L24)) validates that:
- When an already-fitted pipeline is presented with a heavily corrupted test set containing extreme unphysical numerical values ($\text{tenure} = 99,999$, $\text{MonthlyCharges} = 88,888.0$, $\text{TotalCharges} = 9,999,999.0$), the stored training parameters (`scaler.mean_`, `scaler.scale_`, `imputer.statistics_`) remain 100% bit-for-bit identical.
- Re-transforming $X_{\text{train}}$ before and after transforming the corrupted test data produces identical output matrices.

> [!IMPORTANT]
> Merely encapsulating operations in a scikit-learn `Pipeline` does not automatically eliminate all methodological leakage. The cross-sectional nature of the IBM dataset inherently carries survivorship selection and lacks longitudinal event sequences. The platform's predictive capability is strictly characterized as retrospective risk ranking, not future forecasting.
