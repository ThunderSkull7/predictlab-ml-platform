# PredictLab — Exploratory Data Analysis & Diagnostic Insights Report

## 1. Executive Summary

This report delivers the comprehensive empirical findings of **Phase 2: Exploratory Data Analysis (EDA) & Diagnostic Insights** for PredictLab.

Phase 2 establishes an evidence-based understanding of the canonical IBM Telco Customer Churn dataset prior to preprocessing and predictive modeling. All findings, tables, and visualizations are produced directly from the dataset using reusable analytical modules in `src/predictlab/profiling`.

### Core Analytical Findings
1. **Target Distribution & Imbalance**: 26.54% of customer accounts discontinued service ($N = 1,869$), while 73.46% remained active ($N = 5,174$). The class balance ratio is 2.77:1.
2. **Dominant Risk Factor**: **Contract structure** exhibits the strongest association with churn across all attributes (bias-corrected Cramér's $V = 0.408, \chi^2 = 1184.6, p < 10^{-200}$). Customers on Month-to-month contracts churn at **42.71%**, compared to **11.27%** for One-year contracts and **2.83%** for Two-year contracts.
3. **Subscriber Lifecycle Hazard**: Churn is concentrated in early tenure. Customers with tenure $\le 12$ months exhibit an observed churn rate of **47.68%**. Median tenure for churned customers is **10.0 months**, compared to **38.0 months** for retained subscribers (Mann-Whitney $U = 2.45 \times 10^6, p < 10^{-100}$).
4. **Service Vulnerabilities**: Fiber Optic subscribers exhibit an elevated churn rate of **41.89%** (driven by higher monthly bills averaging $91.50). In contrast, accounts with proactive support add-ons show dramatically lower attrition: **OnlineSecurity** (14.61% churn with security vs. 41.77% without) and **TechSupport** (15.17% with support vs. 41.65% without).
5. **Payment Friction**: Customers utilizing **Electronic check** churn at **45.29%** (1.71x baseline), whereas customers enrolled in automatic payment mechanisms (Credit card auto: 15.24%; Bank transfer auto: 16.71%) exhibit low churn rates.
6. **`TotalCharges` Anomaly Resolved**: The 11 whitespace values in `TotalCharges` strictly correspond to `tenure == 0` new signups ($Churn = No$). Evidence proves that `TotalCharges` represents accumulated billed monthly charges ($100\%$ match at tenure 1); setting whitespace to `$0.00` is the mathematically and operationally correct Phase 3 treatment.

---

## 2. Dataset Overview

- **Dataset Source**: Canonical IBM Telco Customer Churn sample dataset (`Telco-Customer-Churn.csv`).
- **Dimensions**: 7,043 rows, 21 columns (1 identifier, 19 predictive candidate features, 1 target variable).
- **Identifier**: `customerID` (7,043 distinct keys, 0 duplicates, 100% uniqueness).
- **Data Integrity**: 0 complete duplicate rows; 0 missing values under standard null checks; exactly 11 whitespace-only strings in `TotalCharges`.

---

## 3. Answers to the 10 Core Analytical Questions

### Q1: What proportion of customers churn, and how imbalanced is the target?
- **Numerical Evidence**:
  - Total Accounts: 7,043
  - Retained (`No`): 5,174 accounts (**73.46%**)
  - Churned (`Yes`): 1,869 accounts (**26.54%**)
  - Negative-to-Positive Ratio: **2.77:1**
- **Visualization**: `reports/figures/01_target_distribution.png`
- **Interpretation**: The target displays moderate class imbalance. While standard classifiers might achieve ~73.5% accuracy by trivially predicting majority class retention, such models offer zero business value. Evaluation must prioritize **PR-AUC**, **Brier score**, and cost-sensitive threshold tuning.
- **Caveat**: The snapshot target represents customers who left within the last month of observation; it does not measure multi-month hazard rates over time.

### Q2: How does customer tenure relate to churn?
- **Numerical Evidence**:
  - Churned Customers: Median tenure = **10.0 months** (Mean = 17.98, IQR = 28.0)
  - Retained Customers: Median tenure = **38.0 months** (Mean = 37.57, IQR = 46.0)
  - Mann-Whitney $U = 2,451,572.0, p < 10^{-100}$; Rank-Biserial correlation $r = -0.370$.
  - Tenure Cohort Churn Rates:
    - 0–12 months: **47.68%** churn (1,037 churns / 2,175 customers)
    - 13–24 months: **28.71%** churn (294 churns / 1,024 customers)
    - 25–36 months: **21.90%** churn (180 churns / 832 customers)
    - 37–48 months: **18.72%** churn (143 churns / 764 customers)
    - 49–60 months: **14.34%** churn (120 churns / 837 customers)
    - 61–72 months: **6.61%** churn (93 churns / 1,407 customers)
- **Visualization**: `reports/figures/03_tenure_vs_churn.png`
- **Interpretation**: Attrition follows an exponential decay hazard profile. The first year of subscription represents the critical retention window.
- **Caveat**: High tenure accounts carry survivorship selection bias; long-tenured customers have already survived early onboarding hazards.

### Q3: How do contract types differ in their observed churn rates?
- **Numerical Evidence**:
  - **Month-to-month**: 3,875 customers (55.0% of cohort); 1,655 churned (**42.71%** churn rate; Relative Risk = **1.61x** baseline).
  - **One year**: 1,473 customers (20.9% of cohort); 166 churned (**11.27%** churn rate; Relative Risk = **0.42x**).
  - **Two year**: 1,695 customers (24.1% of cohort); 48 churned (**2.83%** churn rate; Relative Risk = **0.11x**).
  - $\chi^2 = 1184.6, \text{dof} = 2, p < 10^{-200}$; Bias-corrected Cramér's $V = 0.408$.
- **Visualization**: `reports/figures/06_churn_by_contract.png`
- **Interpretation**: Contract length is the most powerful descriptive discriminator in the dataset. Month-to-month accounts provide zero switching friction, enabling rapid departure when dissatisfied.
- **Caveat**: Association is not causation. Forcing customer cohorts onto two-year contracts will not automatically reduce churn to 2.8% without addressing underlying pricing or service satisfaction.

### Q4: What relationship exists between MonthlyCharges and churn?
- **Numerical Evidence**:
  - Churned Customers: Mean = **$74.44/month**, Median = **$79.65/month** (IQR = 34.6)
  - Retained Customers: Mean = **$61.27/month**, Median = **$64.43/month** (IQR = 60.1)
  - Welch's $t = 18.41, p < 10^{-70}$; Cohen's $d = 0.457$; Mann-Whitney $U = 5,958,542.5, p < 10^{-70}$.
- **Visualization**: `reports/figures/04_monthly_charges_vs_churn.png`
- **Interpretation**: Churned customers pay an average of **$13.17 more per month** than retained customers. A distinct attrition peak occurs in the $70–$100 monthly bill tier, primarily composed of Fiber Optic subscribers.
- **Caveat**: Monthly charges correlate with service bundles. The churn driver is not high charges alone, but high charges coupled with lack of support/security.

### Q5: How do InternetService, OnlineSecurity, and TechSupport relate to churn?
- **Numerical Evidence**:
  - **InternetService**:
    - Fiber optic: 3,096 customers; **41.89%** churn (1,297 churns; Relative Risk = 1.58x).
    - DSL: 2,421 customers; **18.96%** churn (459 churns; Relative Risk = 0.71x).
    - No internet: 1,526 customers; **7.40%** churn (113 churns; Relative Risk = 0.28x).
    - Cramér's $V = 0.320, p < 10^{-150}$.
  - **OnlineSecurity**:
    - No: 3,498 customers; **41.77%** churn.
    - Yes: 2,019 customers; **14.61%** churn.
    - No internet: 1,526 customers; **7.40%** churn.
    - Cramér's $V = 0.347, p < 10^{-180}$.
  - **TechSupport**:
    - No: 3,473 customers; **41.65%** churn.
    - Yes: 2,044 customers; **15.17%** churn.
    - No internet: 1,526 customers; **7.40%** churn.
    - Cramér's $V = 0.344, p < 10^{-180}$.
- **Visualizations**: `reports/figures/08_churn_by_internet_service.png`, `09_churn_by_tech_support.png`, `10_churn_by_online_security.png`, `12_multivariate_internet_and_support.png`.
- **Interpretation**: Fiber optic internet suffers from high churn despite being a premium offering. However, when fiber optic customers subscribe to Tech Support, their churn rate drops from **49.36%** down to **22.84%**! Support bundles mitigate customer frustration with complex broadband installations.

### Q6: Do payment methods show meaningful differences in churn rates?
- **Numerical Evidence**:
  - **Electronic check**: 2,365 customers; **45.29%** churn rate (1,071 churns; Relative Risk = **1.71x**).
  - **Mailed check**: 1,612 customers; **19.11%** churn rate (308 churns).
  - **Bank transfer (automatic)**: 1,544 customers; **16.71%** churn rate (258 churns).
  - **Credit card (automatic)**: 1,522 customers; **15.24%** churn rate (232 churns).
  - $\chi^2 = 648.4, \text{dof} = 3, p < 10^{-135}$; Cramér's $V = 0.302$.
- **Visualization**: `reports/figures/07_churn_by_payment_method.png`
- **Interpretation**: Electronic check requires manual monthly action and frequently exposes customers to billing friction and conscious recurring expenditure reviews. Automated payments create passive retention.
- **Caveat**: Payment method choice correlates with age, digital literacy, and contract choice (electronic check accounts are disproportionately Month-to-month).

### Q7: Are there meaningful differences across senior citizens, partners, and dependents?
- **Numerical Evidence**:
  - **SeniorCitizen**: Non-seniors churn at **23.61%** ($N=5,901$), whereas Senior Citizens churn at **41.68%** ($N=1,142$). $\chi^2 = 159.4, p < 10^{-35}$, Cramér's $V = 0.150$.
  - **Partner**: Customers with partners churn at **19.66%** ($N=3,402$) vs. **32.96%** without partners ($N=3,641$). $\chi^2 = 158.7, p < 10^{-35}$, Cramér's $V = 0.150$.
  - **Dependents**: Customers with dependents churn at **15.45%** ($N=2,110$) vs. **31.28%** without dependents ($N=4,933$). $\chi^2 = 189.1, p < 10^{-40}$, Cramér's $V = 0.163$.
- **Visualization**: `reports/figures/11_churn_by_demographics.png`
- **Interpretation**: Family accounts (partners, children/dependents) exhibit high switching inertia and greater household stability, depressing churn. Senior citizens experience higher churn, likely interacting with higher billing on fixed incomes.

### Q8: What is the relationship between TotalCharges and tenure, and what should happen to the 11 whitespace-only values?
- **Numerical Evidence**:
  - Spearman rank correlation between valid `TotalCharges` and `tenure` is **0.889** ($p < 10^{-300}$).
  - For all 613 customers with `tenure == 1`, $\text{TotalCharges} == \text{MonthlyCharges}$ ($\Delta = 0.000$ across all 613 records).
  - Exactly 11 records have whitespace in `TotalCharges`. All 11 have `tenure == 0` (10 on Two-year contracts, 1 on One-year contract; all `Churn == "No"`).
- **Visualization**: `reports/figures/14_total_charges_tenure_investigation.png`
- **Recommendation**: Impute `0.0` in Phase 3. This reflects the true physical and accounting reality: brand-new subscribers at month 0 have accumulated $0.00 in billed charges.

### Q9: Which features show the strongest descriptive associations with churn?
- **Categorical Hierarchy (Bias-Corrected Cramér's V)**:
  1. `Contract`: $V = 0.408$
  2. `OnlineSecurity`: $V = 0.347$
  3. `TechSupport`: $V = 0.344$
  4. `InternetService`: $V = 0.320$
  5. `PaymentMethod`: $V = 0.302$
  6. `OnlineBackup`: $V = 0.228$
  7. `DeviceProtection`: $V = 0.217$
  8. `StreamingMovies`: $V = 0.163$
  9. `StreamingTV`: $V = 0.162$
  10. `PaperlessBilling`: $V = 0.191$
  11. `Dependents`: $V = 0.163$
  12. `Partner`: $V = 0.150$
  13. `SeniorCitizen`: $V = 0.150$
  14. `MultipleLines`: $V = 0.038$ (weak)
  15. `PhoneService`: $V = 0.011$ (not statistically significant after FDR, $p = 0.329$)
  16. `gender`: $V = 0.000$ (not statistically significant after FDR, $p = 0.487$)
- **Numerical Hierarchy (Rank-Biserial Correlation / Cohen's d)**:
  1. `tenure`: $r = -0.370$ (Mann-Whitney $p < 10^{-100}$, Cohen's $d = -0.832$)
  2. `MonthlyCharges`: $r = +0.228$ (Mann-Whitney $p < 10^{-70}$, Cohen's $d = +0.457$)
  3. `TotalCharges`: $r = -0.231$ (Mann-Whitney $p < 10^{-50}$, Cohen's $d = -0.334$)
- **Visualizations**: `reports/figures/16_categorical_associations_cramers_v.png`, `15_numeric_correlation_matrix.png`.

### Q10: What data-quality or modelling concerns should be addressed before preprocessing?
1. **Identifier Leakage**: `customerID` carries zero generalizable signal and must be dropped.
2. **Whitespace Parsing**: The 11 whitespace values in `TotalCharges` must be converted to `0.0` and cast to float.
3. **Collinearity**: `tenure` and `TotalCharges` share a 0.889 Spearman correlation. Regularized linear models will require L1/L2 penalties or feature selection; tree models will natively split appropriately.
4. **Non-linearities**: Bimodal distributions in `tenure` (peaks at 1 and 72) and `MonthlyCharges` (peaks at $20 and $80) indicate that tree-based non-linear estimators are structurally well-suited for this domain.
5. **Ternary Category Encodings**: Distinct categories (`No phone service`, `No internet service`) must be preserved as distinct tokens or mapped meaningfully rather than erroneously merged into `No`.

---

## 4. Formal Inferential Hypothesis Testing

All hypothesis tests were evaluated at $\alpha = 0.05$ with **Benjamini-Hochberg False Discovery Rate (FDR)** adjustments applied across test families to control type I family-wise error rates.

### Categorical Chi-Square Tests of Independence vs. Churn

| Feature | $\chi^2$ Statistic | dof | Raw $p$-Value | FDR Adjusted $p$-Value | Cramér's $V$ | Statistical Significance |
|---|---|---|---|---|---|---|
| `Contract` | 1184.60 | 2 | $< 10^{-200}$ | $< 10^{-200}$ | 0.408 | **Significant** |
| `OnlineSecurity` | 849.99 | 2 | $< 10^{-180}$ | $< 10^{-180}$ | 0.347 | **Significant** |
| `TechSupport` | 828.20 | 2 | $< 10^{-180}$ | $< 10^{-180}$ | 0.344 | **Significant** |
| `InternetService` | 732.31 | 2 | $< 10^{-155}$ | $< 10^{-155}$ | 0.320 | **Significant** |
| `PaymentMethod` | 648.41 | 3 | $< 10^{-135}$ | $< 10^{-135}$ | 0.302 | **Significant** |
| `OnlineBackup` | 601.85 | 2 | $< 10^{-125}$ | $< 10^{-125}$ | 0.228 | **Significant** |
| `DeviceProtection` | 558.42 | 2 | $< 10^{-115}$ | $< 10^{-115}$ | 0.217 | **Significant** |
| `PaperlessBilling` | 258.28 | 1 | $< 10^{-55}$ | $< 10^{-55}$ | 0.191 | **Significant** |
| `Dependents` | 189.13 | 1 | $< 10^{-40}$ | $< 10^{-40}$ | 0.163 | **Significant** |
| `StreamingMovies` | 375.66 | 2 | $< 10^{-80}$ | $< 10^{-80}$ | 0.163 | **Significant** |
| `StreamingTV` | 372.46 | 2 | $< 10^{-80}$ | $< 10^{-80}$ | 0.162 | **Significant** |
| `Partner` | 158.73 | 1 | $< 10^{-35}$ | $< 10^{-35}$ | 0.150 | **Significant** |
| `SeniorCitizen` | 159.43 | 1 | $< 10^{-35}$ | $< 10^{-35}$ | 0.150 | **Significant** |
| `MultipleLines` | 11.27 | 2 | 0.0036 | 0.0041 | 0.038 | **Significant** |
| `PhoneService` | 0.92 | 1 | 0.3387 | 0.3613 | 0.011 | *Not Significant* |
| `gender` | 0.48 | 1 | 0.4866 | 0.4866 | 0.000 | *Not Significant* |

*Key Inference*: 14 of 16 categorical features are statistically significantly associated with churn. `gender` and `PhoneService` exhibit near-zero association and fail to reject the null hypothesis of independence.

### Numerical Tests Conditioned on Churn

| Feature | Retained Median | Churned Median | Retained Mean | Churned Mean | Mann-Whitney $U$ | MW FDR $p$ | Rank-Biserial $r$ | Welch's $t$ | Welch FDR $p$ | Cohen's $d$ |
|---|---|---|---|---|---|---|---|---|---|---|
| `tenure` | 38.00 | 10.00 | 37.57 | 17.98 | $2.45 \times 10^6$ | $< 10^{-100}$ | **-0.370** | -34.82 | $< 10^{-100}$ | **-0.832** |
| `MonthlyCharges` | 64.43 | 79.65 | 61.27 | 74.44 | $5.96 \times 10^6$ | $< 10^{-70}$ | **+0.228** | +18.41 | $< 10^{-70}$ | **+0.457** |
| `TotalCharges` | 1683.60 | 701.50 | 2555.34 | 1531.80 | $3.73 \times 10^6$ | $< 10^{-50}$ | **-0.231** | -18.84 | $< 10^{-50}$ | **-0.334** |

---

## 5. High-Risk Customer Segments

Using multivariate cross-tabulations, we identified segments exhibiting extreme churn propensities:

| Segment Definition | Cohort Size | Churned Count | Churn Rate (%) | Relative Risk |
|---|---|---|---|---|
| **Month-to-month + Fiber Optic + No TechSupport** | 1,289 | 697 | **54.07%** | **2.04x** |
| **Month-to-month + Electronic Check** | 1,850 | 973 | **52.59%** | **1.98x** |
| **Tenure $\le 12\text{m}$ + Month-to-month** | 1,663 | 888 | **53.40%** | **2.01x** |
| **Senior Citizen + Month-to-month + Electronic Check** | 449 | 260 | **57.91%** | **2.18x** |
| **Two-year + Automatic Payment + Dependents** | 561 | 6 | **1.07%** | **0.04x** |

---

## 6. Recommendations for Phase 3 Preprocessing

1. **Missing Data Handling**: Transform the 11 whitespace strings in `TotalCharges` into `0.0` and cast to `float64`.
2. **Feature Transformation**:
   - `customerID`: Drop from modeling matrices.
   - Categorical Features: Apply One-Hot Encoding for nominal variables (`PaymentMethod`, `InternetService`).
   - Contract: Evaluate Ordinal Encoding (`Month-to-month`: 0, `One year`: 1, `Two year`: 2) as well as One-Hot.
   - Numerical Scaling: RobustScaler or StandardScaler for distance-based estimators.
3. **Feature Engineering Opportunities**:
   - `ChargesRatio`: $\text{TotalCharges} / (\text{tenure} \times \text{MonthlyCharges} + 1e-4)$ to detect billing rate changes.
   - `SupportBundleCount`: Integer count of active security/support add-ons ($0$ to $6$).
   - `AutoPaymentFlag`: Binary indicator (1 if Bank transfer auto or Credit card auto, 0 otherwise).
4. **Validation Guardrail**: Fit all transformers strictly on training partitions to prevent data leakage.
