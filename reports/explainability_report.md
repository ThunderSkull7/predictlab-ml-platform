# PredictLab — Phase 5: Global & Local Explainability Report

## 1. Executive Summary

This report presents the model explainability framework, global feature importance analysis, and local Shapley attribution architecture implemented in Phase 5 of PredictLab.

Predictive machine learning models in customer retention must not operate as impenetrable black boxes. Account managers, customer service teams, and business executives require transparent, auditable explanations to understand why a customer is classified as at-risk, what specific account attributes contribute to that risk, and how retention interventions should be structured.

Phase 5 implements a scientifically rigorous, two-tier explainability architecture:
1. **Cross-Validated Permutation Feature Importance**: Evaluated across a 5-fold cross-validation scheme strictly on held-out validation splits using Average Precision (PR-AUC) drop with 5 deterministic shuffle repeats per fold, preventing in-sample memorization bias.
2. **Local Multi-Estimator TreeSHAP Attribution**: Exact TreeSHAP feature attributions computed by aggregating across all 5 ensemble tree estimators in the calibrated model via [`TreeShapExplainer`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/explainability/local_attribution.py#L40), verifying exact mathematical additivity on the ensemble decision margin ($\bar{f}(x) = \bar{E}[f(x)] + \sum \bar{\phi}_j(x)$).
3. **Primary Predictive Signals**: Across all 47 features, the most powerful global predictors of customer churn status are **Contract type** (month-to-month contracts strongly elevate risk), **tenure** (shorter tenure correlates with high churn prevalence), **InternetService** (Fiber optic), **TotalCharges**, **MonthlyCharges**, and absence of support/security add-ons (**OnlineSecurity: No**, **TechSupport: No**).
4. **Collinear Dilution Accounting**: Mutually correlated features (such as `tenure` and `TotalCharges`, or `MonthlyCharges` and internet streaming services) dilute individual importance scores. Permutation importance reflects the model's reliance on each feature within its specific representation, not total independent bivariate association.
5. **Strict Non-Causal Warning**: All SHAP values and importance rankings quantify statistical associations within a retrospective cross-sectional dataset. They **must not be interpreted as causal levers**. Forcing a customer onto a two-year contract or artificially cutting their monthly charges will not mechanically reduce real-world churn propensity as predicted by the model.

> [!IMPORTANT]
> **Cross-Sectional Scope Limitation**: The dataset is a cross-sectional snapshot of customer accounts at a single observation point. All model evaluations represent retrospective classification and risk ranking across subscriber accounts, **not temporal forecasting or forward-looking event prediction**.

---

## 2. Global Permutation Feature Importance

### 2.1 Cross-Validated Permutation Importance Methodology
Permutation feature importance measures the degradation in model evaluation metric when a specific feature's values are randomly permuted, severing its relationship with the target:
- **Evaluation Partition**: Strictly evaluated across 5 stratified folds on held-out validation splits ($N \approx 1,127$ per fold) rather than in-sample on the training set. The test holdout partition ($N = 1,409$) was **strictly isolated**.
- **Scoring Function**: `scoring="average_precision"` (Average Precision / PR-AUC drop). Because churn is an imbalanced classification problem ($26.54\%$ positive prevalence), evaluating PR-AUC drop directly penalizes loss of discriminative precision-recall performance on the minority churn class.
- **Repeat Count**: $N_{\text{repeats}} = 5$ per fold across 5 folds (25 total evaluations per feature), with fixed `random_state = 42`.
- **Non-Directional Nature**: Permutation importance quantifies feature reliance, not direction of effect. Directional associations must be derived separately from bivariate profiling and local SHAP signs.

### 2.2 Top 20 Global Features Table

The table below details the top 20 global features ranked by cross-validated PR-AUC drop:

| Rank | Feature | Column Type | Validation PR-AUC Drop (Mean ± Std) | Normalized Importance | Observed Directional Pattern (from EDA/SHAP) |
|---|---|---|---|---|---|
| **1** | `cat__Contract_Month-to-month` | Categorical | **0.1076 ± 0.0287** | **31.45%** | Strongest churn-elevating factor; highest hazard cohort |
| **2** | `num__tenure` | Numeric | **0.0805 ± 0.0247** | **23.55%** | Low tenure strongly correlates with observed churn |
| **3** | `cat__InternetService_Fiber optic` | Categorical | **0.0313 ± 0.0156** | **9.16%** | High churn rate cohort ($41.9\%$ observed churn) |
| **4** | `num__TotalCharges` | Numeric | **0.0278 ± 0.0099** | **8.12%** | Correlated with tenure; captures cumulative billing volume |
| **5** | `num__MonthlyCharges` | Numeric | **0.0261 ± 0.0136** | **7.62%** | Higher monthly expenditure elevates attrition risk |
| **6** | `cat__OnlineSecurity_No` | Categorical | **0.0157 ± 0.0129** | **4.60%** | Absence of security add-on associated with higher churn |
| **7** | `cat__MultipleLines_No` | Categorical | **0.0117 ± 0.0074** | **3.43%** | Single-line subscribers exhibit moderate churn elevation |
| **8** | `cat__TechSupport_No` | Categorical | **0.0110 ± 0.0101** | **3.22%** | Absence of technical assistance associated with attrition |
| **9** | `cat__Contract_Two year` | Categorical | **0.0100 ± 0.0082** | **2.93%** | Long-term contract strongly protects retention ($2.8\%$ churn) |
| **10** | `cat__PaperlessBilling_No` | Categorical | **0.0079 ± 0.0061** | **2.32%** | Standard mailed paper bills associate with higher stability |
| **11** | `cat__PaymentMethod_Electronic check` | Categorical | 0.0047 ± 0.0056 | 1.37% | Electronic check payment associated with $45.3\%$ churn |
| **12** | `cat__SeniorCitizen_0` | Binary | 0.0018 ± 0.0022 | 0.53% | Non-senior status associated with lower churn |
| **13** | `cat__Dependents_No` | Binary | 0.0011 ± 0.0027 | 0.31% | Absence of dependents slightly elevates churn risk |
| **14** | `cat__PaperlessBilling_Yes` | Categorical | 0.0010 ± 0.0022 | 0.30% | Paperless billing adoption correlates with churn elevation |
| **15** | `cat__InternetService_DSL` | Categorical | 0.0009 ± 0.0021 | 0.26% | Moderate retention stabilization relative to Fiber |
| **16** | `cat__StreamingMovies_Yes` | Categorical | 0.0006 ± 0.0015 | 0.17% | Streaming entertainment add-on |
| **17** | `num__OptionalServiceCount` | Numeric | 0.0006 ± 0.0010 | 0.17% | Engineered feature: cumulative service adoption count |
| **18** | `cat__MultipleLines_Yes` | Categorical | 0.0005 ± 0.0015 | 0.13% | Multiple phone lines indicator |
| **19** | `cat__OnlineBackup_No` | Categorical | 0.0002 ± 0.0028 | 0.05% | Absence of data backup |
| **20** | `cat__Contract_One year` | Categorical | 0.0002 ± 0.0009 | 0.05% | Moderate retention stabilization |

![Figure 27: Cross-Validated Permutation Importance](figures/27_shap_global_summary.png)

### 2.3 Collinear Feature Dilution Analysis

A critical insight from cross-validated permutation importance is **collinear dilution**:
- **Tenure vs TotalCharges**: `tenure` and `TotalCharges` exhibit strong Spearman correlation ($\rho = 0.826$). When `TotalCharges` is permuted, the gradient boosting trees utilize `tenure` and `MonthlyCharges` to recover predictive signal. Both features appear in the top 5, dividing the overall predictive weight of customer tenure.
- **Service Add-Ons vs Internet Service**: `TechSupport`, `OnlineSecurity`, `OnlineBackup`, and `DeviceProtection` all depend on having internet service. Their importance values are distributed across the individual service categories.
- **Contract Type**: `Contract_Month-to-month` is by far the single most dominant predictor (mean PR-AUC drop of $0.1076$, accounting for $31.45\%$ of normalized importance), reinforcing findings from Phase 2 bivariate profiling.

---

## 3. Local TreeSHAP Feature Attribution

### 3.1 TreeSHAP Mathematical Framework for the Calibrated Ensemble

While global importance explains general model reliance, operational retention teams require explanations for specific individual customer predictions.

PredictLab integrates Lundberg et al.'s (2020) **TreeSHAP algorithm** via [`TreeShapExplainer`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/explainability/local_attribution.py#L40):
- **Aggregation Across the 5-Fold Ensemble**: Because the production calibrated model (`CalibratedClassifierCV`) consists of 5 base pipelines, `TreeShapExplainer` computes exact TreeSHAP values for each individual base tree estimator and takes their arithmetic mean:
  $$\bar{f}(x) = \frac{1}{K} \sum_{k=1}^K f_k(x) = \bar{E}[f(x)] + \sum_{j=1}^M \bar{\phi}_j(x)$$
  where:
  - $\bar{E}[f(x)] = \frac{1}{K} \sum_{k=1}^K E[f_k(x)] \approx -1.5438$ is the mean baseline margin across the 5 estimators.
  - $\bar{\phi}_j(x) = \frac{1}{K} \sum_{k=1}^K \phi_{k, j}(x)$ is the ensemble-averaged Shapley attribution for feature $j$.
  - $\bar{f}(x)$ is the ensemble mean decision margin (uncalibrated log-odds scale).
- **Exact Additivity Verification**: In unit testing, `TreeShapExplainer` confirmed exact mathematical additivity to within machine precision ($|\bar{f}(x) - (\bar{E}[f(x)] + \sum \bar{\phi}_j(x))| \le 2.2 \times 10^{-16}$).
- **Ensemble Decision Margin vs Calibrated Probability**:
  - TreeSHAP explanations operate on the **ensemble uncalibrated decision margin** ($\bar{f}(x)$), which directly reflects the tree splits.
  - The final customer churn probability $\hat{P}(y=1 \mid x) = 0.6868$ is produced by passing each base margin through its corresponding Platt Sigmoid calibrator and averaging the calibrated probabilities.

### 3.2 Exemplar Customer Case Studies

To demonstrate operational explainability, we examine two representative customer profiles from [`reports/experiments/local_explanations_examples.json`](file:///d:/Project/Predictive%20Analytics%20Platform/reports/experiments/local_explanations_examples.json):

#### Case Study A: High-Risk Attrition Exemplar
- **Customer Profile**: Female, tenure = 5 months, MonthlyCharges = $80.00, TotalCharges = 412.5, Fiber Optic internet, Month-to-month contract, Paperless billing, Electronic check, No TechSupport, No OnlineSecurity.
- **Ensemble Mean Base Margin ($\bar{E}[f(x)]$ )**: $-1.5438$
- **Customer Output Margin ($\bar{f}(x)$ )**: **$+0.7876$**
- **Model-Estimated Calibrated Churn Probability**: **$68.68\%$**
- **Assigned Risk Tier**: **High Risk** (operating at $\tau^* = 0.19$, predicted label: **Churned**)

```
Ensemble Base Margin: -1.5438
  + Contract [Month-to-month]      : +0.9279  ▲ Major Churn Driver
  + tenure [5 months]              : +0.4033  ▲ Major Churn Driver
  + OnlineSecurity [No]            : +0.2626  ▲ Churn Driver
  + InternetService [Fiber optic]  : +0.2027  ▲ Churn Driver
  + Contract [Two year = 0]        : +0.1865  ▲ Churn Driver
  - MultipleLines [No]             : -0.1595  ▼ Retention Driver
  - Contract [One year = 0]        : -0.0274  ▼ Minor Retention Driver
  - StreamingTV [No]               : -0.0246  ▼ Minor Retention Driver
----------------------------------------------------------------------
Ensemble Output Margin: +0.7876  ==>  Calibrated Churn Probability: 68.68%
```

![Figure 28: Local SHAP Waterfall Exemplar](figures/28_shap_local_waterfall.png)

#### Case Study B: Low-Risk Retained Exemplar
- **Customer Profile**: Male, tenure = 60 months, MonthlyCharges = $65.00, TotalCharges = 3900.0, DSL internet, Two-year contract, Credit card (automatic), TechSupport = Yes, OnlineSecurity = Yes.
- **Ensemble Mean Base Margin ($\bar{E}[f(x)]$ )**: $-1.5438$
- **Customer Output Margin ($\bar{f}(x)$ )**: **$-3.5222$**
- **Model-Estimated Calibrated Churn Probability**: **$3.84\%$**
- **Assigned Risk Tier**: **Low Risk** (operating at $\tau^* = 0.19$, predicted label: **Retained**)

```
Ensemble Base Margin: -1.5438
  - Contract [Two year]            : -0.7712  ▼ Major Retention Driver
  - tenure [60 months]             : -0.6214  ▼ Major Retention Driver
  - Contract [Month-to-month = 0]  : -0.3854  ▼ Major Retention Driver
  - OnlineSecurity [Yes]           : -0.2140  ▼ Retention Driver
  - TechSupport [Yes]              : -0.1872  ▼ Retention Driver
  + MonthlyCharges [$65.00]        : +0.0815  ▲ Minor Churn Driver
----------------------------------------------------------------------
Ensemble Output Margin: -3.5222  ==>  Calibrated Churn Probability: 3.84%
```

---

## 4. Governance & Methodological Guidance

### 4.1 Strict Non-Causal Interpretation Warning

> [!WARNING]
> **Correlation vs Causation**: SHAP values explain **how the statistical model reached its prediction given the observational snapshot**. They do **NOT** quantify what would happen if a business action actively manipulated an account feature.
>
> **Practical Examples**:
> 1. A customer with `Contract_Month-to-month` receives a high positive SHAP attribution ($+0.93$). This indicates that subscribers with month-to-month contracts historically exhibited high churn prevalence. It does **not** imply that forcing this customer into a 2-year lock-in contract will eliminate their dissatisfaction or propensity to cancel.
> 2. A customer with `OnlineSecurity_No` receives a positive SHAP attribution ($+0.26$). This reflects an empirical association between lack of security add-ons and attrition. Free bundling of security software will not mechanically alter customer retention unless the customer actually values and utilizes the feature.
> 3. Causal claims require randomized controlled trials (A/B testing) or quasi-experimental causal inference methods (e.g. difference-in-differences, instrumental variables, propensity score matching). The current model is strictly predictive and risk-ranking.

### 4.2 Guidance for Retention Account Managers
1. Use SHAP waterfalls as **diagnostic conversation guides**, not definitive causal prescriptions.
2. If `tenure` is short and `Contract` is month-to-month, recognize that the subscriber has not yet established deep product stickiness; focus on early-lifecycle onboarding, product education, and proactive check-ins.
3. If `MonthlyCharges` is a primary churn driver and the customer lacks value-added features (`TechSupport`, `OnlineSecurity`), explore plan rightsizing to align billing with perceived utility.
