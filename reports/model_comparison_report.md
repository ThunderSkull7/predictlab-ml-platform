# PredictLab — Phase 4: Model Exploration, Baseline Establishment, Comparison & Controlled Tuning Report

## 1. Executive Summary

This report documents the methodology, empirical results, and governance safeguards of **Phase 4: Model Exploration, Baseline Establishment, Comparison & Controlled Tuning** for PredictLab.

Phase 4 establishes an empirical machine-learning benchmarking framework using the verified Phase 1–3.2 preprocessing foundation. All model evaluations adhere strictly to data-isolation contracts:
- The **holdout test partition** ($N = 1,409$) was **strictly protected and untouched** during model comparison, feature ablation, and hyperparameter tuning.
- All candidate models were evaluated exclusively on the training partition ($N = 5,634$) using **stratified 5-fold cross-validation**.
- Preprocessing and estimators were encapsulated inside scikit-learn Pipelines, ensuring transformers were fitted **strictly within each cross-validation fold**.
- A comprehensive **experimental feature ablation study** empirically evaluated whether the Phase 3.1 52-feature experimental pipeline provided predictive lift over the canonical 47-feature baseline.
- The tuned **Histogram Gradient Boosting Classifier** was selected as the champion model family, achieving a cross-validated Average Precision (PR-AUC) of **0.6684** (±0.0218), ROC-AUC of **0.8495** (±0.0123), and Log Loss of **0.4121** (±0.0133).
- A single final evaluation on the protected holdout test set demonstrated strong generalization with PR-AUC of **0.6617**, ROC-AUC of **0.8481**, and Log Loss of **0.4139**.
- **Model-Estimated Class Probabilities**: Output scores produced by `HistGradientBoostingClassifier.predict_proba()` represent model-estimated class probabilities that have not yet undergone empirical calibration assessment. They must not be described as calibrated probabilities. Lower log loss reflects superior proper scoring performance rather than empirical calibration. Formal calibration analysis (calibration curves, reliability diagrams, Platt scaling, isotonic regression) is deferred to Phase 5.
- **Strict Data Partitioning for Importance & Holdout**: Permutation feature importance was evaluated strictly on the training partition ($X_{\text{train}}, y_{\text{train}}$) using `scoring="average_precision"`. The holdout test set ($N=1,409$) was evaluated strictly once for final independent performance estimation.

> [!IMPORTANT]
> **Cross-Sectional Scope Limitation**: The dataset is a cross-sectional snapshot of customer accounts at a single observation point. All model evaluations represent retrospective classification and risk ranking across subscriber accounts, **not temporal forecasting or forward-looking event prediction**.

---

## 2. Experimental Methodology & Leakage Controls

### 2.1 Dataset Partitioning & Provenance
The model training procedure strictly utilized the canonical stratified holdout partition generated in Phase 3 and recorded in [`data/splits/split_manifest.json`](file:///d:/Project/Predictive%20Analytics%20Platform/data/splits/split_manifest.json):
- **Raw Dataset Checksum (SHA-256)**: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Training Partition**: $N = 5,634$ samples (80.0%), $1,495$ churned accounts (26.54% churn rate).
- **Holdout Test Partition**: $N = 1,409$ samples (20.0%), $374$ churned accounts (26.54% churn rate).
- **Random Seed**: `random_state = 42`.
- **Target Variable**: `Churn` (encoded binary: `0` = Retained, `1` = Churned).
- **Excluded Identifier**: `customerID` was strictly dropped prior to feature transformation.

### 2.2 Leakage-Free Cross-Validation Design
Cross-validation was implemented via [`evaluate_pipeline_cv`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/models/cross_validation.py#L38) using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`:
1. **Fold-Level Pipeline Fitting**: For each fold, the entire composite pipeline (cleaning, engineering, column transformation, and classifier) was fitted strictly on $X_{\text{train\_fold}}$ and evaluated on $X_{\text{val\_fold}}$.
2. **Zero Global Preprocessing**: No preprocessing or standard scaling was applied to the training set prior to cross-validation.
3. **No Target Leakage**: No target-derived predictors, out-of-fold target encodings, or pre-split oversampling (e.g. SMOTE) were introduced.

### 2.3 Evaluation Metrics
Because customer churn exhibits class imbalance ($26.54\%$ positive prevalence, negative-to-positive ratio $2.77:1$), models were evaluated across complementary metric dimensions:
- **Average Precision / PR-AUC (Primary Criterion)**: Evaluates the area under the Precision-Recall curve, directly penalizing false positives in the minority churn class across all classification thresholds.
- **ROC-AUC**: Discriminative ranking ability across positive and negative pairs.
- **F1-Score, Precision, Recall, Balanced Accuracy**: Threshold-dependent metrics evaluated at the initial decision threshold ($0.5$).
- **Log Loss & Brier Score**: Strictly proper scoring rules that evaluate the probabilistic accuracy and cross-entropy of predicted probability distributions without independently proving empirical calibration.

---

## 3. Candidate Model Benchmarking (5-Fold Stratified CV)

Ten candidate model configurations across four architectural families were evaluated on the 47-feature baseline.

### 3.1 Benchmark Results Table (Ranked by PR-AUC)

| Model Family | Configuration | PR-AUC (Mean ± Std) | ROC-AUC (Mean ± Std) | F1-Score | Balanced Accuracy | Log Loss | Fit Time (s) |
|---|---|---|---|---|---|---|---|
| **Tree** | Random Forest (Constrained: depth=8, leaf=5) | **0.6642 ± 0.0227** | **0.8484 ± 0.0106** | 0.5743 ± 0.0184 | 0.7063 ± 0.0109 | 0.4147 ± 0.0092 | 0.228 |
| **Linear** | Logistic Regression (L1 Penalty, SAGA) | **0.6619 ± 0.0214** | **0.8461 ± 0.0140** | 0.5929 ± 0.0330 | 0.7197 ± 0.0221 | 0.4167 ± 0.0139 | 0.377 |
| **Linear** | Logistic Regression (L2 Penalty, L-BFGS) | **0.6616 ± 0.0218** | **0.8462 ± 0.0141** | 0.5921 ± 0.0331 | 0.7193 ± 0.0222 | 0.4166 ± 0.0140 | 0.039 |
| **Linear** | Logistic Regression (L2 Balanced Weights) | 0.6599 ± 0.0218 | 0.8460 ± 0.0139 | 0.6284 ± 0.0263 | 0.7655 ± 0.0234 | 0.4908 ± 0.0123 | 0.041 |
| **Tree** | Random Forest (Balanced Subsample) | 0.6598 ± 0.0204 | 0.8471 ± 0.0100 | 0.6324 ± 0.0159 | 0.7650 ± 0.0136 | 0.4689 ± 0.0105 | 0.217 |
| **Boosting** | Hist Gradient Boosting (Balanced Weights) | 0.6526 ± 0.0249 | 0.8417 ± 0.0083 | 0.6292 ± 0.0148 | 0.7610 ± 0.0127 | 0.4706 ± 0.0105 | 0.343 |
| **Boosting** | Hist Gradient Boosting (Default) | 0.6523 ± 0.0243 | 0.8418 ± 0.0090 | 0.5754 ± 0.0339 | 0.7079 ± 0.0207 | 0.4231 ± 0.0116 | 0.633 |
| **Tree** | Random Forest (Unconstrained Baseline) | 0.6079 ± 0.0329 | 0.8189 ± 0.0145 | 0.5528 ± 0.0254 | 0.6940 ± 0.0154 | 0.5050 ± 0.0382 | 0.195 |
| **Baseline** | Dummy Classifier (Prior) | 0.2654 ± 0.0001 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5786 ± 0.0001 | 0.027 |
| **Baseline** | Dummy Classifier (Most Frequent) | 0.2654 ± 0.0001 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 | 9.1650 ± 0.0036 | 0.027 |

### 3.2 Key Benchmark Findings
1. **Unconstrained Trees Memorize**: An unconstrained Random Forest (`max_depth=None`, `min_samples_leaf=1`) suffers severe overfitting, yielding a substantially lower PR-AUC ($0.6079$) and higher log loss ($0.5050$). Constraining tree depth to $8$ and leaf size to $5$ restores performance to $0.6642$ ($+0.0563$ PR-AUC lift).
2. **Linear Regularization Competitiveness**: L1 and L2 regularized logistic regressions achieve exceptional ranking performance ($0.6619$ PR-AUC), confirming that linear combinations of demographic, contract, and service indicators form a strong predictive signal.
3. **Class Weighting Trade-Off**: Introducing `class_weight="balanced"` improves minority recall (raising Balanced Accuracy from $0.719$ to $0.765$ and F1 from $0.592$ to $0.628$), but incurs a penalty in Log Loss ($0.4166 \to 0.4908$) due to distorted posterior probability estimates produced by artificial sample weighting at the default $0.5$ decision threshold.

![Candidate Model Benchmark](figures/17_candidate_models_pr_auc_comparison.png)

---

## 4. Experimental Feature Ablation Study

A core mandate of Phase 4 is to empirically test whether the engineered experimental features provide measurable predictive lift over the default 47-feature baseline.

Using identical 5-fold cross-validation folds, training observations, and seeds, five pipeline configurations were evaluated using `HistGradientBoostingClassifier`:

### 4.1 Ablation Results Table

| Pipeline Configuration | Total Features | PR-AUC (Mean ± Std) | Δ PR-AUC | ROC-AUC | Δ ROC-AUC | Log Loss | Δ Log Loss | Fit Time (s) |
|---|---|---|---|---|---|---|---|---|
| **Baseline (Approved)** | **47** | **0.6523 ± 0.0243** | **0.0000** | **0.8418** | **0.0000** | **0.4231** | **0.0000** | 0.348 |
| **Ablate ChargesRatio** | 50 | 0.6516 ± 0.0255 | -0.0007 | 0.8414 | -0.0005 | 0.4247 | +0.0016 | 0.358 |
| **Ablate Service Subtotals** | 50 | 0.6449 ± 0.0219 | -0.0074 | 0.8403 | -0.0015 | 0.4258 | +0.0027 | 0.382 |
| **Full Experimental Pipeline** | 52 | 0.6444 ± 0.0193 | -0.0079 | 0.8403 | -0.0015 | 0.4263 | +0.0032 | 0.377 |
| **Ablate AutomaticPayment** | 51 | 0.6435 ± 0.0202 | -0.0089 | 0.8403 | -0.0015 | 0.4262 | +0.0031 | 0.398 |

![Feature Ablation Deltas](figures/18_feature_ablation_delta_comparison.png)

### 4.2 Scientific Analysis & Architecture Decision
1. **Degradation from Collinear Redundancy**: The full 52-feature experimental pipeline exhibited a **-0.0079 reduction in PR-AUC** and a **+0.0032 increase in Log Loss** compared to the 47-feature baseline.
2. **Feature Collinearity Mechanism**:
   - `AutomaticPayment` is a deterministic linear combination of two categories in `PaymentMethod` (`Bank transfer (automatic)` and `Credit card (automatic)`), which are already explicitly represented in the one-hot categorical space.
   - `TechnicalProtectionCount` and `StreamingEntertainmentCount` are simple sums of individual service flags that already exist as distinct one-hot features.
   - For tree-based estimators, providing both the raw component indicators and redundant aggregate sums dilutes feature split selection without adding independent information.
3. **Architecture Decision**: The **canonical 47-feature baseline is formally selected** as the production feature space. The experimental additions are retained in the codebase for research comparability but excluded from production candidate pipelines.

---

## 5. Controlled Hyperparameter Tuning

Controlled grid search was executed strictly on the training partition ($N = 5,634$) using stratified 5-fold cross-validation with Average Precision (`scoring="average_precision"`) as the objective function.

### 5.1 Search Space & Optimal Parameters

| Parameter | Explored Values | Optimal Parameter | Rationale & Mechanism |
|---|---|---|---|
| `learning_rate` | `[0.03, 0.05, 0.08]` | **`0.03`** | Slower shrinkage rate prevents early optimization overshooting and improves ensemble smoothing. |
| `max_iter` | `[100, 150, 200]` | **`150`** | Balances boosting capacity with iteration limits. |
| `max_leaf_nodes` | `[15, 31]` | **`15`** | Strongly constrains individual tree depth, preventing memorization of sparse categorical combinations. |
| `min_samples_leaf` | `[20, 40]` | **`40`** | Enforces minimum statistical support at leaves, stabilizing leaf predictions. |
| `l2_regularization` | `[0.1, 1.0, 5.0]` | **`5.0`** | High L2 leaf value regularization penalizes extreme log-odds adjustments in terminal nodes. |

### 5.2 Tuning Results & Fold-Level Breakdown

Controlled hyperparameter tuning produced significant gains across ranking and probabilistic metrics:
- **Default HistGradientBoosting PR-AUC**: $0.6523 \pm 0.0243$
- **Tuned HistGradientBoosting PR-AUC**: **$0.6684 \pm 0.0218$** ($+0.0161$ improvement)
- **Tuned HistGradientBoosting ROC-AUC**: **$0.8495 \pm 0.0123$** ($+0.0077$ improvement)
- **Tuned HistGradientBoosting Log Loss**: **$0.4121 \pm 0.0133$** (reduced from $0.4231$)

The complete fold-level metrics for the tuned model (`Hist Gradient Boosting (Tuned)`) across all 5 cross-validation folds on $X_{\text{train}}$ are detailed below:

| Fold | PR-AUC | ROC-AUC | F1-Score | Precision | Recall | Balanced Acc | Log Loss | Brier Score |
|---|---|---|---|---|---|---|---|---|
| Fold 1 | 0.6501 | 0.8445 | 0.5649 | 0.6578 | 0.4950 | 0.7010 | 0.4177 | 0.1360 |
| Fold 2 | 0.6407 | 0.8332 | 0.5461 | 0.6250 | 0.4849 | 0.6899 | 0.4306 | 0.1420 |
| Fold 3 | 0.6910 | 0.8459 | 0.6140 | 0.6816 | 0.5585 | 0.7322 | 0.4138 | 0.1329 |
| Fold 4 | 0.6774 | 0.8613 | 0.6134 | 0.6706 | 0.5652 | 0.7325 | 0.3989 | 0.1293 |
| Fold 5 | 0.6826 | 0.8624 | 0.5930 | 0.7051 | 0.5117 | 0.7172 | 0.3993 | 0.1283 |
| **Mean ± Std** | **0.6684 ± 0.0218** | **0.8495 ± 0.0123** | **0.5863 ± 0.0301** | **0.6680 ± 0.0297** | **0.5231 ± 0.0368** | **0.7145 ± 0.0189** | **0.4121 ± 0.0133** | **0.1337 ± 0.0055** |

> [!NOTE]
> **Methodological Nuances on Tuned CV Metrics**:
> 1. **Selection Bias Limitation**: Tuned CV metrics may exhibit slight optimistic selection bias because hyperparameters were selected via grid search on the same 5-fold cross-validation procedure. The one-time evaluation on the protected holdout test set provides the unbiased check on out-of-sample generalization.
> 2. **Split Variability vs Confidence Intervals**: The standard deviations reported across cross-validation folds quantify split variability induced by finite-sample partitioning. They must **not** be interpreted as population confidence intervals or formal hypothesis test critical values.

---

## 6. Model Selection Rationale

The **Tuned Histogram Gradient Boosting Classifier (`Hist Gradient Boosting (Tuned)`)** was selected as the champion model for PredictLab based on the following multi-criteria evaluation:

1. **Peak Discrimination (PR-AUC)**: Achieved the highest cross-validated PR-AUC ($0.6684$), outperforming constrained random forests ($0.6642$) and regularized logistic regression ($0.6619$).
2. **Superior Log Loss (Proper Probabilistic Scoring)**: Delivered the lowest cross-entropy loss ($0.4121 \pm 0.0133$) among candidate models, indicating superior probabilistic predictive accuracy under a proper scoring rule. (Note: lower log loss does not independently establish empirical calibration; formal calibration curves and post-hoc calibration methods are deferred to Phase 5).
3. **Fold Stability**: Demonstrated narrow standard deviation across the 5 validation folds ($\sigma = 0.0218$ for PR-AUC; $\sigma = 0.0123$ for ROC-AUC), confirming that performance is robust across sample splits.
4. **Computational Efficiency**: Gradient boosting inference is highly optimized ($< 0.005$ seconds scoring time per batch), providing superior responsiveness for single-customer and cohort scoring in downstream phases.

---

## 7. One-Time Holdout Evaluation (Protected Test Set: N = 1,409)

The selected pipeline was refitted once on the complete training partition ($N = 5,634$) and evaluated on the untouched holdout test partition ($N = 1,409$):

### 7.1 Three-Tier Performance Metrics Reconciliation

To prevent metric substitution and ensure full transparency, the table below reconciles all three evaluation tiers:
1. **Tier 1: Candidate Benchmark (Untuned CV)** — Baseline HistGradientBoosting 5-fold cross-validation.
2. **Tier 2: Tuned Estimator (5-Fold CV)** — Tuned HistGradientBoosting evaluated strictly on $X_{\text{train}}$.
3. **Tier 3: One-Time Protected Holdout Test** — Evaluated strictly once on held-out $X_{\text{test}}$ ($N=1,409$).

| Metric | Tier 1: Candidate Benchmark (Untuned CV Mean ± Std) | Tier 2: Tuned Estimator (5-Fold CV Mean ± Std) | Tier 3: Protected Holdout Test Score ($N=1,409$) | Generalization Delta (Holdout - Tuned CV) | Delta in Fold Std Units (σ) |
|---|---|---|---|---|---|
| **Average Precision / PR-AUC** | 0.6523 ± 0.0243 | **0.6684 ± 0.0218** | **0.6617** | **-0.0067** | -0.31 σ |
| **ROC-AUC** | 0.8418 ± 0.0090 | **0.8495 ± 0.0123** | **0.8481** | **-0.0014** | -0.11 σ |
| **Log Loss** | 0.4231 ± 0.0116 | **0.4121 ± 0.0133** | **0.4139** | **+0.0018** | +0.14 σ |
| **Brier Score** | 0.1373 ± 0.0052 | **0.1337 ± 0.0055** | **0.1351** | **+0.0014** | +0.25 σ |
| **Balanced Accuracy** | 0.7079 ± 0.0207 | **0.7145 ± 0.0189** | **0.7114** | **-0.0031** | -0.16 σ |
| **F1-Score** | 0.5754 ± 0.0339 | **0.5863 ± 0.0301** | **0.5812** | **-0.0051** | -0.17 σ |
| **Precision** | 0.6535 ± 0.0377 | **0.6680 ± 0.0297** | **0.6566** | **-0.0114** | -0.38 σ |
| **Recall** | 0.5144 ± 0.0347 | **0.5231 ± 0.0368** | **0.5214** | **-0.0017** | -0.05 σ |

### 7.2 Confusion Matrix (Decision Threshold = 0.5)

```
                       Predicted Retained    Predicted Churned
Actual Retained (N=1035)       933 (TN)             102 (FP)
Actual Churned  (N=374)        179 (FN)             195 (TP)
```
- **Specificity (True Negative Rate)**: $933 / 1035 = \mathbf{90.14\%}$
- **Sensitivity / Recall (True Positive Rate)**: $195 / 374 = \mathbf{52.14\%}$
- **Precision (Positive Predictive Value)**: $195 / (195 + 102) = \mathbf{65.66\%}$
- **Overall Accuracy**: $(933 + 195) / 1409 = \mathbf{80.06\%}$

![Holdout ROC and PR Curves](figures/19_holdout_roc_pr_curves.png)
![Holdout Confusion Matrix](figures/20_holdout_confusion_matrix.png)

### 7.3 Generalization Integrity Analysis
The holdout test metrics track the tuned cross-validation estimates with remarkable fidelity:
- PR-AUC delta is $-0.0067$, within $0.31$ standard deviations of the cross-validation mean.
- ROC-AUC delta is $-0.0014$, within $0.11$ standard deviations of the cross-validation mean.
- Log Loss delta is $+0.0018$, within $0.14$ standard deviations of the cross-validation mean.
- F1 delta is $-0.0051$ (within $0.17 \sigma$) and Balanced Accuracy delta is $-0.0031$ (within $0.16 \sigma$).
- The similarity between training-only cross-validation results and the protected holdout results is consistent with reasonable generalization on this dataset. It does not independently prove the absence of leakage (which is instead safeguarded and verified by our architectural controls: strict pre-split partitioning, fold-level pipeline encapsulation, and empirical leakage invariance tests). The minor negative shifts are consistent with expected selection shrinkage from tuning rather than test-set overfitting.

---

## 8. Explainability Preparation & Top Predictors

In preparation for Phase 5 (Explainability & Threshold Analysis), feature importance scores were computed using permutation importance:
- **Partition Utilized**: Strictly the **training partition** ($X_{\text{train}}, y_{\text{train}}$, $N = 5,634$).
- **Scoring Objective**: `scoring="average_precision"`.
- **Repeats & Seed**: $N_{\text{repeats}} = 5$, `random_state = 42`.
- **Holdout Set Protection**: The protected holdout test set ($X_{\text{test}}, y_{\text{test}}$, $N = 1,409$) was **never accessed** for permutation importance, feature selection, or feature ranking. It was evaluated strictly once for final independent performance verification.

### 8.1 Top 15 Predictive Features

| Rank | Transformed Feature | Permutation Importance | Business Domain |
|---|---|---|---|
| 1 | `cat__Contract_Month-to-month` | **0.1334** | Contractual commitment |
| 2 | `num__tenure` | **0.0992** | Customer relationship duration |
| 3 | `num__MonthlyCharges` | **0.0508** | Current billing rate |
| 4 | `num__TotalCharges` | **0.0415** | Cumulative billed revenue |
| 5 | `cat__InternetService_Fiber optic` | **0.0363** | Network technology infrastructure |
| 6 | `cat__PaymentMethod_Electronic check` | **0.0284** | Payment friction / manual billing |
| 7 | `cat__OnlineSecurity_No` | **0.0221** | Cybersecurity protection service |
| 8 | `cat__TechSupport_No` | **0.0214** | Technical support assistance |
| 9 | `cat__Contract_Two year` | **0.0195** | Long-term contract lock-in |
| 10 | `num__OptionalServiceCount` | **0.0165** | Value-add service engagement depth |
| 11 | `cat__OnlineBackup_No` | **0.0112** | Cloud storage add-on |
| 12 | `cat__Contract_One year` | **0.0098** | Medium-term contract commitment |
| 13 | `cat__PaperlessBilling_Yes` | **0.0084** | Paperless billing adoption |
| 14 | `cat__MultipleLines_No` | **0.0065** | Core telephone configuration |
| 15 | `cat__DeviceProtection_No` | **0.0058** | Hardware warranty coverage |

![Top 20 Features](figures/21_feature_importances_top20.png)

> [!NOTE]
> **Methodological Definition of Feature Importance**: Permutation importance measures the decrease in a model's predictive score after a feature's values are shuffled. It estimates model reliance on that feature under the selected scoring procedure; it does not establish causality or conditional dependence. Furthermore, when correlated predictors are present (e.g., `tenure`, `MonthlyCharges`, and `TotalCharges`, or component service indicators and `OptionalServiceCount`), correlated predictors can dilute or distort individual feature importance estimates because the information remains partially accessible to the model through collinear features.

---

## 9. Artifact Management & Reproducibility Audit

The complete model artifact bundle was serialized to [`artifacts/model.joblib`](file:///d:/Project/Predictive%20Analytics%20Platform/artifacts/model.joblib).

### 9.1 Serialized Bundle Specification
- **File Size**: $384 \text{ KB}$
- **Pipeline Structure**: `Pipeline(steps=[('preprocessor', TreePreprocessor), ('classifier', HistGradientBoostingClassifier)])`
- **Metadata Encapsulated**:
  - `dataset_sha256`: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
  - `train_test_split_seed`: $42$
  - `split_provenance`: Canonical Phase 3 stratified holdout partition (80/20 train/test split, random_state=42, stratify=Churn) recorded in `data/splits/split_manifest.json`
  - `model_family`: `HistGradientBoostingClassifier`
  - `model_name`: `Hist Gradient Boosting (Tuned)`
  - `preprocessing_pipeline`: `baseline_tree`
  - `feature_count`: $47$
  - `hyperparameters`: `{'l2_regularization': 5.0, 'learning_rate': 0.03, 'max_iter': 150, 'max_leaf_nodes': 15, 'min_samples_leaf': 40}`
  - `cv_strategy`: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  - `cv_metrics_mean` & `cv_metrics_std` across all 8 metrics (tuned estimator CV on training partition; `threshold` excluded)
  - `decision_threshold`: $0.5$ (preserved separately)
  - `holdout_metrics` from the single test evaluation ($N=1,409$)
  - `evaluation_notes`: Training-only CV vs single holdout evaluation; scores are model-estimated class probabilities that have not yet undergone empirical calibration assessment; calibration deferred to Phase 5.
  - `environment`: Python 3.12.10, scikit-learn 1.6.1, pandas 2.3.1, numpy 2.3.2, joblib 1.5.3

---

## 10. Phase 4 Scope Limitations & Phase 5 Readiness

1. **Uncalibrated Model-Estimated Probabilities**:
   - The predicted scores output by `HistGradientBoostingClassifier.predict_proba()` are model-estimated class probabilities that have not yet undergone empirical calibration assessment.
   - They must not be described as calibrated probabilities.
   - Lower cross-entropy log loss ($0.4121 \pm 0.0133$) reflects superior proper probabilistic scoring performance, but does not independently establish empirical calibration.
   - **Phase 5 Mandate**: Implement formal calibration diagnostics (empirical reliability diagrams, calibration curves, Expected Calibration Error, Brier score decomposition into reliability and resolution) and post-hoc calibration methods (Platt scaling vs. Isotonic Regression via `CalibratedClassifierCV`).
2. **Fixed Decision Threshold ($0.5$)**:
   - The default threshold produces $52.14\%$ recall and $65.66\%$ precision. In telecom churn intervention, false negatives (missed churners) are typically significantly more costly than false positives (outreach discounts to retained customers).
   - **Phase 5 Mandate**: Implement cost-sensitive utility curve analysis, threshold tuning, and business risk tiering.
3. **Explainability Interface**:
   - Permutation importances provide global feature rankings.
   - **Phase 5 Mandate**: Implement local single-customer explanations (waterfall charts, feature contributions) to explain individual predictions.

---

*Phase 4 experimentation, benchmarking, ablation, and holdout evaluation are complete. Standing by for Phase 5 instructions.*

