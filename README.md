# PredictLab — Portfolio-Grade Predictive Analytics Platform

PredictLab is an end-to-end predictive analytics and machine learning platform engineered to demonstrate production-level ML engineering standards, probabilistic modeling, calibration analysis, and explainable decision support.

The initial machine learning task is formulated as:

> **Retrospective customer churn classification and risk ranking using the available customer snapshot.**

---

## 1. Problem Formulation & Scope

The initial dataset is the canonical IBM Telco Customer Churn sample dataset. Because this dataset is a single static cross-sectional table lacking observation timestamps, event logs, or longitudinal customer histories, the system models statistical patterns distinguishing accounts that churned from accounts that remained active within the snapshot.

PredictLab distinguishes four distinct levels of model outputs:
1. **Model-Estimated Class Probability (Uncalibrated)** ($\hat{s}_i \in [0, 1]$): Model-estimated class probabilities that have not yet undergone empirical calibration assessment.
2. **Empirically Calibrated Probability** ($\hat{p}_i = P(Y=1 \mid \hat{s}_i)$): Evaluated in Phase 5 via Brier score, Expected Calibration Error (ECE), and reliability curves.
3. **Binary Classification Decision** ($\hat{Y}_i \in \{0, 1\}$): Thresholding decision using an explicit decision cutoff $\tau$ (deferred to Phase 5).
4. **Business Risk Tiers**: Categorical prioritization tiers (e.g., Low, Medium, High). Boundaries are deferred to Phase 5 based on empirical score distributions and business utility policies.

---

## 2. Technology Stack (Approved & Restored)

| Layer | Component | Version / Specification |
|---|---|---|
| **Runtime** | Python | 3.12.10 |
| **Data Processing** | Pandas, NumPy | 2.3.1 / 2.3.2 |
| **Machine Learning** | Scikit-learn | 1.6.1 (exact pinned stack) |
| **Explainability** | SHAP | 0.54.1 (TreeSHAP exact additive attributions) |
| **Model Serialization** | Joblib | 1.5.3 |
| **Validation & Settings** | Pydantic, Pydantic-Settings | 2.13.4 / 2.15.0 |
| **Visualization** | Altair | 5.5.0 (exact pinned stack) |
| **Static Plotting** | Matplotlib, Seaborn | 3.10.5 / 0.13.2 |
| **User Interface (Future)** | Streamlit | 1.67.0 |
| **Code Quality & Linting** | Ruff | 0.17.4 |
| **Testing** | Pytest | 9.1.1 |
| **Version Control & CI** | Git, GitHub Actions | Matrix Python 3.12 CI |

---

## 3. Dataset Provenance & Licensing Audit

- **Host Repository**: [`IBM/telco-customer-churn-on-icp4d`](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Direct Source URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Retrieval Date**: `2026-11-07`
- **License Status**: The host repository provides an Apache License 2.0 covering code patterns. However, no dedicated open-data license (e.g., CDLA, CC0, CC-BY) is provided for the CSV file, and IBM hosts the file as sample demonstration data for a tutorial.
- **Redistribution Policy**: Because third-party redistribution rights cannot be conclusively established from upstream metadata alone, PredictLab distinguishes its own MIT software license from third-party sample data terms, documents this limitation, and provides an automated download and SHA-256 verification procedure in `data/README.md`.
- **SHA-256 Checksum**: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` (7,043 rows, 21 columns).
- **`TotalCharges` Preprocessing Directive**: Exactly 11 rows contain `" "` in `TotalCharges`, coinciding strictly with `tenure == 0` and `Churn == "No"`. The raw CSV is preserved completely unmodified. In Phase 3, the preprocessing pipeline will convert whitespace to `NaN`, validate that missingness strictly coincides with `tenure == 0`, and impute `0.0`.

---

## 4. Project Structure

```text
predictlab-ml-platform/
│
├── .github/
│   └── workflows/
│       └── ci.yml                     # Continuous integration workflow
│
├── .streamlit/
│   └── config.toml                    # Streamlit server & theme settings
│
├── data/
│   ├── README.md                      # Data provenance & checksum registry
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv   # Immutable raw canonical dataset
│   └── processed/                     # Engineered feature sets (git-ignored)
│
├── docs/
│   ├── problem_definition.md          # Business problem & retrospective classification formulation
│   ├── data_dictionary.md             # Schema, feature types & value ranges
│   ├── architecture.md                # Component responsibilities & data flow
│   ├── leakage_review.md              # Target & temporal leakage audit
│   └── decisions/
│       └── 0001-initial-tech-stack-and-problem-definition.md
│
├── notebooks/                         # Executed analytical notebooks
│   ├── 01_data_audit.ipynb            # Data quality audit & TotalCharges investigation
│   ├── 02_eda.ipynb                   # Comprehensive exploratory analysis & Q1-Q10 findings
│   ├── 03_preprocessing_and_feature_engineering.ipynb # Preprocessing pipeline, splitting & leakage audit
│   ├── 04_model_training_and_comparison.ipynb         # Model exploration, baselines & tuning
│   └── 05_evaluation_explainability_and_inference.ipynb # Probability calibration, thresholding, SHAP & inference engine
│
├── src/
│   └── predictlab/                    # Headless core Python package
│       ├── __init__.py
│       ├── config.py                  # Pydantic Settings & dynamic path resolution
│       ├── logging_config.py          # Structured logging configuration
│       ├── data/
│       │   ├── __init__.py
│       │   ├── load.py                # CSV ingestion with error boundaries & duplicate header detection
│       │   └── validate.py            # Active Pydantic schema & consistency validation
│       ├── profiling/                 # Profiling & diagnostic insights package (Phase 2)
│       │   ├── __init__.py
│       │   ├── quality.py             # Data quality audit & anomaly investigation
│       │   ├── summary.py             # Univariate numeric & categorical summaries
│       │   ├── insights.py            # Segment churn rates, Chi-square & Mann-Whitney tests
│       │   └── visuals.py             # Publication-grade visual figure generation
│       ├── preprocessing/             # Preprocessing & transformation package (Phase 3)
│       │   ├── __init__.py
│       │   ├── schema.py              # Feature schemas, contracts & allowed category domains
│       │   ├── cleaning.py            # TotalCharges deterministic cleaning & zero-tenure validation
│       │   ├── feature_engineering.py # OptionalServiceCount, AutomaticPayment & ChargesRatio
│       │   ├── transformers.py        # Scikit-learn custom transformers with get_feature_names_out
│       │   ├── split.py               # Stratified holdout partition & manifest persistence
│       │   └── pipelines.py           # Linear and tree ColumnTransformer factory functions
│       └── models/                    # Modeling, evaluation, tuning & selection package (Phase 4)
│           ├── __init__.py
│           ├── baselines.py           # DummyClassifier benchmarks (prior, most_frequent)
│           ├── estimators.py          # Candidate model factories (LR, RF, HistGradientBoosting)
│           ├── evaluation.py          # Multi-metric evaluation (PR-AUC, ROC-AUC, F1, Log Loss)
│           ├── cross_validation.py    # Leakage-free Stratified 5-Fold cross-validation
│           ├── experiments.py         # Model benchmark runner & feature ablation study
│           ├── tuning.py              # Controlled hyperparameter tuning on training partition
│           ├── selection.py           # Objective model selection & holdout evaluation
│           └── persistence.py         # Artifact bundle & experiment JSON serialization
│       ├── calibration/               # Probability calibration & scoring rules (Phase 5)
│       │   ├── __init__.py
│       │   ├── metrics.py             # Brier score, Log Loss, ECE, Murphy decomposition
│       │   ├── curves.py              # Reliability curves & probability distribution histograms
│       │   ├── cross_validation.py    # Leakage-free 5x5 nested CV across calibration methods
│       │   └── selection.py           # Multi-criteria calibration selection & ensemble refit
│       ├── thresholding/              # Decision threshold analysis & risk tiers (Phase 5)
│       │   ├── __init__.py
│       │   ├── metrics.py             # 81-step threshold sweep across classification metrics
│       │   ├── costs.py               # Configurable CostMatrix & cost ratio sensitivity
│       │   ├── optimization.py        # Threshold optimization & nested CV policy optimism audit
│       │   └── risk_tiers.py          # Contiguous 3-tier risk partitioner & cohort profiler
│       ├── explainability/            # Global & local model interpretation (Phase 5)
│       │   ├── __init__.py
│       │   ├── global_importance.py   # Training-only permutation importance with Average Precision
│       │   └── local_attribution.py   # TreeSHAP local attribution on model margin log-odds scale
│       └── inference/                 # Production-grade single & batch inference (Phase 5)
│           ├── __init__.py
│           ├── schemas.py             # Pydantic v2 input/output schemas & predictor validation
│           └── engine.py              # InferenceEngine with atomic & per-record batch modes
│
├── app/                               # Streamlit UI application (Phase 6)
├── models/                            # Serialized model binaries (Phase 4-5)
├── artifacts/                         # Preprocessors, model bundles, metrics (Phase 3-5)
├── reports/                           # Generated evaluation reports & figures
│   ├── data_quality_report.md         # Comprehensive data quality audit report
│   ├── eda_report.md                  # Detailed analytical findings & statistical test report
│   ├── preprocessing_report.md        # Preprocessing, feature engineering & transformation report
│   ├── model_comparison_report.md     # Model benchmarking, ablation & holdout evaluation report
│   ├── calibration_report.md          # Probability calibration & reliability scoring report
│   ├── threshold_analysis_report.md   # Decision threshold & cost sensitivity report
│   ├── explainability_report.md       # Global & local TreeSHAP attribution report
│   ├── inference_engine_report.md     # Production inference engine & schema validation report
│   ├── experiments/                   # Experiment JSON/CSV metrics & tuning summaries
│   └── figures/                       # 28 publication-grade visualizations (Figures 01-28)
│
├── tests/
│   ├── fixtures/
│   │   └── synthetic_data.py          # Deterministic test data generator
│   ├── unit/
│   │   ├── test_config.py             # Config & path resolution tests
│   │   ├── test_load.py               # Ingestion & error-handling tests
│   │   ├── test_validate.py           # Schema & Pydantic consistency validation tests
│   │   ├── test_profiling.py          # Statistical profiling & data quality tests
│   │   ├── test_preprocessing_cleaning.py  # TotalCharges validation & imputation tests
│   │   ├── test_preprocessing_features.py  # OptionalServiceCount & feature engineering tests
│   │   ├── test_preprocessing_split.py     # Stratified partitioning & manifest tests
│   │   ├── test_preprocessing_pipelines.py # ColumnTransformer & serialization tests
│   │   ├── test_models_baselines.py        # Dummy baseline tests
│   │   ├── test_models_estimators.py       # Estimator pipeline construction tests
│   │   ├── test_models_evaluation.py       # Metric & curve calculation tests
│   │   ├── test_models_cross_validation.py # Stratified CV & fold isolation tests
│   │   ├── test_calibration.py        # ECE, Brier decomposition & reliability tests
│   │   ├── test_thresholding.py       # Threshold sweep & cost matrix tests
│   │   ├── test_risk_tiering.py       # Risk tier partitioner & cohort profile tests
│   │   ├── test_explainability.py     # Permutation importance & TreeSHAP tests
│   │   └── test_inference.py          # InferenceEngine, schema validation & batch tests
│   │   ├── test_models_experiments.py      # Feature ablation tests
│   │   └── test_models_selection_and_persistence.py # Selection, tuning & persistence tests
│   └── integration/
│       ├── test_data_pipeline.py      # End-to-end ingestion/validation pipeline
│       ├── test_eda_pipeline.py       # End-to-end profiling pipeline on canonical dataset
│       └── test_preprocessing_pipeline.py # Leakage invariance & full pipeline integration
│
├── .env.example                       # Environment variable template
├── .gitignore                         # Comprehensive ignore rules
├── LICENSE                            # MIT Code License & Third-Party Sample Data Notice
├── README.md                          # Platform documentation
└── pyproject.toml                     # Project metadata, dependencies & tool configs
```

---

## 5. Implementation Roadmap

PredictLab is developed following a rigorous 7-phase engineering plan:

- [x] **Phase 1: Foundation, Architecture & Problem Definition** (Including Phase 1.1 Methodological Corrections)
  - Environment setup, project structure, dynamic configuration, structured logging.
  - Upstream dataset provenance and licensing audit; SHA-256 verification.
  - Retrospective classification problem formulation, data dictionary, and data leakage audit.
  - Active Pydantic validation and CSV ingestion with 100% test pass rate.
  - Restoration and locking of approved dependency versions (`scikit-learn 1.6.1`, `altair 5.5.0`, `streamlit 1.67.0`).
- [x] **Phase 2: Exploratory Data Analysis & Diagnostic Insights**
  - Data quality audit, whitespace detection, and `TotalCharges` anomaly investigation.
  - Univariate, bivariate, and multivariate profiling across all 20 features.
  - Inferential hypothesis testing: Chi-square with Benjamini-Hochberg FDR and Mann-Whitney U tests.
  - Answers to the 10 core analytical questions documented with empirical evidence.
  - 16 publication-grade figures generated in `reports/figures/`.
  - Executed notebooks `01_data_audit.ipynb` and `02_eda.ipynb`.
  - 34 automated unit and integration tests passing.
- [x] **Phase 2.1: Methodological Corrections & EDA Sign-Off**
  - Corrected cross-sectional interpretation; audited and removed survival-analysis/hazard language.
  - Updated Figure 13 to "Cross-Sectional Churn Prevalence by Contract Across Tenure Cohorts".
  - Designated One-Hot Encoding as default for `Contract`; documented equal-spacing assumptions of ordinal encoding.
  - Empirically reassessed `ChargesRatio` as an experimental candidate requiring explicit zero-tenure handling.
  - Renamed and formalized `OptionalServiceCount` across 6 internet add-on services.
  - Clarified `TotalCharges` 0.0 imputation as a Phase 3 pipeline validation step with unmodified raw CSV.
- [x] **Phase 3: Preprocessing, Feature Engineering & Leakage-Free Pipeline**
  - Leakage-safe 80/20 stratified partitioning with `split_manifest.json` persistence; `customerID` excluded.
  - Mandatory `TotalCharges` deterministic cleaning with strict zero-tenure missingness validation and 0.0 imputation.
  - Domain feature engineering: `OptionalServiceCount` (0-6), `AutomaticPayment`, and experimental `ChargesRatio`.
  - Scikit-learn `Pipeline` and `ColumnTransformer` factory functions for Linear and Tree model families.
  - Empirical leakage invariance verification proving training parameters are untouched by corrupted test sets.
  - Fully executed `notebooks/03_preprocessing_and_feature_engineering.ipynb` and `reports/preprocessing_report.md`.
  - 64 automated tests passing (100% pass rate) and clean Ruff static analysis.
- [x] **Phase 4: Model Exploration, Baseline Establishment & Controlled Tuning**
  - Modular modeling package `src/predictlab/models/` implementing Dummy baselines, Logistic Regression, Random Forest, and HistGradientBoosting.
  - Leakage-free Stratified 5-Fold Cross-Validation on $X_{\text{train}}$ only, tracking PR-AUC, ROC-AUC, F1, Balanced Accuracy, and Log Loss.
  - Experimental feature ablation proving that the 47-feature baseline outperforms the 52-feature experimental pipeline.
  - Controlled hyperparameter grid search optimizing HistGradientBoosting (`l2_regularization=5.0`, `learning_rate=0.03`, `max_iter=150`, `max_leaf_nodes=15`, `min_samples_leaf=40`).
  - Final candidate selection and single one-time holdout test evaluation ($N = 1,409$): PR-AUC of **0.6617**, ROC-AUC of **0.8481**, Log Loss of **0.4139**.
  - Serialized model bundle artifact (`artifacts/model.joblib`) with comprehensive provenance metadata.
  - Clean execution of `notebooks/04_model_training_and_comparison.ipynb` and comprehensive `reports/model_comparison_report.md`.
  - 92 passing automated unit and integration tests (100% pass rate) and clean Ruff static analysis.
- [x] **Phase 5: Probability Calibration, Threshold Tuning, Explainability & Production Inference (Audited in Phase 5.1)**
  - Modular packages `predictlab.calibration`, `predictlab.thresholding`, `predictlab.explainability`, and `predictlab.inference`.
  - Leakage-free 5x5 nested cross-validation evaluating Uncalibrated, Sigmoid (Platt), and Isotonic calibration across proper scoring rules (Brier score, Log Loss), ECE (uniform and quantile binning), Murphy (1973) Brier score decomposition, and Cox calibration regression.
  - Selected Sigmoid calibration as champion: Brier Score **0.1335 ± 0.0056**, Log Loss **0.4116 ± 0.0133**, PR-AUC **0.6697 ± 0.0215**, ROC-AUC **0.8498 ± 0.0123**, Quantile ECE **0.0265**. Production artifact is a 5-fold ensemble averaging calibrated probability outputs.
  - Systematic threshold grid sweep (81 points, 0.10 to 0.90) and financial cost modeling. Reconciled 4-term Comprehensive Incentive Model ($C_{\text{TP}}=\$100, C_{\text{FP}}=\$100, C_{\text{FN}}=\$500, C_{\text{TN}}=\$0$), achieving cost-optimal $\tau^* = 0.19$, cutting undetected churn by $72.7\%$, and saving **$20.75\%$ ($\$96,900$)** vs default $\tau=0.50$. Reconciled 2-term Misclassification-Only Model ($\tau^*=0.18$, saving **$38.20\%$ ($\$147,900$)**).
  - Nested CV policy evaluation confirming empirical policy stability across folds (mean fold threshold $\tau = 0.1980 \pm 0.0098$, out-of-fold F1 = $0.6159$, optimism delta = $-0.0012$).
  - Contiguous customer risk tiers: Low ($[0.0, 0.20)$), Medium ($[0.20, 0.50)$), High ($[0.50, 1.00]$), functionally decoupled from the binary classification threshold.
  - Cross-validated permutation importance evaluated on held-out validation folds across 5 splits with 5 repeats, highlighting Contract type, tenure, Fiber optic internet, TotalCharges, MonthlyCharges, and OnlineSecurity, while accounting for collinear feature dilution.
  - Multi-estimator local TreeSHAP attribution aggregated across all 5 ensemble tree estimators on the ensemble decision margin log-odds scale ($\bar{E}[f(x)] \approx -1.5438$), verifying exact additivity with non-causal interpretation guidelines.
  - Production-ready `InferenceEngine` supporting strict 19-predictor Pydantic v2 validation (`CustomerInputRecord`), zero-tenure `TotalCharges` whitespace handling, customerID stripping, atomic and per-record batch error handling modes, deterministic scoring, and verified sub-millisecond vectorized batch latency ($0.121$ ms/record).
  - Serialized production artifact `artifacts/model_phase5_calibrated.joblib`; preserved approved Phase 4 model at `artifacts/model_phase4.joblib`.
  - Four dedicated Phase 5 markdown reports and executed `notebooks/05_evaluation_explainability_and_inference.ipynb`.
  - 126 passing automated unit and integration tests (100% pass rate) and clean Ruff static analysis.
- [x] **Phase 6: Production-Quality Streamlit Application**
  - Modular multipage Streamlit interface (`app.py`, `app_pages/`, `src/predictlab/app/`) integrated directly with the production `InferenceEngine`.
  - **Page 1 — Overview Dashboard**: Verified source dataset statistics (7,043 records, 26.54% churn prevalence), approved 5-fold CV performance benchmarks (Average Precision 0.6684, ROC-AUC 0.8495), cost-sensitive threshold shift ($\tau^* = 0.19$, saving $20.75\%$ / $\$96,900$), and risk tier distribution.
  - **Page 2 — Single Customer Prediction**: Interactive form with the exact 19 raw domain predictors, representative subscriber profile presets, operating decision threshold slider, calibrated probability estimation, decoupled risk tier queue assignment, and local TreeSHAP attribution on the averaged base decision margin.
  - **Page 3 — Batch Customer Analysis**: CSV cohort upload with template download, per-record schema validation diagnostics capturing invalid rows without pipeline termination, vectorized scoring preserving original row ordering, risk tier donut chart, filterable results table, and full CSV exports.
  - **Page 4 — Model Insights & Explainability**: Ranked 5-fold cross-validated permutation importance chart with uncertainty bars, TreeSHAP ensemble log-odds attribution methodology with exact additivity guarantee ($2.2 \times 10^{-16}$), risk tier calibration table, and interactive threshold decoupling simulator.
  - **Page 5 — Methodology & Governance**: Comprehensive transparent documentation covering cross-sectional classification framing, dataset lineage and SHA-256 validation, leakage-free 47-feature baseline preprocessing, model selection, calibration rules, 4-term cost modeling, and explicit limitations (unverified historical 21-case narrative discrepancy, strict holdout test isolation, and non-causal guidance).
  - Automated application test suite with 144 passing tests (100% pass rate) and clean Ruff static analysis.
- [ ] **Phase 7: Packaging, Verification & Production Readiness**

---

## 6. Launching the Streamlit Application

Launch the interactive analytical platform locally using Streamlit:

```powershell
# Launch Streamlit web dashboard
streamlit run app.py
```

The application will initialize at `http://localhost:8501`, automatically loading the cached canonical Phase 5 calibrated model artifact (`artifacts/model_phase5_calibrated.joblib`).

### Application Features & Pages:
1. **📊 Overview Dashboard**: Macro view of dataset parameters, cross-validation metrics, cost curve savings, and risk distributions.
2. **👤 Single Customer Prediction**: Real-time scoring for individual subscribers with profile presets, threshold slider, risk badge, and local TreeSHAP attributions.
3. **📁 Batch Customer Analysis**: High-throughput CSV file upload, automated per-record validation, portfolio risk donut chart, and scored CSV export.
4. **🔍 Model Insights & Explainability**: Global permutation importance with error bars, SHAP properties, and interactive threshold vs risk tier simulator.
5. **📖 Methodology & Governance**: Complete scientific audit trail, data lineage, calibration protocols, and explicit scope limitations.

---

## 7. Setup & Installation (Windows 11 / Cross-Platform)

### 1. Prerequisites
- Python 3.12.10 installed on your system.
- Git installed.

### 2. Install Package and Dependencies
Install the package in editable mode with development dependencies:
```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Verify Dataset Checksum
The raw dataset is stored under `data/raw/Telco-Customer-Churn.csv`. To verify its integrity:
```powershell
python -c "import hashlib; print(hashlib.sha256(open('data/raw/Telco-Customer-Churn.csv','rb').read()).hexdigest())"
# Expected: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

### 4. Run Quality Checks & Test Suite
```powershell
# Run Ruff linting and formatting verification
ruff check .

# Run complete automated test suite
pytest -v
```
