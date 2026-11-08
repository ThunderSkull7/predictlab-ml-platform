# PredictLab Architecture Specification

## 1. System Overview

PredictLab is engineered as a modular, production-grade predictive analytics platform. It adheres to strict separation of concerns, ensuring machine learning pipelines, data validation, and business logic remain completely decoupled from user presentation interfaces.

```mermaid
graph TD
    subgraph Data Layer
        A[Raw CSV Data<br/>data/raw/] --> B[Data Loader<br/>predictlab.data.load]
        B --> C[Data Validator<br/>predictlab.data.validate]
    end

    subgraph Core ML Engine [src/predictlab]
        C --> D[Preprocessing & Feature Pipelines<br/>Phase 3]
        D --> E[Model Exploration & Baseline Comparison<br/>Phase 4]
        E --> F[Probability Calibration & Evaluation<br/>Phase 5]
        F --> G[Serialized Artifacts<br/>models/ & artifacts/]
    end

    subgraph Presentation Layer [app/]
        G --> H[Streamlit UI<br/>Phase 6]
        H --> I[Cohort Risk Analytics]
        H --> J[Single-Customer Scoring]
        H --> K[What-If Scenario Simulation]
    end

    subgraph Cross-Cutting Infrastructure
        CFG[Config System<br/>predictlab.config] -.-> B
        CFG -.-> C
        CFG -.-> D
        CFG -.-> E
        CFG -.-> H
        LOG[Structured Logging<br/>predictlab.logging_config] -.-> B
        LOG -.-> C
        LOG -.-> E
    end
```

---

## 2. Directory Layout & Module Responsibilities

```text
predictlab-ml-platform/
│
├── .github/workflows/          # CI automation (linting, automated tests)
├── .streamlit/                 # UI presentation themes and server settings
├── data/
│   ├── raw/                    # Immutable raw source data
│   ├── processed/              # Transformed feature sets (git-ignored)
│   └── README.md               # Data registry & provenance documentation
├── docs/                       # Technical architecture, data dictionaries, ADRs
│   ├── architecture.md
│   ├── problem_definition.md
│   ├── data_dictionary.md
│   ├── leakage_review.md
│   └── decisions/
├── src/predictlab/             # Core Python package (business & ML logic)
│   ├── config.py               # Central configuration & dynamic path resolution
│   ├── logging_config.py       # Application logging
│   ├── data/                   # Data ingestion and schema validation
│   │   ├── load.py
│   │   └── validate.py
│   ├── profiling/              # Profiling, quality auditing, and diagnostic insights (Phase 2)
│   │   ├── __init__.py
│   │   ├── quality.py          # Data quality audit & TotalCharges anomaly investigation
│   │   ├── summary.py          # Univariate numeric and categorical summaries
│   │   ├── insights.py         # Segment churn rates, Chi-square & Mann-Whitney tests
│   │   └── visuals.py          # Publication-grade visual figure generation
│   ├── preprocessing/          # Leakage-free preprocessing, feature engineering & pipelines (Phase 3)
│   │   ├── __init__.py
│   │   ├── schema.py           # Feature definitions, contracts & domain boundaries
│   │   ├── cleaning.py         # Deterministic TotalCharges cleaning & zero-tenure validation
│   │   ├── feature_engineering.py # OptionalServiceCount, AutomaticPayment & ChargesRatio
│   │   ├── transformers.py     # Scikit-learn compatible custom transformers
│   │   ├── split.py            # Stratified holdout partitioning & manifest persistence
│   │   └── pipelines.py        # Linear and tree ColumnTransformer pipeline factories
│   ├── models/                 # Model benchmarking, tuning, ablation & selection (Phase 4)
│   │   ├── __init__.py
│   │   ├── baselines.py        # DummyClassifier benchmarks (prior, most_frequent)
│   │   ├── estimators.py       # Candidate model factories (LR, RF, HistGradientBoosting)
│   │   ├── evaluation.py       # Multi-metric evaluation (PR-AUC, ROC-AUC, F1, Log Loss)
│   │   ├── cross_validation.py # Leakage-free Stratified 5-Fold cross-validation
│   │   ├── experiments.py      # Benchmark runner & feature ablation studies
│   │   ├── tuning.py           # Controlled hyperparameter tuning on training partition
│   │   ├── selection.py        # Objective model selection & holdout evaluation
│   │   └── persistence.py      # Artifact bundle and experiment JSON serialization
│   ├── calibration/            # Probability calibration, curves & scoring rules (Phase 5)
│   │   ├── __init__.py
│   │   ├── metrics.py          # Brier score, Log Loss, ECE, Murphy decomposition, Cox regression
│   │   ├── curves.py           # Calibration reliability curves & distribution histograms
│   │   ├── cross_validation.py # Leakage-free 5x5 nested CV across calibration strategies
│   │   └── selection.py        # Multi-criteria calibration selection & ensemble refit
│   ├── thresholding/           # Decision threshold analysis & operational risk tiering (Phase 5)
│   │   ├── __init__.py
│   │   ├── metrics.py          # 81-step threshold sweep across classification metrics
│   │   ├── costs.py            # Configurable CostMatrix & cost ratio sensitivity sweep
│   │   ├── optimization.py     # Threshold optimization & nested CV policy optimism audit
│   │   └── risk_tiers.py       # Contiguous 3-tier risk partitioner & cohort profiler
│   ├── explainability/         # Global & local model interpretation (Phase 5)
│   │   ├── __init__.py
│   │   ├── global_importance.py # 5-fold cross-validated permutation importance with Average Precision
│   │   └── local_attribution.py # Multi-estimator TreeSHAP local attribution on ensemble margin log-odds scale
│   └── inference/              # Production-grade single & batch inference engine (Phase 5)
│       ├── __init__.py
│       ├── schemas.py          # Pydantic v2 input/output schemas & predictor validation
│       └── engine.py           # InferenceEngine with atomic & per-record batch error handling
├── app/                        # Streamlit web application (Phase 6)
├── models/                     # Serialized model binaries (Phase 4-5)
├── artifacts/                  # Serialized transformers, metrics, and figures
├── reports/                    # Generated evaluation summaries and figures
├── tests/                      # Automated test suite
│   ├── fixtures/               # Deterministic synthetic test data
│   ├── unit/                   # Isolated component tests
│   └── integration/            # End-to-end data pipeline tests
├── pyproject.toml              # Build, dependencies, pytest, and ruff settings
└── README.md                   # Platform documentation
```

---

## 3. Core Module Responsibilities

### A. Configuration (`predictlab.config`)
- **Single Source of Truth**: Dynamically resolves project root and all asset directories without hardcoding machine-specific absolute paths.
- **Environment Overrides**: Powered by `pydantic-settings` to enable seamless continuous integration, local testing, and staging environment variable overrides.
- **Domain Constants**: Holds canonical schema specifications, target column naming, and class definitions.

### B. Logging (`predictlab.logging_config`)
- **Structured Standard Formatting**: Timestamps, log levels, file/function/line coordinates, and clean messages.
- **Idempotency**: Prevents duplicate handlers when imported across test runners, batch workers, and interactive Streamlit sessions.
- **Safety**: Strictly avoids logging personal identifiers or raw sensitive payload data.

### C. Ingestion & Validation (`predictlab.data.load` & `predictlab.data.validate`)
- **Immutability**: Ingestion functions return isolated copies, leaving raw source files untouched.
- **Strict Error Boundaries**: Handles missing files, 0-byte corruptions, duplicate CSV headers, and parsing failures gracefully with domain-specific exceptions (`DataValidationError`, `EmptyDatasetError`, `DuplicateColumnsError`, etc.).
- **Active Pydantic Validation**: Validation functions produce strongly-typed `ValidationResult` objects with active consistency validation (ensuring validity flags, error lists, and row bounds remain strictly synchronized).

### D. Profiling & Diagnostic Insights (`predictlab.profiling`)
- **Data Quality Auditing (`quality.py`)**: Automates deep structural quality scans, whitespace detection, and dedicated empirical audits of anomalous attributes (such as the 11 zero-tenure `TotalCharges` records).
- **Statistical Summaries (`summary.py`)**: Calculates robust distribution statistics (mean, median, IQR, skewness) and categorical frequency tables without mutating raw inputs.
- **Inferential Testing (`insights.py`)**: Computes segment churn rates, Chi-square tests of independence with Benjamini-Hochberg False Discovery Rate (FDR) adjustments, bias-corrected Cramér's V, Mann-Whitney U, and Welch's t-tests.
- **Visualization Suite (`visuals.py`)**: Generates 16 publication-grade figures saved to `reports/figures/` for exploratory reporting and notebook integration.

### E. Preprocessing & Feature Engineering (`predictlab.preprocessing`)
- **Dataset Partitioning (`split.py`)**: Executes 80/20 stratified holdout partitioning, target extraction, identifier removal, and lightweight manifest serialization (`split_manifest.json`).
- **Data Cleaning (`cleaning.py`)**: Implements strict `TotalCharges` whitespace-to-NaN parsing, zero-tenure validation guardrails, and $0.0$ domain imputation.
- **Feature Engineering (`feature_engineering.py`)**: Implements domain feature extraction (`OptionalServiceCount`, `AutomaticPayment`, and experimental `ChargesRatio`).
- **Custom Transformers (`transformers.py`)**: Exposes scikit-learn compatible transformers (`TotalChargesCleaner`, `OptionalServiceCounter`, `ColumnDropper`, etc.) with `get_feature_names_out()`.
- **Pipeline Factories (`pipelines.py`)**: Assembles production `ColumnTransformer` pipelines for both linear models (with standardization) and tree models (preserving natural units). Includes Joblib serialization utilities.

### F. Modeling, Evaluation & Selection (`predictlab.models`)
- **Baseline Establishment (`baselines.py`)**: Implements DummyClassifier benchmarks (`prior`, `most_frequent`, `stratified`) wrapped in uniform scikit-learn pipelines.
- **Candidate Model Factories (`estimators.py`)**: Encapsulates pipeline composition for Logistic Regression (L1/L2), Random Forest (unconstrained, constrained, balanced), and Histogram Gradient Boosting.
- **Multi-Metric Evaluation (`evaluation.py`)**: Computes ranking metrics (Average Precision / PR-AUC, ROC-AUC), classification metrics (F1, Precision, Recall, Balanced Accuracy), and probabilistic loss (Log Loss, Brier Score), alongside confusion matrices and curve coordinates.
- **Leakage-Free Cross-Validation (`cross_validation.py`)**: Enforces fold-level pipeline fitting using stratified 5-fold CV, tracking mean, standard deviation, fit times, and warnings.
- **Experimental Ablation (`experiments.py`)**: Empirically benchmarks candidate model families and executes controlled feature ablation comparing the 47-feature baseline against the 52-feature experimental pipeline.
- **Controlled Tuning (`tuning.py`)**: Executes bounded hyperparameter search on the training partition only using PR-AUC as the primary optimization target.
- **Model Selection & Holdout Evaluation (`selection.py`)**: Programmatically selects the optimal candidate based on multi-criteria ranking and executes a single holdout evaluation on the protected test partition.
- **Persistence (`persistence.py`)**: Serializes experiment reports to `reports/experiments/` and bundles the final fitted pipeline with complete provenance metadata to `artifacts/model.joblib`.

### G. Probability Calibration (`predictlab.calibration`)
- **Proper Scoring Rules & Metrics (`metrics.py`)**: Calculates Brier score, Log Loss, uniform and quantile Expected Calibration Error (ECE), Murphy (1973) Brier score decomposition (Reliability, Resolution, Uncertainty), and Cox calibration regression (intercept and slope).
- **Reliability Curves (`curves.py`)**: Produces empirical reliability curve coordinates and probability distribution histograms.
- **Nested Cross-Validation (`cross_validation.py`)**: Executes leakage-free 5x5 nested CV comparing uncalibrated, Sigmoid (Platt), and Isotonic calibration without data reuse.
- **Calibration Champion Selection (`selection.py`)**: Automates objective multi-criteria calibration selection and fits production 5-fold ensemble `CalibratedClassifierCV` pipelines on the training dataset.

### H. Decision Thresholding & Cost Optimization (`predictlab.thresholding`)
- **Systematic Threshold Sweeps (`metrics.py`)**: Evaluates 81 discrete thresholds ($\tau \in [0.10, 0.90]$) across Precision, Recall, Specificity, F1, F2, F0.5, Balanced Accuracy, and confusion counts.
- **Cost-Sensitive Decision Modeling (`costs.py`)**: Implements configurable `CostMatrix` structures using the 4-term expected cost formula ($\text{TotalCost} = C_{\text{FP}}\text{FP} + C_{\text{FN}}\text{FN} + C_{\text{TP}}\text{TP} + C_{\text{TN}}\text{TN}$), calculating total operational risk cost, cost-per-customer, and cost sensitivity across asymmetric loss ratios ($C_{\text{FN}} : C_{\text{FP}}$ from 0.2:1 to 10:1).
- **Policy Optimism Audit (`optimization.py`)**: Executes nested CV policy evaluation (`evaluate_threshold_policy_nested`) to quantify empirical threshold policy stability across splits.
- **Operational Customer Risk Tiers (`risk_tiers.py`)**: Enforces contiguous probability partitioning across Low ($[0.0, 0.20)$), Medium ($[0.20, 0.50)$), and High ($[0.50, 1.00]$) tiers with statistical cohort profiling, decoupled from binary decision cutoffs.

### I. Global & Local Explainability (`predictlab.explainability`)
- **Global Permutation Importance (`global_importance.py`)**: Measures Average Precision drop across features via 5-fold cross-validation strictly on held-out validation splits with 5 shuffle repeats, accounting for collinear feature dilution.
- **Local TreeSHAP Attributions (`local_attribution.py`)**: Computes exact additive Shapley values aggregated across all 5 ensemble tree estimators on the ensemble margin log-odds scale using `TreeShapExplainer`, identifying positive (churn-elevating) and negative (retention-stabilizing) feature drivers with explicit non-causal interpretation warnings.

### J. Production-Ready Inference Engine (`predictlab.inference`)
- **Strict Pydantic v2 Schemas (`schemas.py`)**: Validates the 19 raw customer predictors, handles zero-tenure `TotalCharges` whitespace strings safely, strips `customerID`, and returns structured single/batch prediction records.
- **Unified Inference Engine (`engine.py`)**: Supports single-record scoring, high-throughput batch scoring, dual error-handling modes (`atomic` transactional validation vs `per_record` resilient isolation), configurable threshold overrides, risk tiering, reproducible latency benchmarks, and optional local TreeSHAP explanations.

---

## 4. Probabilistic Modeling Terminology & Phased Governance

To avoid methodological ambiguity, PredictLab enforces precise distinctions across model outputs:

1. **Model-Estimated Class Probabilities (Uncalibrated)**: Raw classifier outputs (e.g. tree fraction scores). Lower log loss reflects superior proper scoring rather than empirical calibration.
2. **Empirically Calibrated Probabilities (Phase 5)**: Post-processed probabilities validated against true empirical event frequencies using Sigmoid (Platt) scaling, verified via Brier score ($0.1335$), Log Loss ($0.4116$), and reliability diagrams. Evaluated via a 5-fold bagging ensemble.
3. **Binary Decisions (Phase 5)**: Derived by applying an explicit, cost-optimized threshold ($\tau^* = 0.19$ with an illustrative 20.75% modeled cost reduction under a 4-term operational scenario, or $\tau^* = 0.18$ saving an illustrative 38.20% under a 2-term misclassification penalty scenario over naive $0.50$ demarcation).
4. **Business Risk Tiers (Phase 5)**: Standardized contiguous segments (Low $[0.0, 0.20)$, Medium $[0.20, 0.50)$, High $[0.50, 1.00]$) mapping continuous probabilities to tiered operational intervention workflows, conceptually decoupled from binary decision thresholds.

---

## 5. Separation of ML Logic and User Interface

- **Headless Core (`src/predictlab/`)**: Core data loading, preprocessors, estimators, and inference routines operate independently of Streamlit or any web framework.
- **UI Consumer (`app/`)**: The Streamlit interface acts solely as an interactive client consuming `predictlab` APIs and serialized joblib artifacts.
- **State Protection**: User interactions in the UI cannot mutate raw data assets or trained model binaries.
