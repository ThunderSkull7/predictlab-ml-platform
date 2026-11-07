# PredictLab — Portfolio-Grade Predictive Analytics Platform

PredictLab is an end-to-end predictive analytics and machine learning platform engineered to demonstrate production-level ML engineering standards, probabilistic modeling, calibration analysis, and explainable decision support.

The initial machine learning task is formulated as:

> **Retrospective customer churn classification and risk ranking using the available customer snapshot.**

---

## 1. Problem Formulation & Scope

The initial dataset is the canonical IBM Telco Customer Churn sample dataset. Because this dataset is a single static cross-sectional table lacking observation timestamps, event logs, or longitudinal customer histories, the system models statistical patterns distinguishing accounts that churned from accounts that remained active within the snapshot.

PredictLab distinguishes four distinct levels of model outputs:
1. **Raw Model-Estimated Score / Pseudo-Probability** ($\hat{s}_i \in [0, 1]$): Uncalibrated score produced by an estimator.
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
- **Pending Decision on `TotalCharges`**: Exactly 11 rows contain `" "` in `TotalCharges`, coinciding strictly with `tenure == 0` and `Churn == "No"`. The raw CSV is preserved completely unmodified. The proposed `0.0` interpretation is recorded as a pending preprocessing decision to be investigated in Phase 2 and implemented in Phase 3.

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
├── notebooks/                         # Exploratory research notebooks
│
├── src/
│   └── predictlab/                    # Headless core Python package
│       ├── __init__.py
│       ├── config.py                  # Pydantic Settings & dynamic path resolution
│       ├── logging_config.py          # Structured logging configuration
│       └── data/
│           ├── __init__.py
│           ├── load.py                # CSV ingestion with error boundaries & duplicate header detection
│           └── validate.py            # Active Pydantic schema & consistency validation
│
├── app/                               # Streamlit UI application (Phase 6)
├── models/                            # Serialized model binaries (Phase 4)
├── artifacts/                         # Preprocessors, encoders, metrics (Phase 3-5)
├── reports/                           # Generated evaluation reports & figures
│
├── tests/
│   ├── fixtures/
│   │   └── synthetic_data.py          # Deterministic test data generator
│   ├── unit/
│   │   ├── test_config.py             # Config & path resolution tests
│   │   ├── test_load.py               # Ingestion & error-handling tests
│   │   └── test_validate.py           # Schema & Pydantic consistency validation tests
│   └── integration/
│       └── test_data_pipeline.py      # End-to-end ingestion/validation pipeline
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
- [ ] **Phase 2: Exploratory Data Analysis & Diagnostic Insights**
- [ ] **Phase 3: Preprocessing, Feature Engineering & Leakage-Free Pipeline**
- [ ] **Phase 4: Model Exploration, Baseline Establishment & Comparison**
- [ ] **Phase 5: Probability Calibration, Threshold Tuning & Evaluation**
- [ ] **Phase 6: Interactive Dashboard & Explainable Inference UI**
- [ ] **Phase 7: Packaging, Verification & Production Readiness**

---

## 6. Setup & Installation (Windows 11 / Cross-Platform)

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
