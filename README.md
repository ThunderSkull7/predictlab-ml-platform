# PredictLab — Portfolio-Grade Predictive Analytics Platform

PredictLab is an end-to-end, interactive predictive analytics and machine learning platform engineered to demonstrate production-level ML engineering standards, probabilistic modeling, calibration, and explainable decision support.

The initial machine learning application is **Customer Churn Prediction**, modeled using the verified IBM Telco Customer Churn dataset.

---

## 1. Problem Overview

In recurring revenue and subscription businesses, customer acquisition is substantially more expensive than customer retention. Unplanned customer churn erodes monthly recurring revenue (MRR) and decreases customer lifetime value (LTV).

PredictLab formulates churn prediction as a **probabilistic classification problem**, estimating the calibrated likelihood that an active subscriber will churn within an upcoming cycle:

$$P(\text{Churn} \mid \mathbf{x}) \in [0.0, 1.0]$$

Rather than outputting naive binary decisions, PredictLab preserves probability calibration and confidence bands to enable cost-sensitive decision-making, tier-based retention campaigns, and expected-loss calculations.

---

## 2. Technology Stack

| Layer | Component | Version / Specification |
|---|---|---|
| **Runtime** | Python | 3.12.10 |
| **Data Processing** | Pandas, NumPy | Modern Python 3.12 compatible releases |
| **Machine Learning** | Scikit-learn, Scipy, Joblib | 1.6+ |
| **Validation & Settings** | Pydantic, Pydantic-Settings | 2.x |
| **Visualization** | Altair, Matplotlib, Seaborn | Latest stable |
| **User Interface (Future)** | Streamlit | 1.67+ |
| **Code Quality & Linting** | Ruff | 0.17+ |
| **Testing** | Pytest | 9.x |
| **Version Control & CI** | Git, GitHub Actions | Matrix Python 3.12 CI |

---

## 3. Project Structure

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
│   ├── problem_definition.md          # Business problem & ML formulation
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
│           ├── load.py                # CSV ingestion with error boundaries
│           └── validate.py            # Schema & integrity validation
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
│   │   └── test_validate.py           # Schema & integrity validation tests
│   └── integration/
│       └── test_data_pipeline.py      # End-to-end ingestion/validation pipeline
│
├── .env.example                       # Environment variable template
├── .gitignore                         # Comprehensive ignore rules
├── LICENSE                            # MIT License & Third-Party Notices
├── README.md                          # Platform documentation
└── pyproject.toml                     # Project metadata, dependencies & tool configs
```

---

## 4. Implementation Roadmap

PredictLab is developed following a rigorous 7-phase engineering plan:

- [x] **Phase 1: Foundation, Architecture & Problem Definition** *(Current)*
  - Environment setup, project structure, dynamic configuration, structured logging.
  - Authoritative dataset acquisition, provenance verification, and SHA-256 validation.
  - Detailed problem definition, comprehensive data dictionary, and data leakage review.
  - Foundational data loading and validation modules with 100% test coverage.
- [ ] **Phase 2: Exploratory Data Analysis & Diagnostic Insights**
- [ ] **Phase 3: Preprocessing, Feature Engineering & Leakage-Free Pipeline**
- [ ] **Phase 4: Model Exploration, Baseline Establishment & Comparison**
- [ ] **Phase 5: Probability Calibration, Threshold Tuning & Evaluation**
- [ ] **Phase 6: Interactive Dashboard & Explainable Inference UI**
- [ ] **Phase 7: Packaging, Verification & Production Readiness**

---

## 5. Setup & Installation (Windows 11 / Cross-Platform)

### Prerequisites
- Python 3.12.10 installed on your system.
- Git installed.

### 1. Clone or Open Workspace
```powershell
cd "d:\Project\Predictive Analytics Platform"
```

### 2. Configure Virtual Environment (Optional / Recommended)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Package and Dependencies
Install the package in editable mode with development dependencies:
```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 4. Verify Dataset
The raw dataset is stored under `data/raw/Telco-Customer-Churn.csv`. To verify its integrity:
```powershell
python -c "import hashlib; print(hashlib.sha256(open('data/raw/Telco-Customer-Churn.csv','rb').read()).hexdigest())"
# Expected: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

### 5. Run Quality Checks & Test Suite
```powershell
# Run Ruff linting and formatting verification
ruff check .

# Run complete automated test suite
pytest
```

---

## 6. Current Development Status

Phase 1 is complete. All foundation modules (`config`, `logging_config`, `data.load`, `data.validate`) are tested and passing all test suites and Ruff checks.

Future phases will be implemented sequentially upon approval of the Phase 1 Acceptance Gate.
