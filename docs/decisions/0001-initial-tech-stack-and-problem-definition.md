# ADR 0001: Initial Tech Stack, Problem Definition, Dataset Selection, and Provenance Audit

## Status
Accepted (Amended in Phase 1.1)

## Date
2026-11-07 (Amended: 2026-11-07)

## Context
PredictLab is designed as a portfolio-grade predictive analytics platform. Phase 1 requires selecting an initial business use case, identifying a verifiable dataset with audited provenance and licensing, defining the core technology stack, and creating foundational configuration, data loading, and validation layers. Phase 1.1 conducted a methodological audit of licensing terms, problem formulation, probability claims, and dependency versions.

## Decisions

### 1. Business Problem & Problem Formulation
- **Task**: Retrospective customer churn classification and risk ranking using the available customer snapshot.
- **Dataset**: IBM Telco Customer Churn sample dataset (`Telco-Customer-Churn.csv`).
- **Methodological Scope**: The dataset is a static cross-sectional table without timestamps or longitudinal customer histories. The initial machine learning task is formally delimited to retrospective classification and risk ranking rather than future forecasting. Long-term platform goals include longitudinal forecasting when time-stamped event data becomes available.

### 2. Dataset Provenance & Licensing Audit
- **Host Repository**: `IBM/telco-customer-churn-on-icp4d`
- **Upstream License**: Apache License 2.0 (governing code pattern software).
- **Data Terms Assessment**: No separate open-data license (e.g., CDLA, CC0, CC-BY) is provided in the repository for the CSV file. IBM hosts the dataset as sample demonstration material rather than an official open-data publication.
- **Policy**: Redistribution rights cannot be conclusively established from upstream metadata alone. PredictLab distinguishes its own MIT software license from third-party sample data terms, documents this limitation, and provides an automated download and SHA-256 verification procedure.
- **Integrity**: SHA-256 verified (`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`), 7,043 rows, 21 columns, zero duplicate records.

### 3. Model Output Terminology & Phased Governance
- PredictLab strictly distinguishes four output levels:
  1. Model-Estimated Class Probability (Uncalibrated) ($\hat{s}_i \in [0, 1]$): Model-estimated class probabilities that have not yet undergone empirical calibration assessment.
  2. Empirically Calibrated Probability ($\hat{p}_i = P(Y_i = 1 \mid \hat{s}_i)$).
  3. Binary Classification Decision ($\hat{Y}_i \in \{0, 1\}$ via threshold $\tau$).
  4. Business Risk Tier (e.g., Low, Medium, High).
- **Status**: No model has yet been trained or calibrated in Phase 1. Model outputs will not be labeled as "calibrated" until empirical calibration methods (Platt scaling, isotonic regression) and metrics (Brier score, ECE) are evaluated in Phase 5.
- **Risk Tiers**: Risk-tier threshold boundaries are explicitly deferred to Phase 5 when actual model score distributions and business decision utilities are available.

### 4. Core Technology Stack (Approved & Restored Stack)
- **Runtime**: Python 3.12.10.
- **Machine Learning**: Scikit-Learn 1.6.1 (exact pinned version restored).
- **Visualization**: Altair 5.5.0 (exact pinned version restored), Matplotlib 3.10.5, Seaborn 0.13.2.
- **User Interface (Future)**: Streamlit 1.67.0.
- **Data Validation & Settings**: Pydantic v2 (2.13.4) and `pydantic-settings` (2.15.0).
- **Testing & Quality**: Pytest 9.1.1 and Ruff 0.17.4.

### 5. Pending Preprocessing Decision: `TotalCharges` Whitespace
- Exactly 11 rows in `TotalCharges` contain `" "`, all corresponding to `tenure == 0` and `Churn == "No"`.
- The raw CSV is preserved completely unmodified.
- The proposed conversion to `0.0` (reflecting zero historical charges billed at enrollment) is formally recorded as a pending preprocessing decision to be investigated in Phase 2 and implemented in Phase 3.

### 6. Architectural Guardrails
- `customerID` is strictly excluded from predictive model inputs to prevent memorization.
- Paths are resolved dynamically relative to `PROJECT_ROOT = Path(__file__).resolve().parents[2]` without machine-specific hardcoding.
- Strict isolation is maintained between core ML logic (`src/predictlab/`) and future presentation UI (`app/`).

## Consequences
- The project foundation maintains complete methodological rigor and transparent provenance documentation.
- Downstream modeling phases have unambiguous terminology and explicit evaluation milestones.
