# ADR 0001: Initial Tech Stack, Problem Definition, and Dataset Selection

## Status
Accepted

## Date
2026-11-07

## Context
PredictLab is designed as a portfolio-grade predictive analytics and machine learning platform. To ensure high maintainability, testability, and real-world applicability, Phase 1 requires selecting an appropriate initial business use case, identifying a verifiable dataset with clear provenance and licensing, defining the core technology stack, and creating foundational configuration, data loading, and validation layers.

## Decisions

### 1. Business Problem & Dataset Selection
- **Task**: Customer Churn Prediction.
- **Dataset**: IBM Telco Customer Churn dataset (`Telco-Customer-Churn.csv`).
- **Provenance & License**: IBM Sample Data Sets repository (`IBM/telco-customer-churn-on-icp4d`), governed under Apache License 2.0, permitting redistribution, modification, and commercial application.
- **Integrity**: Verified with SHA-256 (`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`), 7,043 rows, 21 columns, zero duplicate records.

### 2. Output Formulation: Probabilistic Classification
- We formulate the machine learning task to predict **calibrated posterior probabilities** ($P(\text{Churn} \mid \mathbf{x}) \in [0.0, 1.0]$) rather than rigid binary labels.
- This empowers downstream business users to set asymmetric decision thresholds matching organizational intervention costs and calculate expected financial exposure.

### 3. Core Technology Stack
- **Runtime**: Python 3.12.10.
- **Data & Computation**: Pandas, NumPy.
- **Machine Learning**: Scikit-Learn 1.6+ (with Joblib serialization).
- **Validation & Settings**: Pydantic v2 and `pydantic-settings`.
- **Quality Assurance**: Ruff (linting and formatting) and Pytest (automated unit and integration testing).
- **Presentation (Future)**: Streamlit (isolated in `app/`).

### 4. Configuration & Modular Architecture
- Centralized configuration via `predictlab.config.Settings`.
- Project paths dynamically resolved relative to `PROJECT_ROOT = Path(__file__).resolve().parents[2]` to ensure zero machine-specific path hardcoding.
- Complete separation between `src/predictlab` (headless library) and `app/` (presentation layer).

### 5. Leakage & Feature Guardrails
- `customerID` is classified as a pure identifier and strictly excluded from predictive model inputs to avoid memorization.
- The 11 whitespace records in `TotalCharges` are mapped to brand-new accounts (`tenure == 0`) and flagged for numeric handling in Phase 3.
- All 19 candidate features are validated as available prior to or at the prediction cutoff point.

## Consequences
- The codebase begins with a robust, reproducible foundation verified by automated tests.
- Clear error handling and schema validation prevent downstream pipelines from processing silent corruptions.
- Future phases (EDA, Preprocessing, Modeling, Evaluation, UI) have clean extension points without technical debt.
