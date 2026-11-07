# PredictLab — Phase 5: Production-Ready Inference Engine Report

## 1. Executive Summary

This report documents the architectural design, schema validation protocols, error handling mechanisms, and performance verification of the **Production-Ready Inference Engine** implemented in Phase 5 of PredictLab.

Transitioning machine learning models from offline cross-validation into production operations requires rigorous engineering safeguards. Real-world input payloads frequently contain missing attributes, malformed data types, non-finite values, and anomalous whitespace strings. In addition, operational systems require both real-time, low-latency single-customer scoring and high-throughput batch scoring with transactional error isolation.

Phase 5 delivers [`InferenceEngine`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/inference/engine.py#L27) and its companion Pydantic v2 schemas:
- **Strict Input Schema Validation**: Implemented via [`CustomerInputRecord`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/inference/schemas.py#L70), validating all 19 raw predictors. Rejects invalid categorical levels, negative tenure, and non-finite charges.
- **Identifier Decoupling**: If an optional `customerID` is supplied in raw payloads, the engine validates its uniqueness, preserves it in batch response mappings, and strictly strips it prior to model transformation, ensuring identifiers never enter the feature matrix.
- **Zero Transformation Duplication**: Raw dictionaries or DataFrames are passed directly to the encapsulated scikit-learn pipeline (`preprocessor.transform()`), guaranteeing that inference transforms are 100% bit-for-bit identical to training transforms.
- **Dual Batch Error Handling Modes**: Supports both `atomic` mode (transactional validation where any single error rejects the batch) and `per_record` mode (graceful degradation where valid records are scored and invalid records return localized diagnostics).
- **Deterministic Calibrated Inference**: Loads the serialized Phase 5 calibrated artifact (`model_phase5_calibrated.joblib`), applying the cost-optimal decision threshold ($\tau^* = 0.18$) and contiguous risk tiering.
- **Integrated Local TreeSHAP Explanations**: Seamlessly generates local feature attributions on demand, identifying top churn and retention drivers without external script dependencies.

> [!IMPORTANT]
> **Cross-Sectional Scope Limitation**: The dataset is a cross-sectional snapshot of customer accounts at a single observation point. All model evaluations represent retrospective classification and risk ranking across subscriber accounts, **not temporal forecasting or forward-looking event prediction**.

---

## 2. Architectural Design & Schema Specifications

The inference engine architecture follows a modular, defensively programmed pipeline:

```
Raw Input Payload (Dict or DataFrame)
      │
      ▼
Pydantic v2 Schema Validation (CustomerInputRecord)
├── Validates 19 raw customer attributes
├── Extracts and decouples 'customerID'
├── Handles whitespace TotalCharges (tenure == 0)
└── Rejects invalid categories, negative tenure, NaNs
      │
      ▼
Encapsulated Pipeline Execution
├── Preprocessor: 19 raw inputs -> 47 baseline features
└── Calibrated Classifier: Computes calibrated probability P(Churn)
      │
      ▼
Operational Decision Rules & Tiering
├── Applies Decision Threshold (default: tau* = 0.18)
└── Assigns Operational Risk Tier (Low, Medium, High)
      │
      ▼
Optional Local Explainability (TreeShapExplainer)
└── Computes exact additive margin SHAP values & drivers
      │
      ▼
Structured Output (SinglePredictionOutput / BatchPredictionOutput)
```

### 2.1 The 19 Raw Predictors

The input schema enforces strict typing for the canonical 19 predictors:

| Field Name | Type | Allowed Values / Constraints | Validation Rule |
|---|---|---|---|
| `gender` | `str` | `"Male"`, `"Female"` | Exact string match |
| `SeniorCitizen` | `int` | `0`, `1` | Casts and validates binary integer |
| `Partner` | `str` | `"Yes"`, `"No"` | Exact string match |
| `Dependents` | `str` | `"Yes"`, `"No"` | Exact string match |
| `tenure` | `float` | $\ge 0.0$, finite | Rejects negative or non-finite values |
| `PhoneService` | `str` | `"Yes"`, `"No"` | Exact string match |
| `MultipleLines` | `str` | `"Yes"`, `"No"`, `"No phone service"` | Exact string match |
| `InternetService` | `str` | `"DSL"`, `"Fiber optic"`, `"No"` | Exact string match |
| `OnlineSecurity` | `str` | `"Yes"`, `"No"`, `"No internet service"` | Exact string match |
| `OnlineBackup` | `str` | `"Yes"`, `"No"`, `"No internet service"` | Exact string match |
| `DeviceProtection` | `str` | `"Yes"`, `"No"`, `"No internet service"` | Exact string match |
| `TechSupport` | `str` | `"Yes"`, `"No"`, `"No internet service"` | Exact string match |
| `StreamingTV` | `str` | `"Yes"`, `"No"`, `"No internet service"` | Exact string match |
| `StreamingMovies` | `str` | `"Yes"`, `"No"`, `"No internet service"` | Exact string match |
| `Contract` | `str` | `"Month-to-month"`, `"One year"`, `"Two year"` | Exact string match |
| `PaperlessBilling` | `str` | `"Yes"`, `"No"` | Exact string match |
| `PaymentMethod` | `str` | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` | Exact string match |
| `MonthlyCharges` | `float` | $\ge 0.0$, finite | Rejects negative or non-finite values |
| `TotalCharges` | `float` \| `str` | $\ge 0.0$, finite or `" "` (whitespace string) | Preserves whitespace string for zero-tenure accounts |

---

## 3. Input Validation & Error Handling Protocols

### 3.1 TotalCharges Whitespace Anomaly Handling
As established during Phase 1–3 audits, brand-new accounts with `tenure == 0` contain whitespace strings in `TotalCharges`.
- In `CustomerInputRecord`, field validators check if `TotalCharges` is a whitespace string (`" "`).
- If present, it is validated and passed safely to the `TreePreprocessor`, which imputes it to `$0.00` without triggering unhandled exceptions or data corruption.

### 3.2 Dual Batch Error Handling Modes

When executing `predict_batch(records, error_handling=...)`:

#### Mode A: `atomic` Mode (Default for Transactional Pipelines)
- Validates all input records upfront.
- If even one record violates schema constraints (e.g. `tenure = -5`, or `gender = "Unknown"`), the engine immediately raises `InferenceValidationError`, detailing the offending record index and validation failure.
- Guarantees transactional consistency: no partial predictions are emitted.

#### Mode B: `per_record` Mode (Designed for Bulk Processing)
- Processes records sequentially while maintaining input ordering.
- If a record is invalid, its result in `BatchPredictionOutput.predictions` is marked with `is_valid = False` and an informative `error_message`.
- Valid records in the batch are scored normally (`is_valid = True`, with full probability and tier assignments).
- Enables resilient data warehouse pipelines where bad records are routed to an error dead-letter queue without stalling batch processing.

---

## 4. API Usage Examples & Output Schemas

### 4.1 Single Record Prediction

```python
from predictlab.inference.engine import InferenceEngine

# Initialize engine (automatically loads artifacts/model_phase5_calibrated.joblib)
engine = InferenceEngine()

# Input customer record
payload = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 4,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 340.0,
}

# Execute prediction with local SHAP explanation
result = engine.predict_single(payload, include_explanation=True)

print(result.model_dump())
```

**Output Structure**:
```json
{
  "model_version": "5.0.0",
  "model_name": "HistGradientBoostingClassifier",
  "calibration_method": "sigmoid",
  "estimated_probability": 0.6842,
  "predicted_class": 1,
  "predicted_label": "Churned",
  "applied_threshold": 0.18,
  "risk_tier": "High",
  "explanation": {
    "base_value_margin": -1.0205,
    "model_output_margin": 0.7725,
    "top_churn_drivers": [
      {"clean_feature": "Contract_Month-to-month", "feature_value": 1.0, "shap_value": 0.5421},
      {"clean_feature": "tenure", "feature_value": 4.0, "shap_value": 0.3982},
      {"clean_feature": "TechSupport_No", "feature_value": 1.0, "shap_value": 0.2814}
    ],
    "top_retention_drivers": [
      {"clean_feature": "PhoneService_Yes", "feature_value": 1.0, "shap_value": -0.0410}
    ]
  }
}
```

### 4.2 Batch Record Prediction

```python
batch_payload = [
    payload_customer_1,
    payload_customer_2,
    # ... up to thousands of records
]

batch_result = engine.predict_batch(batch_payload, error_handling="per_record")

print(f"Total: {batch_result.total_records}")
print(f"Successful: {batch_result.successful_predictions}")
print(f"Failed: {batch_result.failed_predictions}")
```

---

## 5. Automated Unit Test Verification

The inference system is comprehensively verified by 8 dedicated automated unit tests in [`tests/unit/test_inference.py`](file:///d:/Project/Predictive%20Analytics%20Platform/tests/unit/test_inference.py):

| Test Name | Test Target | Verification Outcome |
|---|---|---|
| `test_customer_input_record_valid` | Pydantic schema instantiation | PASSED |
| `test_customer_input_record_negative_tenure_raises` | Rejection of negative tenure | PASSED (`ValidationError`) |
| `test_customer_input_record_non_finite_charges_raises` | Rejection of NaN charges | PASSED (`ValidationError`) |
| `test_inference_engine_single_prediction` | Single prediction scoring & risk tiering | PASSED |
| `test_inference_engine_missing_column_raises` | Rejection when required predictor omitted | PASSED (`InferenceValidationError`) |
| `test_inference_engine_unexpected_column_raises` | Rejection on unknown extra features | PASSED (`InferenceValidationError`) |
| `test_inference_engine_batch_prediction_ordering` | Verification that output order == input order | PASSED |
| `test_inference_engine_batch_atomic_vs_per_record` | Comparison of atomic vs per-record modes | PASSED |

All tests execute cleanly as part of the 121-test suite in `pytest`.

---

## 6. Production Deployment Readiness Sign-Off

1. **Model Persistence**: Production artifact serialized to `artifacts/model_phase5_calibrated.joblib`.
2. **Schema Invariance**: Fully validated Pydantic models decouple raw external client inputs from internal scikit-learn numpy arrays.
3. **Reproducibility**: Global deterministic seeds and fixed preprocessors ensure identical scoring across repeated invocations.
4. **Readiness**: The inference engine is ready for deployment in automated batch jobs or real-time web services (Phase 6).
