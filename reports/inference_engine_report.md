# PredictLab — Phase 5: Production-Ready Inference Engine Report

## 1. Executive Summary

This report documents the architectural design, schema validation protocols, error handling mechanisms, reproducible latency benchmarks, and performance verification of the **Production-Ready Inference Engine** implemented in Phase 5 of PredictLab.

Transitioning machine learning models from offline cross-validation into production operations requires rigorous engineering safeguards. Real-world input payloads frequently contain missing attributes, malformed data types, non-finite values, and anomalous whitespace strings. In addition, operational systems require both real-time, low-latency single-customer scoring and high-throughput batch scoring with transactional error isolation.

Phase 5 delivers [`InferenceEngine`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/inference/engine.py#L27) and its companion Pydantic v2 schemas:
- **Strict Input Schema Validation**: Implemented via [`CustomerInputRecord`](file:///d:/Project/Predictive%20Analytics%20Platform/src/predictlab/inference/schemas.py#L70), validating all 19 raw predictors. Rejects invalid categorical levels, negative tenure, and non-finite charges.
- **Identifier Decoupling**: If an optional `customerID` is supplied in raw payloads, the engine validates its uniqueness, preserves it in batch response mappings, and strictly strips it prior to model transformation, ensuring identifiers never enter the feature matrix.
- **Zero Transformation Duplication**: Raw dictionaries or DataFrames are passed directly to the encapsulated scikit-learn pipeline (`preprocessor.transform()`), guaranteeing that inference transforms are 100% bit-for-bit identical to training transforms.
- **Dual Batch Error Handling Modes**: Supports both `atomic` mode (transactional validation where any single error rejects the batch) and `per_record` mode (graceful degradation where valid records are scored and invalid records return localized diagnostics).
- **Deterministic Calibrated Inference**: Loads the serialized Phase 5 calibrated artifact (`model_phase5_calibrated.joblib`), applying the cost-optimal decision threshold ($\tau^* = 0.19$) and contiguous risk tiering.
- **Integrated Local TreeSHAP Explanations**: Seamlessly generates local feature attributions on demand across the 5-fold ensemble, identifying top churn and retention drivers without external script dependencies.
- **Empirical Latency Benchmarking**: Rigorously benchmarked across single-record and batch workloads on local hardware, demonstrating high-throughput batch processing at **$0.096$ ms per record** ($~10,398$ rec/s).

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
└── Calibrated Classifier: 5-fold bagging ensemble computes P(Churn)
      │
      ▼
Operational Decision Rules & Tiering
├── Applies Decision Threshold (default: tau* = 0.19)
└── Assigns Operational Risk Tier (Low, Medium, High)
      │
      ▼
Optional Local Explainability (TreeShapExplainer)
└── Computes exact additive ensemble margin SHAP values & drivers
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
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0,
    "TotalCharges": 412.5,
}

# Execute prediction with local SHAP explanation
result = engine.predict_single(payload, include_explanation=True)

print(result.model_dump())
```

**Output Structure**:
```json
{
  "model_version": "5.0.0",
  "model_name": "Hist Gradient Boosting (Sigmoid Calibrated)",
  "calibration_method": "sigmoid",
  "estimated_probability": 0.6868,
  "predicted_class": 1,
  "predicted_label": "Churned",
  "applied_threshold": 0.19,
  "risk_tier": "High",
  "explanation": {
    "base_value_margin": -1.5438,
    "model_output_margin": 0.7876,
    "top_churn_drivers": [
      {"clean_feature": "Contract_Month-to-month", "feature_value": 1.0, "shap_value": 0.9279},
      {"clean_feature": "tenure", "feature_value": 5.0, "shap_value": 0.4033},
      {"clean_feature": "OnlineSecurity_No", "feature_value": 1.0, "shap_value": 0.2626},
      {"clean_feature": "InternetService_Fiber optic", "feature_value": 1.0, "shap_value": 0.2027}
    ],
    "top_retention_drivers": [
      {"clean_feature": "MultipleLines_No", "feature_value": 1.0, "shap_value": -0.1595},
      {"clean_feature": "Contract_One year", "feature_value": 0.0, "shap_value": -0.0274}
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

## 5. Inference Latency & Throughput Benchmark

To establish reproducible production performance characteristics, we benchmarked the inference engine across multiple operational workloads on the host platform using the dedicated benchmarking harness in [`scripts/benchmark_inference.py`](file:///d:/Project/Predictive%20Analytics%20Platform/scripts/benchmark_inference.py). Raw benchmark distributions are preserved in [`reports/experiments/inference_benchmarks.json`](file:///d:/Project/Predictive%20Analytics%20Platform/reports/experiments/inference_benchmarks.json).

### 5.1 Benchmark Hardware & Runtime Environment
- **CPU Model**: `13th Gen Intel(R) Core(TM) i5-13450HX` (10 cores [6 Performance cores up to 4.6 GHz, 4 Efficient cores up to 3.4 GHz], 16 logical threads)
- **Host Memory**: `15.69 GB` total physical RAM (`3.58 GB` available at test execution)
- **Operating System**: `Microsoft Windows 11 Enterprise` (Build 10.0.26300, 64-bit architecture)
- **Runtime Environment**: `Python 3.12.10` (64-bit)
- **Key Library Versions**: `scikit-learn 1.6.1`, `pydantic 2.13.4`, `pandas 2.3.1`, `numpy 2.3.2`, `psutil 7.0.0`
- **Model Artifact Evaluated**: `artifacts/model_phase5_calibrated.joblib` (5-fold ensemble of HistGradientBoosting pipelines with 5 Sigmoid calibrators)

### 5.2 Benchmarking Methodology & Measurement Boundaries
- **Timing Primitive**: `time.perf_counter()` (monotonic high-resolution timer with sub-microsecond precision).
- **Warm-Up Procedure**: 5 unmeasured warm-up iterations executed for each mode prior to recording measurements to prime Python bytecode compilation, OS thread pools, and NumPy memory buffers.
- **Repetition Counts**:
  - 50 repetitions for single-record request scoring (no SHAP).
  - 20 repetitions for single-record request scoring with 5-estimator TreeSHAP explanations.
  - 30 repetitions for batch scoring of 100 customer records.
  - 30 repetitions for batch scoring of 1,000 customer records.
- **Pipeline Measurement Scope**:
  - **Single-Record Mode**: Begins at dictionary ingestion; includes Pydantic schema validation (`CustomerInputRecord`), DataFrame assembly, ColumnTransformer preprocessing, 5-fold ensemble scoring (`predict_proba` averaging), risk tier assignment, and `SinglePredictionOutput` serialization.
  - **Single-Record Mode with SHAP**: Includes all single-record steps plus TreeSHAP margin attribution traversal across all 5 ensemble tree estimators, additivity checks, and `CustomerLocalExplanation` packaging.
  - **Batch Mode**: Includes batch validation, atomic/per-record checking, vectorized DataFrame preprocessing, 5-fold ensemble probability estimation, risk tier assignment, and `BatchPredictionOutput` packaging.
- **Statistical Aggregation**:
  - Median = 50th percentile of measured execution times.
  - P95 = 95th percentile; P99 = 99th percentile.
  - Throughput = `batch_size / (median_batch_latency_seconds)`.

### 5.3 Latency Results Across Execution Modes

The table below contrasts both **Total Batch Latency** and **Normalized Per-Record Latency**:

| Workload Mode | Batch Size | Repeats | Explanations Enabled | Total Latency (Median) | Total Latency (Mean) | Total Latency (P95) | Total Latency (P99) | Normalized Per-Record (Median) | Normalized Per-Record (Mean) | Throughput (records / sec) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Vectorized Batch** | 1,000 | 30 | No | **96.17 ms** | 101.71 ms | 122.60 ms | 140.00 ms | **0.0962 ms** ($96.2 \ \mu\text{s}$) | 0.1017 ms | **~10,398.5 rec/s** |
| **Vectorized Batch** | 100 | 30 | No | **52.69 ms** | 54.85 ms | 68.26 ms | 76.21 ms | **0.5269 ms** ($526.9 \ \mu\text{s}$) | 0.5485 ms | **~1,897.8 rec/s** |
| **Single Record** | 1 | 50 | No | **47.65 ms** | 49.02 ms | 59.07 ms | 67.23 ms | **47.65 ms** | 49.02 ms | **~21.0 req/s** |
| **Single Record** | 1 | 20 | **Yes (5-Tree SHAP)** | **118.69 ms** | 118.35 ms | 135.39 ms | 146.22 ms | **118.69 ms** | 118.35 ms | **~8.4 req/s** |

> [!IMPORTANT]
> **Batch Latency vs Request Latency Equivalence Warning**:
> Normalized batch latency ($0.0962$ ms per record) measures amortized execution speed when hundreds or thousands of records are processed concurrently via vectorized C extensions and multithreaded tree scoring.
> **Normalized batch latency must NOT be confused with single-record request latency** ($47.65$ ms). In single-record scoring, per-request overhead (Python dictionary unpacking, Pydantic field-by-field validation, DataFrame allocation, and single-row matrix transformation) represents over $90\%$ of total wall-clock time.

### 5.4 Performance Engineering Analysis
1. **Vectorized Scaling in Bulk Operations**: Batch scoring achieves over $10,000$ records per second on a single machine. The ColumnTransformer and underlying HistGradientBoosting C extensions scale efficiently, making nightly batch scoring of millions of subscribers feasible in under 2 minutes.
2. **Interactive Real-Time Service**: Single-record scoring completes in $\approx 47.7$ ms (P95 = $59.1$ ms). For real-time REST API endpoints or CRM customer-care lookups, response latency remains comfortably within standard interactive SLAs ($< 100$ ms).
3. **Local TreeSHAP Overhead**: Generating local SHAP explanations adds $\approx 71.0$ ms per record, bringing total response time to $\approx 118.7$ ms. Because `TreeShapExplainer` averages attributions across all 5 ensemble tree estimators, this cost scales linearly with ensemble size. SHAP generation is recommended on-demand for high-risk accounts rather than universally across bulk batches.

---

## 6. Automated Unit Test Verification

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

All tests execute cleanly as part of the 126-test suite in `pytest`.

---

## 7. Production Deployment Readiness Sign-Off

1. **Model Persistence**: Production artifact serialized to `artifacts/model_phase5_calibrated.joblib`.
2. **Schema Invariance**: Fully validated Pydantic models decouple raw external client inputs from internal scikit-learn numpy arrays.
3. **Reproducibility**: Global deterministic seeds and fixed preprocessors ensure identical scoring across repeated invocations.
4. **Empirically Benchmarked**: Vectorized batch throughput benchmarked at $>10,000$ records/sec; real-time single-record scoring under $50$ ms.
5. **Readiness**: The inference engine is ready for deployment in automated batch jobs or real-time web services (Phase 6).
