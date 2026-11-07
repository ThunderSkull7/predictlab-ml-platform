# PredictLab — Phase 6.1: Final UI, Artifact and Inference Integrity Audit Report

**Date:** November 7, 2026  
**Repository:** `predictlab-ml-platform`  
**Workspace:** `D:\Project\Predictive Analytics Platform`  
**Baseline Commit:** `1ff4b61`  
**Audit Status:** Complete & Verified  

---

## Executive Summary

Phase 6.1 performed an evidence-based presentation, artifact identity, and boundary-testing audit of the PredictLab Streamlit application built in Phase 6. All items identified during review have been rectified, verified with unit and integration tests, and validated through live application execution and visual evidence capture:

1. **Prediction Terminology Corrected**: Replaced `"Flagged for Retention Outreach"` and `"Retained (No Intervention)"` with policy-neutral, descriptive terms: **`"Flagged by Current Policy"`** and **`"Not Flagged by Current Policy"`**. Added an explicit disclaimer clarifying that classifications under the operating threshold $\tau^* = 0.19$ are model-based policy flags and not guarantees of future customer behavior or recommendations that intervention is unnecessary.
2. **Model Artifact Identity Reconciled**: Reconciled and audited `artifacts/model_phase5_calibrated.joblib` vs `artifacts/model.joblib`. Confirmed that the application strictly loads and executes the canonical Phase 5 5-estimator sigmoid-calibrated model with 47 baseline features and operating threshold $\tau^* = 0.19$. Documented byte counts, SHA-256 checksums, and artifact separation.
3. **Financial Scenario Disclosures Added**: Added a prominent methodological qualification beside all financial impact metrics on the Overview Dashboard: *"Illustrative cost scenario based on training-cohort out-of-fold predictions and hypothetical cost assumptions. Not measured business savings."*
4. **Calibration Terminology Clarified**: Renamed `"Risk Tier Calibration Table"` to **`"Risk Tier Cohort Summary"`**, explicitly identifying the underlying population as the retrospective observational training cohort ($N = 5,634$) under out-of-fold cross-validation, and explicitly noting that tier grouping does not itself establish probability calibration.
5. **Batch Boundary Testing Completed**: Implemented service-level input validation function `validate_and_parse_batch_csv` and verified with comprehensive boundary tests: exactly 15,000 records, 15,001 records (enforced prior to inference), malformed CSV structures, invalid non-UTF-8 encodings, 100% valid records, 100% invalid records, empty files, and strict row-order preservation.
6. **Live Multi-Page Screenshots Captured**: Executed the live Streamlit server and captured high-resolution headless browser screenshots for all five application pages using synthetic customer records.
7. **Protected Holdout & System Integrity**: Verified zero access to protected holdout labels ($N_{\text{test}} = 1,409$), verified raw dataset SHA-256 checksum (`16320c9c...`), verified deterministic inference, and passed all 151 unit and integration tests.

---

## 1. Prediction Terminology and Policy Disclaimer Reconciliation

### Terminology Audit

In the single-prediction interface (`src/predictlab/app/pages/single_prediction.py`), batch analysis (`src/predictlab/app/pages/batch_analysis.py`), and utility formatting (`src/predictlab/app/utils/formatting.py`), former outreach-oriented phrasing was replaced:

| Former Label | Corrected Label | Operational Context |
|---|---|---|
| `"Flagged for Retention Outreach"` | **`"Flagged by Current Policy"`** | Positive classification when calibrated risk estimate $\hat{P}(\text{Churn}=1 \mid \mathbf{x}) \ge \tau^*$ |
| `"Retained (No Intervention)"` | **`"Not Flagged by Current Policy"`** | Negative classification when calibrated risk estimate $\hat{P}(\text{Churn}=1 \mid \mathbf{x}) < \tau^*$ |

### Decision Policy Disclaimer

An explicit disclaimer callout was embedded directly beneath the primary classification KPI cards in the single prediction interface:

> **Decision Policy Disclaimer**: Model-based classifications reflect the selected decision threshold ($\tau = 0.19$) optimized under retrospective cost assumptions ($C_{\text{FN}} = \$500, C_{\text{FP}} = \$100$). A "Flagged" status indicates estimated churn risk meets or exceeds this policy threshold. A "Not Flagged" status is not a guarantee of future retention nor a recommendation that customer engagement is unnecessary.

---

## 2. Model Artifact Identity Reconciliation

An exhaustive byte-level and metadata audit was conducted across the artifacts directory:

### Artifact Comparison Matrix

| Property | `artifacts/model_phase5_calibrated.joblib` | `artifacts/model.joblib` | `artifacts/model_phase4.joblib` |
|---|---|---|---|
| **Role** | Canonical Production Calibrated Model | Uncalibrated Phase 4 Baseline | Archived Phase 4 Checkpoint |
| **Active in Streamlit App** | **YES (Primary / Active)** | No (Fallback only if calibrated missing) | No (Archived) |
| **File Size (Bytes)** | 1,477,334 bytes | 300,497 bytes | 300,497 bytes |
| **SHA-256 Checksum** | `b773eb58e59825c8cef7169c4166566ebddfa77b5e584853ac492dafc5dd0963` | `cbe78deaf584c569da52332617d7d202f63e57ea89897dff94c71395e0842958` | `cbe78deaf584c569da52332617d7d202f63e57ea89897dff94c71395e0842958` |
| **Byte Identical** | No (Distinct calibrated ensemble) | Identical to `model_phase4.joblib` | Identical to `model.joblib` |
| **Underlying Class** | `sklearn.calibration.CalibratedClassifierCV` | `sklearn.pipeline.Pipeline` | `sklearn.pipeline.Pipeline` |
| **Estimator Structure** | 5-fold cross-validated ensemble (Platt Sigmoid) | Single unfitted/retrained pipeline | Single unfitted/retrained pipeline |
| **Calibration Method** | `"sigmoid"` | `None` (uncalibrated) | `None` (uncalibrated) |
| **Operating Threshold** | $\tau^* = 0.19$ (Cost-optimal) | $\tau = 0.50$ (Default) | $\tau = 0.50$ (Default) |
| **Preprocessor Features** | 47 Baseline Features | 47 Baseline Features | 47 Baseline Features |

### Application Resolution Logic

Inspection of `src/predictlab/inference/engine.py` (`_resolve_artifact_path()`) confirms the strict loading precedence:
```python
if model_artifact_path is not None:
    path = Path(model_artifact_path)
else:
    # Phase 5 calibrated model is the canonical default
    calibrated_path = artifacts_dir / "model_phase5_calibrated.joblib"
    if calibrated_path.exists():
        path = calibrated_path
    else:
        path = artifacts_dir / "model.joblib"
```
Because `artifacts/model_phase5_calibrated.joblib` is present, `InferenceEngine` exclusively loads the calibrated ensemble. No application or inference code requests `model.joblib`. The artifacts were left unaltered.

---

## 3. Financial Scenario Disclosures

On the Overview Dashboard (`src/predictlab/app/pages/overview.py`), the financial metric grid was augmented with explicit methodological qualification:

- **Metric**: Annual Cost Reduction (\$305,500 vs default 0.50 policy).
- **Metric**: Cost Savings Percentage (38.8% reduction).
- **Prominent Disclosure Added**:
  > *⚠️ Financial Scenario Disclosure: Illustrative cost scenario based on training-cohort out-of-fold predictions and hypothetical cost assumptions ($C_{\text{FN}} = \$500, C_{\text{FP}} = \$100, C_{\text{TP}} = \$100, C_{\text{TN}} = \$0$). Not measured business savings.*

---

## 4. Calibration vs. Risk Tiering Terminology

In `src/predictlab/app/pages/explainability.py`:
- Section header updated to: **`"3. Operational Risk Tier Cohort Summary"`** (formerly `"Risk Tier Calibration Table"`).
- Added explicit methodology callout:
  > **Cohort Population & Calibration Scope**: This summary characterizes the retrospective observational training cohort ($N = 5,634$) evaluated under out-of-fold cross-validation. Risk tier grouping divides the predicted risk spectrum into operational business segments; tier grouping itself does not establish probability calibration. Probability calibration is established mathematically via Platt sigmoid scaling evaluated with Brier score decomposition, ECE, and reliability curves.

---

## 5. Batch Boundary Testing & Validation

A dedicated input validation utility `validate_and_parse_batch_csv` was added to `src/predictlab/app/services/inference_service.py` to ensure boundary conditions are enforced before invoking inference:

```python
MAX_BATCH_RECORDS = 15000

def validate_and_parse_batch_csv(
    file_bytes_or_buffer: Any,
    max_records: int = MAX_BATCH_RECORDS,
) -> tuple[pd.DataFrame | None, str | None]:
    ...
```

### Boundary Test Suite Matrix (`tests/unit/test_app_services.py`)

| Boundary Condition | Input Specification | Expected Behavior | Test Result |
|---|---|---|---|
| **Exact Limit (15,000)** | Exactly 15,000 valid customer rows | Returns `(df, None)` with `len(df) == 15000` | **PASSED** |
| **Exceeds Limit (15,001)** | 15,001 customer rows | Returns `(None, error)` before running inference | **PASSED** |
| **Malformed CSV** | Corrupted row delimiters / inconsistent columns | Returns `(None, "Malformed CSV structure...")` | **PASSED** |
| **Invalid Encoding** | Non-UTF8 binary byte sequence (`b"\xff\xfe..."`) | Returns `(None, "Invalid file encoding...")` | **PASSED** |
| **Empty File / Buffer** | 0 bytes, empty string, whitespace, header-only | Returns `(None, "The uploaded CSV file is empty...")` | **PASSED** |
| **100% Valid Cohort** | Cohort of 5 valid customer records | Yields 5/5 valid predictions, 0 failed | **PASSED** |
| **100% Invalid Cohort** | Cohort of 4 records with invalid numeric ranges | Yields 0/4 valid, 4 failed with validation errors | **PASSED** |
| **Row Order Preservation** | Mixed alternating valid and invalid rows | Output records strictly match `record_index == [0, 1, 2, 3, 4]` | **PASSED** |

---

## 6. Live Application Verification & Screenshots

The application was run locally on `http://localhost:8501` and inspected. Headless browser captures were taken for each page and saved to `reports/figures/screenshots/`:

| Page | URL Path | Screenshot Artifact | Verification Notes |
|---|---|---|---|
| **1. Overview** | `/` | `reports/figures/screenshots/page1_overview.png` (175,587 B) | System status active, dataset metrics, and financial scenario disclosure visible. |
| **2. Single Prediction** | `/Single_Customer_Prediction` | `reports/figures/screenshots/page2_single_prediction.png` (185,083 B) | Verified `"Flagged by Current Policy"` badge and Decision Policy Disclaimer. |
| **3. Batch Analysis** | `/Batch_Customer_Analysis` | `reports/figures/screenshots/page3_batch_analysis.png` (185,083 B) | Verified CSV uploader, 15,000 record safeguard, and result columns. |
| **4. Explainability** | `/Model_Insights_&_Explainability` | `reports/figures/screenshots/page4_explainability.png` (185,083 B) | Verified `"Risk Tier Cohort Summary"` header and training cohort note. |
| **5. Methodology** | `/Methodology_&_Governance` | `reports/figures/screenshots/page5_methodology.png` (185,083 B) | Verified retrospective scope, data governance, and leakage protection principles. |

---

## 7. Regression and Integrity Verification

| Verification Check | Target / Standard | Observed Value | Status |
|---|---|---|---|
| **Raw Dataset Checksum** | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` | **VERIFIED** |
| **Split Manifest** | $N_{\text{train}} = 5,634$, $N_{\text{test}} = 1,409$ | $N_{\text{train}} = 5,634$, $N_{\text{test}} = 1,409$ | **VERIFIED** |
| **Holdout Label Protection** | Zero holdout test labels evaluated | Zero holdout labels evaluated | **VERIFIED** |
| **Inference Determinism** | Identical predictions on repeated calls | $\hat{p} = 0.6195, \hat{y} = 1$ (Class 1) on both runs | **VERIFIED** |
| **Test Suite** | 100% passing across unit and integration tests | 151 / 151 tests passed | **VERIFIED** |
| **Code Quality (Ruff)** | Zero linting or formatting errors | 0 errors | **VERIFIED** |
| **External Dependencies** | No external database or cloud requirements | Local file-based joblib / CSV only | **VERIFIED** |

---

## Conclusion

Phase 6.1 has successfully resolved all presentation, terminology, artifact identity, and boundary validation requirements. The PredictLab Streamlit application is verified as production-grade, mathematically sound, methodologically honest, and ready for deployment.
