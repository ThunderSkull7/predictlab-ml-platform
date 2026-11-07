# PredictLab Data Registry & Provenance Audit

This directory contains data assets used by PredictLab.

## 1. Directory Structure

```text
data/
├── README.md               # Data documentation, provenance audit, and verification registry
├── raw/                    # Immutable raw source data
│   └── Telco-Customer-Churn.csv
└── processed/              # Cleaned, transformed feature sets (Phase 3+)
    └── .gitkeep
```

## 2. Canonical Dataset: IBM Telco Customer Churn

### Upstream Repository & Provenance Audit
- **Host Repository**: [`IBM/telco-customer-churn-on-icp4d`](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Direct Source URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Retrieval Date**: `2026-11-07`
- **Upstream Repository License**: Apache License, Version 2.0 (governing code patterns and repository software).

### Data Licensing & Provenance Assessment
- **Software vs. Data Scope**: The upstream repository provides an Apache 2.0 license file and states that the "code pattern is licensed under the Apache Software License, Version 2." However, the repository contains no dedicated open-data license (such as CDLA, CC0, CC-BY, or ODbL) explicitly defining dataset redistribution terms.
- **Publisher vs. Host Status**: The repository merely hosts `data/Telco-Customer-Churn.csv` as sample demonstration data for an IBM Cloud Pak for Data tutorial. It does not establish IBM as the primary publisher of real customer records nor does it provide explicit commercial data redistribution covenants.
- **Licensing Conclusion**: Third-party dataset redistribution rights cannot be conclusively established from upstream repository metadata alone. PredictLab does not claim unrestricted redistribution rights for the dataset.
- **Policy**: PredictLab's own code is licensed under the MIT License. The dataset is managed as an external sample asset. A reproducible download procedure is provided below so any developer can fetch and verify the canonical CSV directly from the upstream IBM URL.

### Integrity Checksum & Dimensions
- **File Name**: `Telco-Customer-Churn.csv`
- **File Location**: `data/raw/Telco-Customer-Churn.csv`
- **File Size**: `970,457 bytes` (~948 KB)
- **SHA-256 Checksum**: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Row Count**: 7,043 customer accounts
- **Column Count**: 21 attributes (1 identifier, 19 predictive candidate features, 1 ground-truth target)

### Target Distribution
- **Target Feature**: `Churn`
- **Negative Class (`No`)**: 5,174 instances (73.46%) — retained accounts
- **Positive Class (`Yes`)**: 1,869 instances (26.54%) — accounts that discontinued service
- **Class Balance**: Moderate imbalance (~2.77:1 ratio). Evaluation must prioritize metrics such as PR-AUC, Brier score, and probability calibration rather than naive accuracy.

## 3. Pending Preprocessing Decision: `TotalCharges` Whitespace

- **Observation**: Exactly 11 rows in `Telco-Customer-Churn.csv` contain whitespace strings (`" "`) in the `TotalCharges` column.
- **Association**: All 11 records have `tenure == 0` and `Churn == "No"`. These represent brand-new subscribers who enrolled but have not yet completed their initial monthly billing cycle.
- **Integrity Rule**: The raw CSV file in `data/raw/` remains **completely unmodified**.
- **Pending Decision**: The proposal to convert whitespace to `0.0` (reflecting zero historical charges billed at enrollment) is recorded as a pending preprocessing decision to be investigated empirically in Phase 2 and implemented in Phase 3 pipelines.

## 4. Reproducible Download & Verification Procedure

To retrieve or independently verify the raw dataset from upstream:

```bash
python -c "
import urllib.request, hashlib
from pathlib import Path

url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
dest = Path('data/raw/Telco-Customer-Churn.csv')
dest.parent.mkdir(parents=True, exist_ok=True)

print(f'Fetching from {url}...')
with urllib.request.urlopen(url) as resp:
    data = resp.read()

sha256 = hashlib.sha256(data).hexdigest()
expected_sha = '16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91'
assert sha256 == expected_sha, f'Checksum mismatch: {sha256} vs {expected_sha}'

dest.write_bytes(data)
print(f'Verified and saved to {dest} ({len(data)} bytes, SHA-256: {sha256}).')
"
```

## 5. Storage & Versioning Policy
- `data/raw/`: Read-only, immutable raw data.
- `data/processed/`: Machine-generated intermediate artifacts (created in later preprocessing phases). Excluded from version control via `.gitignore`.
