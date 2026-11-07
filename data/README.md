# PredictLab Data Registry

This directory contains data assets used by PredictLab.

## 1. Directory Structure

```text
data/
├── README.md               # Data documentation, provenance, and verification registry
├── raw/                    # Immutable source data (raw files, as retrieved)
│   └── Telco-Customer-Churn.csv
└── processed/              # Cleaned, transformed, or partitioned feature stores (future phases)
    └── .gitkeep
```

## 2. Canonical Dataset: IBM Telco Customer Churn

### Provenance & Authoritative Source
- **Origin**: IBM Watson Analytics Sample Data Sets / IBM Business Analytics Community.
- **Authoritative Repository**: `IBM/telco-customer-churn-on-icp4d` (IBM Developer Pattern repository).
- **Direct Source URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Retrieval Date**: 2026-11-07
- **License**: Apache License, Version 2.0 (Permits commercial use, modification, and redistribution with copyright notice and attribution).

### Integrity Checksum & Dimensions
- **File Name**: `Telco-Customer-Churn.csv`
- **File Location**: `data/raw/Telco-Customer-Churn.csv`
- **File Size**: `970,457 bytes` (~948 KB)
- **SHA-256 Checksum**: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Row Count**: 7,043 customer accounts
- **Column Count**: 21 attributes (1 identifier, 19 predictive candidate features, 1 ground-truth target)

### Target Distribution
- **Target Feature**: `Churn`
- **Positive Class (`Yes`)**: 1,869 instances (26.54%)
- **Negative Class (`No`)**: 5,174 instances (73.46%)
- **Class Balance**: Moderate imbalance (~2.77:1 ratio). Evaluation must prioritize metrics such as PR-AUC, ROC-AUC, Brier score, and calibrated probabilities rather than naive accuracy.

## 3. Reproducible Download & Verification

To verify or independently retrieve the raw dataset into `data/raw/`, run:

```bash
python -c "
import urllib.request, hashlib
from pathlib import Path

url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
dest = Path('data/raw/Telco-Customer-Churn.csv')
dest.parent.mkdir(parents=True, exist_ok=True)

print(f'Downloading from {url}...')
with urllib.request.urlopen(url) as resp:
    data = resp.read()

sha256 = hashlib.sha256(data).hexdigest()
expected_sha = '16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91'
assert sha256 == expected_sha, f'Checksum mismatch: {sha256} vs {expected_sha}'

dest.write_bytes(data)
print(f'Verified and saved to {dest} ({len(data)} bytes).')
"
```

## 4. Storage & Versioning Policy
- `data/raw/`: Read-only, immutable source data. Never modify these files in-place.
- `data/processed/`: Machine-generated intermediate artifacts (created in later preprocessing phases). Excluded from version control via `.gitignore`.
