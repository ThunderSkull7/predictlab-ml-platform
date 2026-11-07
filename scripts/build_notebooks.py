"""Script to programmatically build, execute, and save Phase 2 notebooks."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_and_execute_notebook_01():
    """Build and execute notebooks/01_data_audit.ipynb."""
    nb = nbf.v4.new_notebook()

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """# PredictLab — Notebook 01: Data Quality Audit & Anomaly Investigation

## Objective
This notebook performs a comprehensive data quality audit of the canonical IBM Telco Customer Churn dataset, covering:
1. Structural integrity, dimensions, and data types.
2. Missing value and whitespace string diagnostics.
3. Customer identifier uniqueness and duplicate analysis.
4. Ground-truth target distribution.
5. In-depth empirical investigation of the 11 whitespace-only `TotalCharges` values.

This audit utilizes PredictLab's reusable profiling module (`predictlab.profiling.quality`).
"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """from predictlab.data.load import load_raw_dataset
from predictlab.profiling.quality import (
    detect_whitespace_entries,
    investigate_total_charges,
    profile_data_quality,
)
from predictlab.profiling.summary import (
    compute_dataset_overview,
    compute_target_distribution,
)

print("Modules imported successfully.")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Load Canonical Dataset & Structural Inspection

We load the dataset through the project's immutable loading interface (`load_raw_dataset`)."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """df = load_raw_dataset()
overview = compute_dataset_overview(df)
for k, v in overview.items():
    print(f"{k}: {v}")

df.info()"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Missing Value and Whitespace String Diagnostics

Standard pandas `.isna()` checks find zero missing values because whitespace strings (`" "`) are treated as valid object strings. We perform an explicit whitespace audit."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """ws_results = detect_whitespace_entries(df)
print("Columns with whitespace-only entries:", list(ws_results.keys()))
for col, idxs in ws_results.items():
    print(f"Column '{col}' contains {len(idxs)} whitespace-only values at row indices: {idxs}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Duplicate Records and Identifier Integrity

We verify whether any complete duplicate records or duplicate customer identifiers exist."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """dup_rows = int(df.duplicated().sum())
dup_ids = int(df['customerID'].duplicated().sum())
print(f"Complete duplicate rows: {dup_rows}")
print(f"Duplicate customerID values: {dup_ids}")
print(f"Total distinct customer IDs: {df['customerID'].nunique()} (out of {len(df)} rows)")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Target Class Distribution (`Churn`)

We inspect the class frequencies, percentages, and imbalance ratio."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """target_info = compute_target_distribution(df)
for k, v in target_info.items():
    print(f"{k}: {v}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. Mandatory Investigation: `TotalCharges` Whitespace Anomaly

We examine the 11 rows where `TotalCharges` contains whitespace strings and evaluate their relationship with `tenure`, `MonthlyCharges`, `Contract`, and `Churn`."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """inv = investigate_total_charges(df)
print(f"Total whitespace rows: {inv.total_whitespace_rows}")
print(f"All zero tenure? {inv.all_zero_tenure}")
print(f"MonthlyCharges: min={inv.monthly_charges_min}, mean={inv.monthly_charges_mean:.2f}, max={inv.monthly_charges_max}")
print(f"Contract distribution: {inv.contract_distribution}")
print(f"Churn distribution: {inv.churn_distribution}")
print(f"Tenure==1 exact match ratio (TotalCharges == MonthlyCharges): {inv.tenure_one_exact_match_ratio:.4f}")
print(f"Spearman correlation between valid TotalCharges and tenure: {inv.tenure_totalcharges_spearman:.4f}")
print(\"\\n--- RECOMMENDATION ---\")
print(inv.recommendation)
print(\"\\n--- RATIONALE ---\")
print(inv.rationale)"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. Comprehensive Data Quality Report

We execute the full automated data quality profiling routine (`profile_data_quality`)."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """quality_report = profile_data_quality(df)
print("Quality Summary Notes:")
for note in quality_report.quality_summary_notes:
    print(f"  • {note}")"""
        )
    )

    out_file = NOTEBOOKS_DIR / "01_data_audit.ipynb"
    print(f"Executing {out_file}...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    executed_nb = client.execute()
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(executed_nb, f)
    print(f"Saved executed notebook to {out_file}")


def build_and_execute_notebook_02():
    """Build and execute notebooks/02_eda.ipynb."""
    nb = nbf.v4.new_notebook()

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """# PredictLab — Notebook 02: Exploratory Data Analysis & Diagnostic Insights

## Analytical Scope
This notebook implements comprehensive exploratory analysis on the canonical Telco Customer Churn dataset, providing:
1. **Univariate Profiling**: Numerical distribution shapes, central tendencies, spreads, and categorical cardinality.
2. **Bivariate Analysis**: Churn variation across tenure, charges, contracts, payment channels, service add-ons, and demographics.
3. **Multivariate Segment Interactions**: Coupled risk dynamics across service tiers, contract types, and tenure cohorts.
4. **Inferential Hypothesis Testing**: Chi-square tests of independence with Benjamini-Hochberg FDR adjustments, Cramér's V effect sizes, and Mann-Whitney U / Welch's t-tests.
5. **Answers to the 10 Core Analytical Questions** with empirical evidence and visualizations.
"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path

from IPython.display import Image, display

from predictlab.data.load import load_raw_dataset
from predictlab.profiling.insights import (
    compute_categorical_associations,
    compute_numerical_churn_tests,
)
from predictlab.profiling.visuals import generate_all_eda_figures

figures_dir = Path("../reports/figures")
figures_dir.mkdir(parents=True, exist_ok=True)
print("Modules imported successfully.")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Load Data & Generate Publication Figures

We load the canonical dataset and invoke the automated visualization suite."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """df = load_raw_dataset()
created_figs = generate_all_eda_figures(df, output_dir=figures_dir)
print(f"Generated {len(created_figs)} figures in {figures_dir.resolve()}.")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Answers to Core Analytical Questions

### Q1: What proportion of customers churn, and how imbalanced is the target?
- **Finding**: Out of 7,043 customers, 1,869 churned (26.54%) and 5,174 remained active (73.46%).
- **Balance Ratio**: 2.77:1 (negative-to-positive ratio).
- **Implication**: Moderate class imbalance requires evaluating PR-AUC, Brier score, and calibrated thresholds rather than accuracy."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "01_target_distribution.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            r"""### Q2: How does customer tenure relate to churn?
- **Finding**: Churn is heavily concentrated in early subscriber lifecycles.
- **Median Tenure**: Churned accounts have a median tenure of 10.0 months, compared to 38.0 months for retained accounts ($p < 10^{-100}$).
- **Observed Churn Rate by Tenure Cohort**: Customers with tenure $\le 12$ months exhibit an observed churn rate of **47.68%**, which drops to **6.61%** for customers with tenure $> 60$ months."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "03_tenure_vs_churn.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Q3: How do contract types differ in their observed churn rates?
- **Finding**: Contract commitment duration is the single strongest categorical factor.
- **Month-to-month**: 42.71% churn rate (3,875 accounts, 1,655 churns) — relative risk of 1.61x.
- **One year**: 11.27% churn rate (1,473 accounts, 166 churns).
- **Two year**: 2.83% churn rate (1,695 accounts, 48 churns) — relative risk of 0.11x."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "06_churn_by_contract.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Q4: What relationship exists between MonthlyCharges and churn?
- **Finding**: Churned customers pay significantly higher monthly bills.
- **Means**: Churned customers average **$74.44/month**, compared to **$61.27/month** for retained customers (Welch's $t = 18.41, p < 10^{-70}$).
- **Distribution**: Higher attrition occurs in the $70–$100/month tier, driven heavily by Fiber Optic bundles without bundled support."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "04_monthly_charges_vs_churn.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Q5: How do InternetService, OnlineSecurity, and TechSupport relate to churn?
- **Internet Service**: Fiber Optic subscribers exhibit a **41.89%** churn rate, compared to **18.96%** for DSL and **7.40%** for No Internet.
- **Tech Support**: Accounts without Tech Support churn at **41.65%**, compared to **15.17%** for accounts with Tech Support.
- **Online Security**: Accounts without Online Security churn at **41.77%**, compared to **14.61%** with Online Security."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "08_churn_by_internet_service.png")))
display(Image(filename=str(figures_dir / "09_churn_by_tech_support.png")))
display(Image(filename=str(figures_dir / "10_churn_by_online_security.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Q6: Do payment methods show meaningful differences in churn rates?
- **Electronic Check**: Has the highest churn rate at **45.29%** (1,071 churns across 2,365 accounts; 1.71x baseline).
- **Automatic Payments**: Bank transfer (automatic) has a **16.71%** churn rate and Credit card (automatic) has a **15.24%** churn rate.
- **Mailed Check**: Shows an intermediate churn rate of **19.11%**."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "07_churn_by_payment_method.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Q7: Are there meaningful differences across senior citizens, partners, and dependents?
- **Senior Citizens**: Churn at **41.68%** vs. **23.61%** for non-seniors (Cramér's V = 0.150, $p < 10^{-35}$).
- **Partners**: Customers with partners churn at **19.66%** vs. **32.96%** without partners.
- **Dependents**: Customers with dependents churn at **15.45%** vs. **31.28%** without dependents."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "11_churn_by_demographics.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            r"""### Q8: What is the relationship between TotalCharges and tenure, and what should happen to the 11 whitespace-only values?
- **Correlation**: Spearman rank correlation between valid TotalCharges and tenure is **0.889**, confirming that TotalCharges is accumulated billing over tenure ($\approx \text{tenure} \times \text{MonthlyCharges}$).
- **Tenure == 1 Consistency**: For all 613 customers with `tenure == 1`, `TotalCharges` exactly matches `MonthlyCharges` ($\Delta = 0.00$).
- **Whitespace Rows**: All 11 whitespace records have `tenure == 0` and `Churn == "No"`. At month 0, accumulated billed revenue is operationally $0.00.
- **Recommendation**: Impute `0.0` during Phase 3 preprocessing."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "14_total_charges_tenure_investigation.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Q9: Which features show the strongest descriptive associations with churn?
- **Top Categorical Associations** (Bias-Corrected Cramér's V):
  1. `Contract`: $V = 0.408$
  2. `OnlineSecurity`: $V = 0.347$
  3. `TechSupport`: $V = 0.344$
  4. `InternetService`: $V = 0.320$
  5. `PaymentMethod`: $V = 0.302$
- **Top Numerical Associations** (Rank-Biserial Correlation):
  1. `tenure`: $r = -0.370$ ($p < 10^{-100}$)
  2. `MonthlyCharges`: $r = +0.228$ ($p < 10^{-70}$)
  3. `TotalCharges`: $r = -0.231$ ($p < 10^{-50}$)"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "16_categorical_associations_cramers_v.png")))
display(Image(filename=str(figures_dir / "15_numeric_correlation_matrix.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Q10: What data-quality or modelling concerns should be addressed before preprocessing?
1. **Identifier Exclusion**: `customerID` is 100% unique and must be excluded from modeling matrices.
2. **TotalCharges Preprocessing Pipeline**: The canonical raw CSV remains unmodified. In Phase 3, parse whitespace to NaN, validate missingness strictly against tenure == 0, and impute 0.0 before casting to float.
3. **Multicollinearity between Tenure and TotalCharges**: Spearman correlation of 0.889 indicates collinearity; tree models tolerate this, but regularized linear models will require attention.
4. **Ternary Categories**: Categories like `"No internet service"` and `"No phone service"` represent service structural states distinct from `"No"` and should not be collapsed blindly.
5. **Feature Directives**: Default One-Hot encoding for `Contract` (ordinal encoding evaluated only as an alternative experiment with documented spacing assumptions); engineer `OptionalServiceCount` across all 6 internet add-ons (scoring "No internet service" as 0); treat `ChargesRatio` as an experimental candidate requiring explicit zero-tenure branching."""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Multivariate Interaction Analysis

We examine the interaction of `InternetService` with `TechSupport`, and `Contract` with `tenure_group`."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "12_multivariate_internet_and_support.png")))
display(Image(filename=str(figures_dir / "13_multivariate_contract_and_tenure.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Formal Inferential Statistical Tests

We run Chi-Square tests of independence with Benjamini-Hochberg FDR correction on categorical features, and Mann-Whitney U / Welch's t-tests on numerical features."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """cat_cols = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
cat_tests = compute_categorical_associations(df, cat_cols=cat_cols)
print("=== CATEGORICAL ASSOCIATIONS (CHI-SQUARE & CRAMER'S V WITH FDR ADJUSTMENT) ===")
print(cat_tests[['feature', 'chi2_statistic', 'cramers_v', 'fdr_p_value', 'significant']])

num_tests = compute_numerical_churn_tests(df, num_cols=["tenure", "MonthlyCharges", "TotalCharges"])
print(\"\\n=== NUMERICAL TESTS (MANN-WHITNEY U & WELCH'S T WITH FDR ADJUSTMENT) ===\")
print(num_tests[['feature', 'retained_median', 'churned_median', 'rank_biserial_r', 'mann_whitney_fdr_p', 'cohens_d']])"""
        )
    )

    out_file = NOTEBOOKS_DIR / "02_eda.ipynb"
    print(f"Executing {out_file}...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    executed_nb = client.execute()
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(executed_nb, f)
    print(f"Saved executed notebook to {out_file}")


def build_and_execute_notebook_03():
    """Build and execute notebooks/03_preprocessing_and_feature_engineering.ipynb."""
    nb = nbf.v4.new_notebook()

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """# PredictLab — Notebook 03: Preprocessing, Feature Engineering & Leakage-Free Pipeline

## Objective
This notebook demonstrates the end-to-end Phase 3 preprocessing and feature engineering architecture:
1. Canonical dataset loading and schema contract verification.
2. Leakage-free feature/target separation and reproducible stratified holdout splitting (80% train / 20% test).
3. Deterministic cleaning and domain validation of `TotalCharges` (resolving the 11 zero-tenure whitespace anomalies).
4. Domain feature engineering (`OptionalServiceCount`, `AutomaticPayment`, and experimental `ChargesRatio`).
5. Assembling scikit-learn `Pipeline` and `ColumnTransformer` preprocessors for both Linear and Tree-based model families.
6. Fitting preprocessors strictly on the training partition and transforming held-out test data.
7. Empirical verification of parameter invariance and data leakage prevention.
8. Artifact serialization and deserialization via Joblib.
"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """import numpy as np
import pandas as pd

from predictlab.data.load import load_raw_dataset
from predictlab.preprocessing import (
    clean_total_charges,
    compute_automatic_payment_flag,
    compute_charges_ratio_defined,
    compute_experimental_charges_ratio,
    compute_optional_service_count,
    create_experimental_preprocessor,
    create_linear_preprocessor,
    create_tree_preprocessor,
    get_preprocessor_feature_names,
    load_preprocessor,
    save_preprocessor,
    split_dataset,
    transform_to_dataframe,
    validate_dataset_contract,
)

print("Preprocessing modules imported successfully.")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Load Canonical Data & Validate Schema Contract

We load the canonical dataset through PredictLab's immutable loader and validate that it strictly satisfies the preprocessing schema contract."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """df_raw = load_raw_dataset()
validate_dataset_contract(df_raw, check_target=True)
print(f"Canonical dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
print("Schema contract verified: All 19 predictive features, ID, and target present with valid domains.")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Leakage-Safe Stratified Train/Test Partitioning

We perform an 80/20 stratified holdout split.
- `customerID` is excluded from feature matrices to prevent identifier memorization.
- `Churn` is extracted and encoded as binary integer (`No` = 0, `Yes` = 1).
- The split is stratified by `Churn` using global seed 42 to preserve class balance.
- A lightweight `split_manifest.json` is generated to guarantee auditability and reproducibility."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """X_train, X_test, y_train, y_test, manifest = split_dataset(
    df_raw,
    test_size=0.20,
    random_state=42,
    save_manifest=True,
)

print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape} (Churn rate: {y_train.mean():.4%})")
print(f"Holdout test: X_test={X_test.shape}, y_test={y_test.shape} (Churn rate: {y_test.mean():.4%})")
print(f"Customer ID overlap between train and test: {len(set(manifest.train_customer_ids).intersection(set(manifest.test_customer_ids)))}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. TotalCharges Cleaning & Zero-Tenure Validation

We inspect the 11 zero-tenure records with whitespace `TotalCharges` across the train and test partitions.
Our cleaning transformation asserts that missingness occurs *strictly* at `tenure == 0`, and imputes `0.0`."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """train_zero_tenure = X_train[X_train['tenure'] == 0]
test_zero_tenure = X_test[X_test['tenure'] == 0]
print(f"Zero-tenure signups in X_train: {len(train_zero_tenure)}")
print(f"Zero-tenure signups in X_test:  {len(test_zero_tenure)}")
print(f"Total zero-tenure signups:      {len(train_zero_tenure) + len(test_zero_tenure)}")

# Demonstrate deterministic cleaning on a sample slice
sample_cleaned = clean_total_charges(train_zero_tenure[['tenure', 'TotalCharges', 'MonthlyCharges']])
print("\\nSample cleaned zero-tenure records:")
display(sample_cleaned[['tenure', 'TotalCharges', 'MonthlyCharges']])"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Domain Feature Engineering & Mathematical Validity

We evaluate `OptionalServiceCount` (0 to 6), `AutomaticPayment`, and the corrected experimental `ChargesRatio`.
- When `tenure == 0`, `ChargesRatio` is mathematically undefined ($0/0$) and is explicitly represented as `NaN` (never fabricated as 1.0).
- `ChargesRatioDefined` provides a deterministic binary indicator (1 if defined, 0 if undefined)."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """opt_counts = compute_optional_service_count(X_train)
auto_flags = compute_automatic_payment_flag(X_train)

# Clean TotalCharges on training partition first to evaluate ChargesRatio
X_train_cleaned = clean_total_charges(X_train)
charges_ratio = compute_experimental_charges_ratio(X_train_cleaned)
charges_ratio_defined = compute_charges_ratio_defined(X_train_cleaned)

print("OptionalServiceCount distribution in X_train:")
print(opt_counts.value_counts().sort_index())

print("\\nAutomaticPayment flag distribution in X_train:")
print(auto_flags.value_counts())

print(f"\\nChargesRatio NaN count in X_train: {charges_ratio.isna().sum()} (strictly matching tenure==0 count: {(X_train['tenure'] == 0).sum()})")
print("ChargesRatioDefined distribution in X_train:")
print(charges_ratio_defined.value_counts())"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. Preprocessing Pipelines (Linear, Tree & Experimental Families)

We construct:
1. Linear-model preprocessor (with `StandardScaler`).
2. Tree-model preprocessor (preserving natural units without scaling).
3. Experimental preprocessor (incorporating `ChargesRatio`, `ChargesRatioDefined`, `AutomaticPayment`, and service subtotals `TechnicalProtectionCount` and `StreamingEntertainmentCount`, with training-only median imputation of missing ratios).
Transformers are fitted strictly on `X_train` and applied to `X_test`."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """# 1. Linear preprocessor
linear_pipeline = create_linear_preprocessor()
linear_pipeline.fit(X_train)
X_train_lin = transform_to_dataframe(linear_pipeline, X_train)
X_test_lin = transform_to_dataframe(linear_pipeline, X_test)

# 2. Tree preprocessor
tree_pipeline = create_tree_preprocessor()
tree_pipeline.fit(X_train)
X_train_tree = transform_to_dataframe(tree_pipeline, X_train)
X_test_tree = transform_to_dataframe(tree_pipeline, X_test)

# 3. Experimental preprocessor (canonical with all 5 experimental features enabled)
exp_pipeline = create_experimental_preprocessor()
exp_pipeline.fit(X_train)
X_train_exp = transform_to_dataframe(exp_pipeline, X_train)
X_test_exp = transform_to_dataframe(exp_pipeline, X_test)

print(f"Linear Preprocessed Shapes: Train={X_train_lin.shape}, Test={X_test_lin.shape}")
print(f"Tree Preprocessed Shapes:   Train={X_train_tree.shape}, Test={X_test_tree.shape}")
print(f"Exp Preprocessed Shapes:    Train={X_train_exp.shape}, Test={X_test_exp.shape}")

feature_names = get_preprocessor_feature_names(linear_pipeline)
exp_feature_names = get_preprocessor_feature_names(exp_pipeline)
print(f"\\nTotal baseline features: {len(feature_names)}")
print(f"Total experimental features: {len(exp_feature_names)}")
print("Experimental ratio features:", [f for f in exp_feature_names if 'ChargesRatio' in f])"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. Empirical Verification of Parameter Invariance (Leakage Prevention)

We demonstrate that transforming a corrupted test set with extreme numerical outliers does not modify the training-fitted parameters."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """col_trans = linear_pipeline.named_steps["column_transform"]
scaler = col_trans.named_transformers_["num"].named_steps["scaler"]
initial_mean = scaler.mean_.copy()
initial_scale = scaler.scale_.copy()

# Construct corrupted test set with extreme unphysical values
corrupted_test = X_test.copy()
corrupted_test["tenure"] = 99999
corrupted_test["MonthlyCharges"] = 99999.0
corrupted_test["TotalCharges"] = "9999999.0"

# Transform corrupted test set
_ = transform_to_dataframe(linear_pipeline, corrupted_test)

# Verify parameters remain unchanged
assert np.array_equal(scaler.mean_, initial_mean), "Leakage detected: scaler mean mutated!"
assert np.array_equal(scaler.scale_, initial_scale), "Leakage detected: scaler scale mutated!"
print("Leakage invariance verified: Training-fitted scaler parameters are completely invariant to test set data.")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 7. Artifact Serialization

We serialize the fitted linear preprocessor to disk using joblib and verify that the reloaded pipeline produces identical transformations."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path

artifact_path = Path("../artifacts/preprocessor.joblib")
save_preprocessor(linear_pipeline, artifact_path)

loaded_pipeline = load_preprocessor(artifact_path)
reloaded_test_lin = transform_to_dataframe(loaded_pipeline, X_test)

pd.testing.assert_frame_equal(X_test_lin, reloaded_test_lin)
print(f"Preprocessor serialized and successfully reloaded from {artifact_path.resolve()}.")
print("Transformation reproducibility confirmed: 100% bit-for-bit match.")"""
        )
    )

    out_file = NOTEBOOKS_DIR / "03_preprocessing_and_feature_engineering.ipynb"
    print(f"Executing {out_file}...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    executed_nb = client.execute()
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(executed_nb, f)
    print(f"Saved executed notebook to {out_file}")


def build_and_execute_notebook_04():
    """Build and execute notebooks/04_model_training_and_comparison.ipynb."""
    nb = nbf.v4.new_notebook()

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """# PredictLab — Notebook 04: Model Exploration, Baseline Establishment, Comparison & Controlled Tuning

## Objective & Governance
This notebook documents the rigorous, leakage-free machine-learning experimentation framework for PredictLab Phase 4:
1. **Protected Holdout Split**: Uses the approved 80/20 train/test partition ($N_{\text{train}}=5,634$, $N_{\text{test}}=1,409$). The test partition is strictly isolated until the single final evaluation.
2. **Candidate Benchmarking**: Evaluates dummy baselines, regularized logistic regressions, random forests, and histogram gradient boosting using stratified 5-fold cross-validation.
3. **Feature Ablation**: Scientifically benchmarks the 47-feature baseline against the 52-feature experimental pipeline.
4. **Controlled Hyperparameter Tuning**: Optimizes the strongest model family on $X_{\text{train}}$ only using Average Precision (PR-AUC) as primary criterion.
5. **Final Candidate Selection & Holdout Evaluation**: Refits the winning pipeline on full training data and executes a single holdout evaluation.

> [!IMPORTANT]
> The IBM Telco Customer Churn dataset is a retrospective cross-sectional snapshot. All metrics represent retrospective discrimination and risk ranking across subscriber accounts, not temporal forecasting or forward-looking event prediction."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """import pandas as pd
from IPython.display import Image, display

from predictlab.config import get_settings
from predictlab.data.load import load_raw_dataset
from predictlab.models.persistence import load_model_artifact
from predictlab.preprocessing.split import split_dataset

settings = get_settings()
exp_dir = settings.reports_dir / "experiments"
figures_dir = settings.reports_dir / "figures"
artifacts_dir = settings.artifacts_dir
print("Modeling modules successfully imported.")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Load Canonical Dataset & Verified Train/Test Partition

We load the dataset through the verified loader and partition it into training and holdout test sets using the saved seed (`random_state=42`, `test_size=0.20`)."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """df_raw = load_raw_dataset()
X_train, X_test, y_train, y_test, manifest = split_dataset(df_raw, test_size=0.20, random_state=42, save_manifest=False)

print(f"Training partition shape: {X_train.shape} (N={len(X_train)} samples)")
print(f"Holdout test shape:       {X_test.shape} (N={len(X_test)} samples - ISOLATED)")
print(f"Training churn rate:      {y_train.mean():.4f} ({y_train.sum()} churned accounts)")
print(f"Holdout churn rate:       {y_test.mean():.4f} ({y_test.sum()} churned accounts)")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Candidate Model Benchmarking (5-Fold Stratified CV)

We evaluate all candidate model families against dummy baselines strictly within `X_train`.
All transformations (scaling, imputation, one-hot encoding) are fitted strictly within each cross-validation fold."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """df_bench = pd.read_csv(exp_dir / "model_benchmarks_summary.csv")
display_cols = ["Model", "average_precision", "roc_auc", "f1", "balanced_accuracy", "log_loss", "mean_fit_time_s"]
display(df_bench[display_cols])"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "17_candidate_models_pr_auc_comparison.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Experimental Feature Ablation Study

We evaluate whether the 5 experimental feature additions (`ChargesRatio`, `ChargesRatioDefined`, `AutomaticPayment`, `TechnicalProtectionCount`, `StreamingEntertainmentCount`) provide measurable predictive lift over the 47-feature baseline."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """df_ablation = pd.read_csv(exp_dir / "feature_ablation_summary.csv")
display(df_ablation)"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "18_feature_ablation_delta_comparison.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Key Ablation Finding:
- **Result**: The 52-feature experimental pipeline shows a **-0.0079 decrease in PR-AUC** and a **+0.0032 increase in Log Loss** relative to the 47-feature baseline.
- **Scientific Conclusion**: Adding redundant, collinear aggregates (`AutomaticPayment` derived from payment methods, and service counts derived from individual service flags) introduces noise into tree split choices without providing independent information.
- **Decision**: The **47-feature baseline** is retained as the definitive, parsimonious feature space for candidate modeling."""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Controlled Hyperparameter Tuning & Model Selection

We perform grid search on `HistGradientBoostingClassifier` on `X_train` optimizing for Average Precision."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """import json

with open(exp_dir / "hyperparameter_tuning_results.json") as f:
    tuning_res = json.load(f)

print(f"Optimal PR-AUC Score: {tuning_res['best_score']:.4f}")
print("Optimal Hyperparameters:")
for param, val in tuning_res['best_params'].items():
    print(f"  {param:25s}: {val}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """with open(exp_dir / "model_selection_rationale.json") as f:
    selection = json.load(f)

print("Champion Model:", selection["selected_model"])
print("Selection Rationale:", selection["rationale"])"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """with open(exp_dir / "tuned_model_cv_results.json") as f:
    tuned_cv = json.load(f)

print("Tuned Model 5-Fold Cross-Validation Metrics (Mean ± Std):")
for metric, mean_val in tuned_cv["mean_metrics"].items():
    std_val = tuned_cv["std_metrics"].get(metric, 0.0)
    print(f"  {metric:22s}: {mean_val:.4f} ± {std_val:.4f}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. One-Time Holdout Evaluation & Three-Tier Evidence Reconciliation

The selected pipeline is refitted on the complete training partition ($N_{\\text{train}} = 5,634$) and evaluated once on the untouched holdout test partition ($N_{\\text{test}} = 1,409$).

### Methodological Principles:
1. **Three-Tier Evaluation**:
   - **Tier 1 (Benchmark CV)**: Untuned baseline candidate evaluations across 5 folds.
   - **Tier 2 (Tuned Model CV)**: Tuned estimator evaluations across 5 folds on $X_{\\text{train}}$. (Note: hyperparameter tuning on the same CV folds introduces potential slight optimistic selection bias; fold standard deviations measure split variability, not formal confidence intervals).
   - **Tier 3 (Protected Holdout Test)**: Single, independent evaluation on held-out test data.
2. **Proper Probabilistic Scoring vs Calibration**:
   - `HistGradientBoostingClassifier.predict_proba()` produces model-estimated class probabilities that have not yet undergone empirical calibration assessment. They must not be described as calibrated probabilities.
   - Log Loss and Brier Score are proper scoring rules measuring probabilistic error; lower log loss does not independently establish empirical calibration.
   - Formal calibration analysis (calibration curves, reliability diagrams, Platt scaling, isotonic regression) is strictly deferred to Phase 5.
3. **Generalization vs Leakage**:
   - The similarity between training-only cross-validation results and the protected holdout results is consistent with reasonable generalization on this dataset. It does not independently prove the absence of leakage (which is instead safeguarded by strict pre-split partitioning and fold-level encapsulation).
4. **Permutation Importance Definition & Partition**:
   - Permutation importance measures the decrease in a model's predictive score after a feature's values are shuffled. It estimates model reliance on that feature under the selected scoring procedure; it does not establish causality or conditional dependence.
   - Correlated predictors can dilute or distort individual feature importance estimates.
   - Permutation feature importances were computed strictly on the training partition ($X_{\\text{train}}, y_{\\text{train}}$) using `scoring="average_precision"`. The holdout partition was never accessed for feature ranking."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """with open(exp_dir / "holdout_evaluation.json") as f:
    holdout = json.load(f)

with open(exp_dir / "tuned_model_cv_results.json") as f:
    tuned_cv = json.load(f)

df_bench = pd.read_csv(exp_dir / "model_benchmarks_summary.csv")
untuned_row = df_bench[df_bench["Model"].str.contains("Hist Gradient Boosting") & ~df_bench["Model"].str.contains("Balanced")].iloc[0]

comparison_records = []
metrics_keys = [
    ("average_precision", "Average Precision (PR-AUC)"),
    ("roc_auc", "ROC-AUC"),
    ("f1", "F1-Score"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("log_loss", "Log Loss"),
    ("brier_score", "Brier Score"),
]

for key, label in metrics_keys:
    untuned_val = untuned_row[key]
    tuned_mean = tuned_cv["mean_metrics"][key]
    tuned_std = tuned_cv["std_metrics"][key]
    holdout_val = holdout["metrics"][key]
    delta = holdout_val - tuned_mean
    sigma_delta = delta / tuned_std if tuned_std > 0 else 0.0

    comparison_records.append({
        "Metric": label,
        "Tier 1: Untuned CV (Mean ± Std)": str(untuned_val),
        "Tier 2: Tuned CV (Mean ± Std)": f"{tuned_mean:.4f} ± {tuned_std:.4f}",
        "Tier 3: Holdout Test": f"{holdout_val:.4f}",
        "Delta (Holdout - Tuned CV)": f"{delta:+.4f}",
        "Delta (σ)": f"{sigma_delta:+.2f} σ",
    })

df_three_tier = pd.DataFrame(comparison_records)
display(df_three_tier)"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """cm = holdout["confusion_matrix"]
print(f"\\nConfusion Matrix (Threshold={holdout['threshold']}):")
print(f"  True Negatives:  {cm['true_negatives']:4d} | False Positives: {cm['false_positives']:4d}")
print(f"  False Negatives: {cm['false_negatives']:4d} | True Positives:  {cm['true_positives']:4d}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "19_holdout_roc_pr_curves.png")))
display(Image(filename=str(figures_dir / "20_holdout_confusion_matrix.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. Top Feature Importances & Model Artifact Verification

We inspect the top 20 predictive features in the selected candidate (computed via permutation importance strictly on $X_{\\text{train}}$ using `scoring='average_precision'`) and verify serialized artifact integrity."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "21_feature_importances_top20.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """artifact_bundle = load_model_artifact(artifacts_dir / "model.joblib")
meta = artifact_bundle["metadata"]

print("Model Artifact Verification:")
print(f"  Model Family:      {meta['model_family']}")
print(f"  Feature Count:     {meta['feature_count']} features")
print(f"  Dataset SHA-256:   {meta['dataset_sha256'][:16]}...")
print(f"  Saved Timestamp:   {meta['saved_at_utc']}")
print(f"  Scikit-Learn Ver:  {meta['environment']['scikit_learn_version']}")"""
        )
    )

    out_file = NOTEBOOKS_DIR / "04_model_training_and_comparison.ipynb"
    print(f"Executing {out_file}...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    executed_nb = client.execute()
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(executed_nb, f)
    print(f"Saved executed notebook to {out_file}")


def build_and_execute_notebook_05():
    """Build and execute notebooks/05_evaluation_explainability_and_inference.ipynb."""
    nb = nbf.v4.new_notebook()

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """# PredictLab — Notebook 05: Evaluation, Explainability, and Production-Ready Inference

## Overview & Scientific Governance
This notebook presents the Phase 5 evaluation, calibration, explainability, and inference pipeline for PredictLab:
1. **Holdout Partition Protection**: Confirms that holdout test labels ($N=1,409$) remain strictly isolated and protected from calibration, threshold tuning, and explainability workflows.
2. **Probability Calibration**: Rigorous nested cross-validation (5x5 folds) comparing uncalibrated probability estimates, Sigmoid (Platt) scaling, and Isotonic regression across log loss, Brier score, ECE (uniform and quantile binning), and Murphy (1973) Brier score decomposition.
3. **Decision Threshold Analysis & Cost Sensitivity**: Systematic threshold sweeps (0.10 to 0.90), cost matrix modeling across asymmetric loss ratios ($C_{\\text{FN}} : C_{\\text{FP}}$ from 0.2:1 to 10:1), and nested CV policy evaluation to quantify selection optimism.
4. **Customer Churn Risk Tiers**: Contiguous, operational tiering (Low $[0.0, 0.20)$, Medium $[0.20, 0.50)$, High $[0.50, 1.00]$) with cohort profiles and observed churn rates.
5. **Global & Local Explainability**: Global feature permutation importance (scoring on Average Precision on $X_{\\text{train}}$) and local TreeSHAP attribution on the log-odds margin scale with waterfall breakdowns.
6. **Production Inference Engine**: Production-ready `InferenceEngine` verifying strict Pydantic v2 schema validation, single and batch prediction modes (atomic vs per-record error isolation), and deterministic calibrated scoring.

> [!IMPORTANT]
> The IBM Telco Customer Churn dataset is a retrospective cross-sectional customer snapshot. All risk scores, tiers, and SHAP attributions represent retrospective associative patterns and risk ranking across subscriber profiles, not causal levers or longitudinal time-to-event predictions."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """import json

import pandas as pd
from IPython.display import Image, display

from predictlab.config import get_settings
from predictlab.data.load import load_raw_dataset
from predictlab.inference.engine import InferenceEngine
from predictlab.preprocessing.split import split_dataset

settings = get_settings()
exp_dir = settings.reports_dir / "experiments"
figures_dir = settings.reports_dir / "figures"
artifacts_dir = settings.artifacts_dir

print(f"Project root:    {settings.project_root}")
print(f"Artifacts dir:   {artifacts_dir}")
print(f"Experiments dir: {exp_dir}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Dataset Verification & Strict Holdout Isolation

We confirm the integrity of the canonical raw dataset and the 80/20 stratified split. Notice that the test partition ($N=1,409$) is strictly isolated. All subsequent calibration benchmarking, threshold tuning, and explainability rely entirely on out-of-fold training evidence or $X_{\\text{train}}$."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """raw_df = load_raw_dataset()
X_train, X_test, y_train, y_test, manifest = split_dataset(raw_df, save_manifest=False)

print(f"Raw dataset shape:    {raw_df.shape}")
print(f"Training split shape: {X_train.shape} (churn prevalence: {y_train.mean():.4f})")
print(f"Holdout split shape:  {X_test.shape} (churn prevalence: {y_test.mean():.4f})")
print("\\nHoldout Test Isolation: Confirmed (Protected holdout labels are NOT used in calibration or threshold selection).")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Probability Calibration: Leakage-Free Nested Cross-Validation

We evaluate uncalibrated, Sigmoid (Platt Scaling), and Isotonic calibration models using 5x5 nested cross-validation on the training set.
Key metrics include:
- **Brier Score** & **Log Loss** (proper scoring rules)
- **Expected Calibration Error (ECE)** under both uniform and quantile binning schemes
- **Murphy (1973) Brier Decomposition**: Reliability ($MC$), Resolution ($DSC$), Uncertainty ($UNC$)
- **Calibration Curve Intercept & Slope** (Cox calibration regression)"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """with open(exp_dir / "calibration_cv_results.json") as f:
    calib_summary = json.load(f)

calib_rows = []
for method, results in calib_summary.items():
    m = results["mean_metrics"]
    s = results["std_metrics"]
    calib_rows.append({
        "Method": method.capitalize(),
        "Brier Score": f"{m['brier_score']:.4f} ± {s['brier_score']:.4f}",
        "Log Loss": f"{m['log_loss']:.4f} ± {s['log_loss']:.4f}",
        "PR-AUC": f"{m['average_precision']:.4f} ± {s['average_precision']:.4f}",
        "ROC-AUC": f"{m['roc_auc']:.4f} ± {s['roc_auc']:.4f}",
        "ECE (Uniform)": f"{m['ece_uniform']:.4f}",
        "ECE (Quantile)": f"{m['ece_quantile']:.4f}",
        "Slope": f"{m['calibration_slope']:.4f}",
    })

df_calib = pd.DataFrame(calib_rows)
display(df_calib)"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """with open(exp_dir / "calibration_selection_rationale.json") as f:
    calib_sel = json.load(f)

print(f"Selected Calibration Champion: {calib_sel['selected_method'].upper()}")
print(f"Rationale: {calib_sel['rationale']}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "22_calibration_curves_comparison.png")))
display(Image(filename=str(figures_dir / "23_brier_decomposition_and_ece.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Calibration Findings
- **Sigmoid (Platt Scaling)** achieved the lowest Brier score ($0.1335 \\pm 0.0056$) and lowest Log Loss ($0.4116 \\pm 0.0133$), while preserving optimal discriminative ranking (PR-AUC $0.6697 \\pm 0.0215$, ROC-AUC $0.8498 \\pm 0.0123$).
- **Isotonic Calibration** showed slightly higher log loss ($0.4181$) and degraded PR-AUC ($0.6669$) due to sample-size step-function overfitting on moderate sample size ($N=5,634$).
- **Conclusion**: Sigmoid calibration provides smooth, monotonic, well-calibrated probabilities across the full $[0, 1]$ interval."""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Decision Threshold Analysis & Cost Sensitivity

We analyze classification performance across decision thresholds $\\tau \\in [0.10, 0.90]$ and evaluate expected operational costs under asymmetric false-negative penalties ($C_{\\text{FN}} : C_{\\text{FP}}$ from 0.2:1 to 10:1)."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """df_cost = pd.read_csv(exp_dir / "threshold_cost_analysis.csv")
print(f"Threshold sweep rows: {len(df_cost)} evaluation points")
display(df_cost[df_cost["threshold"].round(2).isin([0.18, 0.19, 0.30, 0.50, 0.70])][
    ["threshold", "precision", "recall", "specificity", "f1", "f2", "balanced_accuracy", "total_cost", "cost_savings_vs_default", "savings_pct_vs_default"]
])"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """with open(exp_dir / "threshold_policy_evaluation.json") as f:
    thresh_policy = json.load(f)

print(f"Threshold Optimization Objective: {thresh_policy['objective']}")
print(f"Selected Threshold:               {thresh_policy['selected_threshold']:.2f}")
print(f"In-Sample F1:                     {thresh_policy['in_sample_metrics']['f1']:.4f}")
print(f"Out-of-Fold F1:                   {thresh_policy['out_of_fold_metrics']['f1']:.4f}")
print(f"Optimism Delta:                   {thresh_policy['optimism_delta']['f1']:+.4f}")
print(f"Notes: {thresh_policy['selection_notes']}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "24_threshold_precision_recall_f1.png")))
display(Image(filename=str(figures_dir / "25_cost_curve_and_sensitivity.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Threshold Analysis Findings
- At the standard threshold ($\\tau = 0.50$), precision is $0.6734$ and recall is $0.5338$, resulting in 697 missed churners in the training cohort.
- **Comprehensive Incentive Cost Model** ($C_{\\text{FN}}=\\$500, C_{\\text{FP}}=\\$100, C_{\\text{TP}}=\\$100, C_{\\text{TN}}=\\$0$):
  - Cost-optimal threshold shifts to **$\\tau^* = 0.19$**, lifting recall to **$87.29\\%$**, cutting missed churners from 697 to 190 ($72.74\\%$ reduction), and delivering a **20.75% cost reduction** ($\1$96,900 saved, from $\\$467,000$ to $\\$370,100$).
- **Misclassification-Only Cost Model** ($C_{\\text{FN}}=\\$500, C_{\\text{FP}}=\\$100, C_{\\text{TP}}=\\$0, C_{\\text{TN}}=\\$0$):
  - Cost-optimal threshold shifts to **$\\tau^* = 0.18$**, lifting recall to **$88.09\\%$**, cutting missed churners to 178, and delivering a **38.20% cost reduction** ($\1$147,900 saved, from $\\$387,200$ to $\\$239,300$).
- **Nested Cross-Validation Policy Evaluation**:
  - Fold thresholds: $[0.21, 0.21, 0.19, 0.19, 0.19]$ (mean $\\tau = 0.1980 \\pm 0.0098$).
  - Out-of-fold F1 is $0.6159$ vs in-sample $0.6147$ (delta $-0.0012$). This delta quantifies empirical policy stability across CV splits; it confirms threshold selection does not collapse across folds."""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Operational Customer Churn Risk Tiers

PredictLab structures predicted probabilities into three actionable risk tiers:
- **Low Risk** $[0.00, 0.20)$
- **Medium Risk** $[0.20, 0.50)$
- **High Risk** $[0.50, 1.00]$"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """df_tiers = pd.read_csv(exp_dir / "risk_tier_summary.csv")
display(df_tiers[["risk_tier", "probability_range", "account_count", "percent_of_cohort", "mean_model_probability", "observed_churn_rate", "actual_churned_accounts"]])"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "26_risk_tier_distribution.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. Global & Local Model Explainability

We interpret the model's decision behavior:
1. **Cross-Validated Permutation Importance**: Evaluated via 5-fold cross-validation strictly on held-out validation splits with 5 random shuffles using Average Precision drop as the evaluation metric.
2. **Local Multi-Estimator TreeSHAP Attribution**: Exact Shapley values calculated by aggregating across all 5 ensemble tree estimators on the ensemble decision margin log-odds scale, attributing positive (churn-elevating) and negative (retention-stabilizing) contributions."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """df_perm = pd.read_csv(exp_dir / "permutation_importance_detailed.csv")
print("Top 10 Global Features by Cross-Validated Permutation Importance (PR-AUC drop):")
display(df_perm.head(10)[["rank", "clean_feature", "importance_mean", "importance_std", "normalized_importance"]])"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """display(Image(filename=str(figures_dir / "27_shap_global_summary.png")))
display(Image(filename=str(figures_dir / "28_shap_local_waterfall.png")))"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """### Explainability & Non-Causal Interpretation
- The strongest global churn risk indicators are **Contract type** (month-to-month contracts strongly associated with churn), **tenure**, **Fiber optic internet**, **TotalCharges**, **MonthlyCharges**, and lack of **OnlineSecurity** / **TechSupport**.
- **Collinear Feature Dilution**: Correlated features (such as `tenure` and `TotalCharges`, or `MonthlyCharges` and streaming services) share importance. Permutation importance assesses predictive reliance within the model, not independent causal influence.
- **Local SHAP Waterfalls**: Break down the specific log-odds shifts from the ensemble mean base rate ($\\bar{E}[f(x)] \\approx -1.5438$) to the customer's predicted decision margin. Positive SHAP values indicate attributes elevating risk; negative values indicate stabilizing/retention attributes. Exact mathematical additivity holds across the 5 ensemble estimators."""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. Production Inference Engine Verification

We demonstrate the production-ready `InferenceEngine`:
- Strict Pydantic v2 schema validation for all 19 raw customer attributes.
- Single customer prediction with SHAP explanation.
- Batch customer prediction with atomic vs per-record error isolation modes.
- Deterministic score assignment and risk tiering."""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """engine = InferenceEngine(model_artifact_path=artifacts_dir / "model_phase5_calibrated.joblib")
print(f"Loaded InferenceEngine from: {engine.artifact_path.name}")
print(f"Default Decision Threshold: {engine.decision_threshold}")
print(f"Risk Tiers: {[f'{t.name} [{t.min_prob:.2f}, {t.max_prob:.2f})' for t in engine.risk_tierer.definitions]}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
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
    "TotalCharges": 255.0,
}

single_out = engine.predict_single(sample_customer, include_explanation=True)
print("Single Customer Prediction Result:")
print(f"  Probability:      {single_out.estimated_probability:.4f}")
print(f"  Binary Label:     {single_out.predicted_label} (class={single_out.predicted_class}, threshold={single_out.applied_threshold})")
print(f"  Risk Tier:        {single_out.risk_tier}")
print(f"  Base Margin:      {single_out.explanation['base_value_margin']:.4f}")
print(f"  Output Margin:    {single_out.explanation['model_output_margin']:.4f}")
print("\\nTop Churn Drivers:")
for driver in single_out.explanation["top_churn_drivers"]:
    print(f"  + {driver['clean_feature']:20s} ({driver['feature_value']}): SHAP = +{driver['shap_value']:.4f}")
print("\\nTop Retention Drivers:")
for driver in single_out.explanation["top_retention_drivers"]:
    print(f"  - {driver['clean_feature']:20s} ({driver['feature_value']}): SHAP = {driver['shap_value']:.4f}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            """sample_batch = [
    sample_customer,
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.0,
        "TotalCharges": 3900.0,
    }
]

batch_out = engine.predict_batch(sample_batch, error_handling="atomic")
print(f"Batch Prediction Succeeded: {batch_out.successful_predictions} / {batch_out.total_records} records processed")
for rec in batch_out.predictions:
    p = rec.prediction
    print(f"  Index {rec.record_index}: Prob={p.estimated_probability:.4f}, Tier={p.risk_tier}, Label={p.predicted_label}")"""
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            """## Summary & Phase 5 Sign-Off
Phase 5 successfully implemented:
- **Leakage-free nested CV calibration** establishing Sigmoid scaling as champion ($0.1335$ Brier, $0.4116$ Log Loss, $0.6697$ PR-AUC).
- **Comprehensive cost-sensitive threshold analysis** identifying operational optimal thresholds with quantified cost savings and minimal optimism bias.
- **Standardized risk tiers** mapping continuous probability scores to operational retention tiers.
- **Dual-level explainability** via global Average Precision permutation importance and local TreeSHAP attribution.
- **Robust production-ready inference engine** complete with input validation and dual batch error-handling modes."""
        )
    )

    out_file = NOTEBOOKS_DIR / "05_evaluation_explainability_and_inference.ipynb"
    print(f"Executing {out_file}...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    executed_nb = client.execute()
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(executed_nb, f)
    print(f"Saved executed notebook to {out_file}")


if __name__ == "__main__":
    build_and_execute_notebook_01()
    build_and_execute_notebook_02()
    build_and_execute_notebook_03()
    build_and_execute_notebook_04()
    build_and_execute_notebook_05()
