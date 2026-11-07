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


if __name__ == "__main__":
    build_and_execute_notebook_01()
    build_and_execute_notebook_02()
