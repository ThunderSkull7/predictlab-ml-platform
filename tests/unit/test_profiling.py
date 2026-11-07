"""Unit tests for profiling, data quality, and diagnostic insights modules."""

import pandas as pd
import pytest

from predictlab.profiling.insights import (
    benjamini_hochberg_correction,
    compute_categorical_associations,
    compute_correlation_matrix,
    compute_cramers_v,
    compute_numerical_churn_tests,
    compute_segment_churn_rates,
)
from predictlab.profiling.quality import (
    DataQualityReport,
    TotalChargesInvestigation,
    detect_whitespace_entries,
    investigate_total_charges,
    profile_data_quality,
)
from predictlab.profiling.summary import (
    compute_categorical_summary,
    compute_dataset_overview,
    compute_numeric_summary,
    compute_target_distribution,
)
from tests.fixtures.synthetic_data import get_synthetic_telco_data


def test_dataset_overview():
    """Verify calculation of dataset overview dimensions and metrics."""
    df = get_synthetic_telco_data(num_rows=15)
    overview = compute_dataset_overview(df)

    assert overview["total_observations"] == 15
    assert overview["total_features"] == 21
    assert overview["unique_customers"] == 15
    assert overview["duplicate_rows"] == 0
    assert overview["duplicate_identifiers"] == 0
    assert overview["identifier_column"] == "customerID"
    assert overview["memory_usage_mb"] > 0


def test_numeric_summary_correctness():
    """Verify statistical summary metrics on known numerical series."""
    df = pd.DataFrame(
        {
            "num_val": [10.0, 20.0, 30.0, 40.0, 50.0],
            "with_ws": ["100.0", " ", "200.0", "300.0", "400.0"],
        }
    )
    summary = compute_numeric_summary(df, columns=["num_val", "with_ws"])

    assert len(summary) == 2
    row0 = summary.loc[summary["feature"] == "num_val"].iloc[0]
    assert row0["valid_count"] == 5
    assert row0["mean"] == 30.0
    assert row0["median"] == 30.0
    assert row0["min"] == 10.0
    assert row0["max"] == 50.0
    assert row0["q25"] == 20.0
    assert row0["q75"] == 40.0
    assert row0["iqr"] == 20.0

    row1 = summary.loc[summary["feature"] == "with_ws"].iloc[0]
    assert row1["valid_count"] == 4
    assert row1["whitespace_count"] == 1
    assert row1["mean"] == 250.0


def test_categorical_summary():
    """Verify category frequencies and percentage calculations."""
    df = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
            "Churn": ["Yes", "No", "No", "No"],
        }
    )
    cat_summary = compute_categorical_summary(df, columns=["Contract"])
    assert "Contract" in cat_summary
    c_df = cat_summary["Contract"]
    assert len(c_df) == 3

    m2m_row = c_df[c_df["category"] == "Month-to-month"].iloc[0]
    assert m2m_row["frequency"] == 2
    assert m2m_row["percentage"] == 50.0


def test_target_distribution():
    """Verify target distribution counts, percentages, and class balance ratios."""
    df = pd.DataFrame({"Churn": ["Yes", "No", "No", "No"]})
    res = compute_target_distribution(df, target_col="Churn", positive_class="Yes", negative_class="No")

    assert res["total_records"] == 4
    assert res["positive_count"] == 1
    assert res["negative_count"] == 3
    assert res["positive_percentage"] == 25.0
    assert res["negative_percentage"] == 75.0
    assert res["negative_to_positive_ratio"] == 3.0


def test_target_distribution_missing_column():
    """Verify ValueError is raised if target column is missing."""
    df = pd.DataFrame({"Feature": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column 'Churn' not found"):
        compute_target_distribution(df, target_col="Churn")


def test_whitespace_detection():
    """Verify detection of whitespace-only strings."""
    df = pd.DataFrame(
        {
            "col_clean": ["A", "B", "C"],
            "col_ws": ["Valid", "  ", "\t \n"],
            "col_num": [1, 2, 3],
        }
    )
    ws_dict = detect_whitespace_entries(df)
    assert "col_clean" not in ws_dict
    assert "col_ws" in ws_dict
    assert ws_dict["col_ws"] == [1, 2]


def test_total_charges_investigation():
    """Verify total charges investigation on synthetic data mimicking the 11 zero-tenure anomaly."""
    records = []
    # 5 zero tenure rows with whitespace TotalCharges
    for i in range(5):
        records.append(
            {
                "customerID": f"C-ZERO-{i}",
                "tenure": 0,
                "MonthlyCharges": 50.0 + i,
                "TotalCharges": " ",
                "Contract": "Two year",
                "Churn": "No",
            }
        )
    # 10 tenure==1 rows with exact match
    for i in range(10):
        records.append(
            {
                "customerID": f"C-ONE-{i}",
                "tenure": 1,
                "MonthlyCharges": 60.0 + i,
                "TotalCharges": f"{60.0 + i}",
                "Contract": "Month-to-month",
                "Churn": "Yes" if i % 2 == 0 else "No",
            }
        )
    # 10 tenure==10 rows
    for i in range(10):
        records.append(
            {
                "customerID": f"C-TEN-{i}",
                "tenure": 10,
                "MonthlyCharges": 70.0,
                "TotalCharges": f"{700.0 + i}",
                "Contract": "One year",
                "Churn": "No",
            }
        )
    df = pd.DataFrame(records)

    inv = investigate_total_charges(df)
    assert isinstance(inv, TotalChargesInvestigation)
    assert inv.total_whitespace_rows == 5
    assert inv.all_zero_tenure is True
    assert inv.tenure_one_exact_match_ratio == 1.0
    assert inv.churn_distribution == {"No": 5}
    assert "0.0" in inv.recommendation


def test_profile_data_quality():
    """Verify comprehensive DataQualityReport generation."""
    df = get_synthetic_telco_data(num_rows=10)
    report = profile_data_quality(df)

    assert isinstance(report, DataQualityReport)
    assert report.total_rows == 10
    assert report.total_columns == 21
    assert report.duplicate_rows == 0
    assert report.duplicate_ids == 0
    assert "customerID" in report.column_profiles
    assert report.column_profiles["customerID"].is_unique_identifier is True


def test_segment_churn_rates():
    """Verify segment churn rates and relative risk calculations."""
    df = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "Month-to-month", "Two year", "Two year"],
            "Churn": ["Yes", "Yes", "No", "No"],
        }
    )
    rates = compute_segment_churn_rates(df, group_cols="Contract")

    assert len(rates) == 2
    m2m = rates[rates["Contract"] == "Month-to-month"].iloc[0]
    assert m2m["total_customers"] == 2
    assert m2m["churned_customers"] == 2
    assert m2m["churn_rate_pct"] == 100.0
    assert m2m["relative_risk"] == 2.0  # 100% vs 50% baseline

    two_yr = rates[rates["Contract"] == "Two year"].iloc[0]
    assert two_yr["churn_rate_pct"] == 0.0
    assert two_yr["relative_risk"] == 0.0


def test_benjamini_hochberg_correction():
    """Verify Benjamini-Hochberg adjustment monotonic property and values."""
    p_vals = [0.001, 0.01, 0.03, 0.04, 0.80]
    adj = benjamini_hochberg_correction(p_vals)

    assert len(adj) == 5
    assert adj[0] <= adj[1] <= adj[2] <= adj[3] <= adj[4]
    assert all(0.0 <= x <= 1.0 for x in adj)
    # 0.001 * 5 / 1 = 0.005
    assert adj[0] == pytest.approx(0.005)


def test_cramers_v_and_categorical_associations():
    """Verify Cramér's V and Chi-square association calculations."""
    # Synthetic dataset with strong association
    df = pd.DataFrame(
        {
            "Segment": ["A"] * 50 + ["B"] * 50,
            "Churn": ["Yes"] * 45 + ["No"] * 5 + ["No"] * 45 + ["Yes"] * 5,
        }
    )
    ct = pd.crosstab(df["Segment"], df["Churn"])
    cv = compute_cramers_v(ct)
    assert cv > 0.5  # Strong association

    assoc = compute_categorical_associations(df, cat_cols=["Segment"])
    assert len(assoc) == 1
    assert assoc.iloc[0]["feature"] == "Segment"
    assert bool(assoc.iloc[0]["significant"])
    assert assoc.iloc[0]["cramers_v"] > 0.5


def test_numerical_churn_tests():
    """Verify non-parametric and parametric test computations for numeric columns."""
    df = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 5, 50, 60, 70, 71, 72],
            "MonthlyCharges": [70.0, 80.0, 90.0, 85.0, 95.0, 20.0, 25.0, 22.0, 19.0, 21.0],
            "Churn": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No"],
        }
    )
    res = compute_numerical_churn_tests(df, num_cols=["tenure", "MonthlyCharges"])

    assert len(res) == 2
    tenure_row = res[res["feature"] == "tenure"].iloc[0]
    assert tenure_row["churned_median"] == 3.0
    assert tenure_row["retained_median"] == 70.0
    assert tenure_row["mann_whitney_p"] < 0.05
    assert bool(tenure_row["significant"])


def test_correlation_matrix():
    """Verify Spearman and Pearson correlation computation."""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [5, 4, 3, 2, 1],
        }
    )
    spearman_corr = compute_correlation_matrix(df, num_cols=["A", "B", "C"], method="spearman")
    assert spearman_corr.loc["A", "B"] == 1.0
    assert spearman_corr.loc["A", "C"] == -1.0
