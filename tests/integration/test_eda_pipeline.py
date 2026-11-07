"""Integration tests for end-to-end exploratory profiling on the canonical dataset."""

from predictlab.config import get_settings
from predictlab.data.load import load_raw_dataset
from predictlab.profiling.insights import (
    compute_categorical_associations,
    compute_correlation_matrix,
    compute_numerical_churn_tests,
    compute_segment_churn_rates,
)
from predictlab.profiling.quality import (
    profile_data_quality,
)
from predictlab.profiling.summary import (
    compute_categorical_summary,
    compute_dataset_overview,
    compute_numeric_summary,
    compute_target_distribution,
)


def test_canonical_dataset_profiling_pipeline():
    """Verify end-to-end data quality profiling and summary generation on the canonical raw dataset."""
    settings = get_settings()
    if not settings.raw_data_path.exists():
        return

    df = load_raw_dataset()
    assert df.shape == (7043, 21)

    # 1. Dataset overview
    overview = compute_dataset_overview(df)
    assert overview["total_observations"] == 7043
    assert overview["total_features"] == 21
    assert overview["unique_customers"] == 7043
    assert overview["duplicate_rows"] == 0

    # 2. Data quality audit
    quality = profile_data_quality(df)
    assert quality.total_rows == 7043
    assert quality.duplicate_rows == 0
    assert quality.columns_with_whitespace == ["TotalCharges"]
    assert quality.total_charges_investigation.total_whitespace_rows == 11
    assert quality.total_charges_investigation.all_zero_tenure is True
    assert quality.total_charges_investigation.tenure_one_exact_match_ratio == 1.0

    # 3. Numeric and categorical summaries
    num_summary = compute_numeric_summary(df, columns=["tenure", "MonthlyCharges", "TotalCharges"])
    assert len(num_summary) == 3
    features = set(num_summary["feature"].tolist())
    assert features == {"tenure", "MonthlyCharges", "TotalCharges"}

    cat_summaries = compute_categorical_summary(df)
    assert "Contract" in cat_summaries
    assert "InternetService" in cat_summaries
    assert "PaymentMethod" in cat_summaries

    # 4. Target distribution
    target_dist = compute_target_distribution(df)
    assert target_dist["positive_count"] == 1869
    assert target_dist["negative_count"] == 5174
    assert target_dist["positive_percentage"] == 26.54
    assert target_dist["negative_percentage"] == 73.46

    # 5. Segment churn rates
    contract_rates = compute_segment_churn_rates(df, group_cols="Contract")
    assert len(contract_rates) == 3
    # Month-to-month should have the highest churn rate
    assert contract_rates.iloc[0]["Contract"] == "Month-to-month"
    assert contract_rates.iloc[0]["churn_rate_pct"] > 40.0

    # 6. Categorical associations
    cat_cols = [
        "Contract",
        "InternetService",
        "OnlineSecurity",
        "TechSupport",
        "PaymentMethod",
        "PaperlessBilling",
    ]
    assoc = compute_categorical_associations(df, cat_cols=cat_cols)
    assert len(assoc) == len(cat_cols)
    assert all(assoc["significant"])
    # Contract should be the strongest association
    assert assoc.iloc[0]["feature"] == "Contract"

    # 7. Numerical hypothesis tests
    num_tests = compute_numerical_churn_tests(df, num_cols=["tenure", "MonthlyCharges", "TotalCharges"])
    assert len(num_tests) == 3
    assert all(num_tests["significant"])

    # 8. Correlation matrix
    corr = compute_correlation_matrix(df, num_cols=["tenure", "MonthlyCharges", "TotalCharges"])
    assert corr.shape == (3, 3)
    assert corr.loc["tenure", "TotalCharges"] > 0.8
