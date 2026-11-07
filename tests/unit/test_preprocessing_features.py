"""Unit tests for feature engineering transformers and domain calculations."""

import numpy as np
import pandas as pd
import pytest

from predictlab.preprocessing.feature_engineering import (
    compute_automatic_payment_flag,
    compute_charges_ratio_defined,
    compute_experimental_charges_ratio,
    compute_optional_service_count,
    compute_streaming_entertainment_count,
    compute_technical_protection_count,
)
from predictlab.preprocessing.schema import InvalidCategoryValueError
from predictlab.preprocessing.transformers import (
    AutomaticPaymentTransformer,
    ExperimentalChargesRatioTransformer,
    OptionalServiceCounter,
)


@pytest.fixture
def service_sample_df() -> pd.DataFrame:
    """Fixture providing service attributes with known combinations."""
    return pd.DataFrame(
        {
            "OnlineSecurity": ["Yes", "No", "No internet service", "Yes"],
            "OnlineBackup": ["Yes", "No", "No internet service", "No"],
            "DeviceProtection": ["Yes", "No", "No internet service", "Yes"],
            "TechSupport": ["Yes", "No", "No internet service", "No"],
            "StreamingTV": ["Yes", "No", "No internet service", "Yes"],
            "StreamingMovies": ["Yes", "No", "No internet service", "No"],
            "PaymentMethod": [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            "tenure": [10, 0, 24, 1],
            "MonthlyCharges": [70.0, 20.0, 50.0, 80.0],
            "TotalCharges": [700.0, 0.0, 1200.0, 80.0],
        }
    )


def test_optional_service_count_combinations(service_sample_df: pd.DataFrame) -> None:
    """Test OptionalServiceCount across all-yes, all-no, no-internet, and mixed records."""
    counts = compute_optional_service_count(service_sample_df)

    assert counts.iloc[0] == 6  # All 6 Yes
    assert counts.iloc[1] == 0  # All No
    assert counts.iloc[2] == 0  # All No internet service -> 0
    assert counts.iloc[3] == 3  # Exactly 3 Yes (OnlineSecurity, DeviceProtection, StreamingTV)
    assert counts.dtype in (np.int32, np.int64, int)


def test_service_subtotals(service_sample_df: pd.DataFrame) -> None:
    """Test TechnicalProtectionCount and StreamingEntertainmentCount sub-aggregations."""
    tech_counts = compute_technical_protection_count(service_sample_df)
    streaming_counts = compute_streaming_entertainment_count(service_sample_df)

    # Row 0: All Yes -> Tech=4, Streaming=2
    assert tech_counts.iloc[0] == 4
    assert streaming_counts.iloc[0] == 2

    # Row 1: All No -> Tech=0, Streaming=0
    assert tech_counts.iloc[1] == 0
    assert streaming_counts.iloc[1] == 0

    # Row 3: OnlineSecurity(Yes), DeviceProtection(Yes), StreamingTV(Yes) -> Tech=2, Streaming=1
    assert tech_counts.iloc[3] == 2
    assert streaming_counts.iloc[3] == 1


def test_optional_service_count_raises_on_invalid_category() -> None:
    """Test that unexpected categorical values in service columns raise InvalidCategoryValueError."""
    bad_df = pd.DataFrame(
        {
            "OnlineSecurity": ["Maybe"],  # Invalid!
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
        }
    )
    with pytest.raises(InvalidCategoryValueError) as exc:
        compute_optional_service_count(bad_df)
    assert "unrecognized values" in str(exc.value).lower()


def test_automatic_payment_flag(service_sample_df: pd.DataFrame) -> None:
    """Test AutomaticPayment binary assignment across payment mechanisms."""
    flags = compute_automatic_payment_flag(service_sample_df)

    assert flags.iloc[0] == 1  # Bank transfer (automatic)
    assert flags.iloc[1] == 1  # Credit card (automatic)
    assert flags.iloc[2] == 0  # Electronic check
    assert flags.iloc[3] == 0  # Mailed check


def test_automatic_payment_raises_on_invalid_method() -> None:
    """Test that unrecognized payment method strings raise InvalidCategoryValueError."""
    bad_df = pd.DataFrame({"PaymentMethod": ["Bitcoin / Crypto"]})
    with pytest.raises(InvalidCategoryValueError):
        compute_automatic_payment_flag(bad_df)


def test_experimental_charges_ratio(service_sample_df: pd.DataFrame) -> None:
    """Test ChargesRatio calculation and mathematically valid zero-tenure NaN representation."""
    ratios = compute_experimental_charges_ratio(service_sample_df)

    # Row 0: tenure=10, mc=70, tc=700 -> 700 / (10 * 70) = 1.0
    assert pytest.approx(ratios.iloc[0], rel=1e-4) == 1.0

    # Row 1: tenure=0 -> strictly NaN (mathematically undefined 0/0; never 1.0 or epsilon)
    assert np.isnan(ratios.iloc[1])

    # Row 2: tenure=24, mc=50, tc=1200 -> 1200 / 1200 = 1.0
    assert pytest.approx(ratios.iloc[2], rel=1e-4) == 1.0

    # Row 3: tenure=1, mc=80, tc=80 -> 80 / 80 = 1.0
    assert pytest.approx(ratios.iloc[3], rel=1e-4) == 1.0


def test_charges_ratio_defined_indicator(service_sample_df: pd.DataFrame) -> None:
    """Test ChargesRatioDefined deterministic binary indicator (1 if defined, 0 if undefined)."""
    defined = compute_charges_ratio_defined(service_sample_df)

    assert defined.iloc[0] == 1  # tenure=10 -> defined
    assert defined.iloc[1] == 0  # tenure=0  -> undefined
    assert defined.iloc[2] == 1  # tenure=24 -> defined
    assert defined.iloc[3] == 1  # tenure=1  -> defined
    assert defined.dtype in (np.int32, np.int64, int)


def test_experimental_charges_ratio_invalid_denominators() -> None:
    """Test that zero or non-positive denominators for positive tenure raise ValueError."""
    bad_denom_df = pd.DataFrame(
        {
            "tenure": [10],
            "MonthlyCharges": [0.0],  # Denominator evaluates to 0 on positive tenure
            "TotalCharges": [100.0],
        }
    )
    with pytest.raises(ValueError) as exc:
        compute_experimental_charges_ratio(bad_denom_df)
    assert "non-positive denominator" in str(exc.value).lower()


def test_experimental_charges_ratio_non_finite_inputs() -> None:
    """Test that non-finite inputs on positive tenure raise ValueError."""
    bad_df = pd.DataFrame(
        {
            "tenure": [10],
            "MonthlyCharges": [np.inf],
            "TotalCharges": [100.0],
        }
    )
    with pytest.raises(ValueError) as exc:
        compute_experimental_charges_ratio(bad_df)
    assert "non-finite" in str(exc.value).lower()


def test_experimental_charges_ratio_immutability(service_sample_df: pd.DataFrame) -> None:
    """Test that input DataFrame is not mutated during ratio calculation."""
    original_copy = service_sample_df.copy()
    _ = compute_experimental_charges_ratio(service_sample_df)
    _ = compute_charges_ratio_defined(service_sample_df)
    pd.testing.assert_frame_equal(service_sample_df, original_copy)


def test_feature_transformers_sklearn_interface(service_sample_df: pd.DataFrame) -> None:
    """Test that feature transformers conform to scikit-learn's fit/transform interface."""
    opt_counter = OptionalServiceCounter(include_subtotals=True)
    opt_counter.fit(service_sample_df)
    res_opt = opt_counter.transform(service_sample_df)

    assert "OptionalServiceCount" in res_opt.columns
    assert "TechnicalProtectionCount" in res_opt.columns
    assert "StreamingEntertainmentCount" in res_opt.columns

    auto_pay = AutomaticPaymentTransformer()
    auto_pay.fit(service_sample_df)
    res_pay = auto_pay.transform(service_sample_df)
    assert "AutomaticPayment" in res_pay.columns

    exp_ratio = ExperimentalChargesRatioTransformer()
    exp_ratio.fit(service_sample_df)
    res_ratio = exp_ratio.transform(service_sample_df)
    assert "ChargesRatio" in res_ratio.columns
    assert "ChargesRatioDefined" in res_ratio.columns
    feature_names = exp_ratio.get_feature_names_out()
    assert "ChargesRatio" in feature_names
    assert "ChargesRatioDefined" in feature_names
