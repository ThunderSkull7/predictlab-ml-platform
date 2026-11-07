"""Unit tests for TotalCharges data cleaning and structural validation."""

import numpy as np
import pandas as pd
import pytest

from predictlab.data.validate import DataValidationError
from predictlab.preprocessing.cleaning import (
    TotalChargesValidationError,
    clean_total_charges,
)
from predictlab.preprocessing.transformers import TotalChargesCleaner


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture providing standard valid subscriber records including zero tenure signups."""
    return pd.DataFrame(
        {
            "tenure": [0, 1, 12, 24, 0],
            "TotalCharges": [" ", "50.25", "600.00", "1200.50", "   "],
            "MonthlyCharges": [25.0, 50.25, 50.0, 50.0, 30.0],
            "Churn": ["No", "No", "Yes", "No", "No"],
        }
    )


def test_clean_total_charges_valid_imputation(sample_valid_df: pd.DataFrame) -> None:
    """Test that whitespace entries at tenure == 0 are correctly converted to 0.0."""
    result = clean_total_charges(sample_valid_df)

    assert result is not sample_valid_df
    assert sample_valid_df["TotalCharges"].iloc[0] == " "  # Immutability
    assert result["TotalCharges"].iloc[0] == 0.0
    assert result["TotalCharges"].iloc[1] == 50.25
    assert result["TotalCharges"].iloc[2] == 600.00
    assert result["TotalCharges"].iloc[3] == 1200.50
    assert result["TotalCharges"].iloc[4] == 0.0
    assert result["TotalCharges"].dtype == np.float64
    assert np.isfinite(result["TotalCharges"]).all()


def test_clean_total_charges_immutability(sample_valid_df: pd.DataFrame) -> None:
    """Test that the input DataFrame is strictly untouched."""
    original_copy = sample_valid_df.copy()
    _ = clean_total_charges(sample_valid_df)
    pd.testing.assert_frame_equal(sample_valid_df, original_copy)


def test_clean_total_charges_idempotent(sample_valid_df: pd.DataFrame) -> None:
    """Test that applying cleaning multiple times produces identical results."""
    res1 = clean_total_charges(sample_valid_df)
    res2 = clean_total_charges(res1)
    pd.testing.assert_frame_equal(res1, res2)


def test_clean_total_charges_raises_on_positive_tenure_missing() -> None:
    """Test that missing TotalCharges for accounts with tenure > 0 raises TotalChargesValidationError."""
    invalid_df = pd.DataFrame(
        {
            "tenure": [12, 0],
            "TotalCharges": [" ", " "],  # First record has positive tenure!
        }
    )
    with pytest.raises(TotalChargesValidationError) as exc_info:
        clean_total_charges(invalid_df)

    assert "positive tenure" in str(exc_info.value).lower()
    assert "tenure > 0" in str(exc_info.value)


def test_clean_total_charges_raises_on_malformed_text() -> None:
    """Test that unparseable non-numeric strings raise TotalChargesValidationError."""
    invalid_df = pd.DataFrame(
        {
            "tenure": [10],
            "TotalCharges": ["invalid_numeric_value"],
        }
    )
    with pytest.raises(TotalChargesValidationError) as exc_info:
        clean_total_charges(invalid_df)

    assert "unparseable non-numeric" in str(exc_info.value).lower()


def test_clean_total_charges_empty_dataframe_raises() -> None:
    """Test that passing an empty DataFrame raises DataValidationError."""
    with pytest.raises(DataValidationError):
        clean_total_charges(pd.DataFrame())


def test_clean_total_charges_missing_columns_raises() -> None:
    """Test that missing required columns raises DataValidationError."""
    df_missing_tc = pd.DataFrame({"tenure": [1, 2]})
    with pytest.raises(DataValidationError):
        clean_total_charges(df_missing_tc)

    df_missing_tenure = pd.DataFrame({"TotalCharges": ["100.0"]})
    with pytest.raises(DataValidationError):
        clean_total_charges(df_missing_tenure, validate_zero_tenure=True)


def test_total_charges_cleaner_transformer(sample_valid_df: pd.DataFrame) -> None:
    """Test scikit-learn compatible transformer interface for TotalChargesCleaner."""
    transformer = TotalChargesCleaner()
    transformer.fit(sample_valid_df)
    transformed = transformer.transform(sample_valid_df)

    assert transformed["TotalCharges"].iloc[0] == 0.0
    assert transformed["TotalCharges"].iloc[4] == 0.0
    feature_names = transformer.get_feature_names_out()
    assert list(feature_names) == list(sample_valid_df.columns)
