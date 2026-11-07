"""Unit tests for dataset validation module."""

import pandas as pd
import pytest

from predictlab.data.validate import (
    DuplicateColumnsError,
    EmptyDatasetError,
    SchemaValidationError,
    TargetColumnMissingError,
    validate_raw_data,
)
from tests.fixtures.synthetic_data import get_synthetic_telco_data


def test_validate_valid_dataset():
    """Verify validation passes cleanly on a valid dataset."""
    df = get_synthetic_telco_data(num_rows=5)
    result = validate_raw_data(df)

    assert result.is_valid is True
    assert result.row_count == 5
    assert result.column_count == 21
    assert result.target_present is True
    assert result.target_classes == ["No", "Yes"]
    assert len(result.errors) == 0
    assert len(result.missing_columns) == 0


def test_validate_empty_dataset():
    """Verify empty dataframe detection."""
    df_empty = pd.DataFrame()

    result = validate_raw_data(df_empty, raise_on_error=False)
    assert result.is_valid is False
    assert any("empty" in err.lower() for err in result.errors)

    with pytest.raises(EmptyDatasetError):
        validate_raw_data(df_empty, raise_on_error=True)


def test_validate_missing_target():
    """Verify missing target column detection."""
    df = get_synthetic_telco_data(num_rows=5).drop(columns=["Churn"])

    result = validate_raw_data(df, raise_on_error=False)
    assert result.is_valid is False
    assert result.target_present is False
    assert any("target column 'churn' not found" in err.lower() for err in result.errors)

    with pytest.raises(TargetColumnMissingError):
        validate_raw_data(df, raise_on_error=True)


def test_validate_duplicate_columns():
    """Verify duplicate column names detection."""
    df = pd.DataFrame(
        [[1, 2, "Yes"], [3, 4, "No"]],
        columns=["tenure", "tenure", "Churn"],
    )

    result = validate_raw_data(df, expected_columns=["tenure", "Churn"], raise_on_error=False)
    assert result.is_valid is False
    assert "tenure" in result.duplicate_columns
    assert any("duplicate column headers" in err.lower() for err in result.errors)

    with pytest.raises(DuplicateColumnsError):
        validate_raw_data(df, expected_columns=["tenure", "Churn"], raise_on_error=True)


def test_validate_missing_schema_columns():
    """Verify missing expected schema columns detection."""
    df = get_synthetic_telco_data(num_rows=5).drop(columns=["tenure", "Contract"])

    result = validate_raw_data(df, raise_on_error=False)
    assert result.is_valid is False
    assert "tenure" in result.missing_columns
    assert "Contract" in result.missing_columns

    with pytest.raises(SchemaValidationError):
        validate_raw_data(df, raise_on_error=True)


def test_validate_extra_columns_warning():
    """Verify unexpected extra columns trigger a warning without failing validity."""
    df = get_synthetic_telco_data(num_rows=5)
    df["extra_feature_one"] = "abc"

    result = validate_raw_data(df, raise_on_error=False)
    assert result.is_valid is True
    assert "extra_feature_one" in result.extra_columns
    assert any("extra columns" in w.lower() for w in result.warnings)


def test_validate_duplicate_rows_and_ids():
    """Verify warnings for duplicate rows and duplicate IDs."""
    df = get_synthetic_telco_data(num_rows=2)
    # Duplicate first row
    df_with_dupes = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    result = validate_raw_data(df_with_dupes, raise_on_error=False)
    assert result.is_valid is True
    assert result.duplicate_rows == 1
    assert result.duplicate_ids == 1
    assert any("completely duplicated rows" in w.lower() for w in result.warnings)
    assert any("duplicate keys" in w.lower() for w in result.warnings)
