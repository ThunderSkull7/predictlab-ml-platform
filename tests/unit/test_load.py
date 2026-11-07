"""Unit tests for dataset loading module."""

from pathlib import Path

import pytest

from predictlab.data.load import load_raw_dataset
from predictlab.data.validate import DataValidationError
from tests.fixtures.synthetic_data import get_synthetic_telco_data


def test_load_valid_csv(tmp_path: Path):
    """Verify loading a valid synthetic CSV file."""
    df_synthetic = get_synthetic_telco_data(num_rows=10)
    file_path = tmp_path / "synthetic.csv"
    df_synthetic.to_csv(file_path, index=False)

    loaded_df = load_raw_dataset(file_path=file_path)

    assert loaded_df.shape == (10, 21)
    assert list(loaded_df.columns) == list(df_synthetic.columns)
    assert loaded_df["customerID"].iloc[0] == "TEST-0000"


def test_load_missing_file_raises_error(tmp_path: Path):
    """Verify loading a nonexistent file raises FileNotFoundError."""
    missing_file = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        load_raw_dataset(file_path=missing_file)


def test_load_empty_file_raises_error(tmp_path: Path):
    """Verify loading an empty (0 byte) file raises DataValidationError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()

    with pytest.raises(DataValidationError, match="Raw dataset file is empty"):
        load_raw_dataset(file_path=empty_file)


def test_load_directory_path_raises_error(tmp_path: Path):
    """Verify passing a directory instead of a file raises DataValidationError."""
    dir_path = tmp_path / "some_dir"
    dir_path.mkdir()

    with pytest.raises(DataValidationError, match="not a regular file"):
        load_raw_dataset(file_path=dir_path)


def test_load_corrupted_csv_raises_error(tmp_path: Path):
    """Verify header-only empty CSV raises DataValidationError."""
    file_path = tmp_path / "corrupted.csv"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(DataValidationError):
        load_raw_dataset(file_path=file_path)


def test_load_csv_with_duplicate_headers_raises_error(tmp_path: Path):
    """Verify CSV file containing duplicate header names raises DuplicateColumnsError."""
    from predictlab.data.validate import DuplicateColumnsError

    file_path = tmp_path / "dupe_header.csv"
    file_path.write_text("colA,colB,colA\n1,2,3\n", encoding="utf-8")

    with pytest.raises(DuplicateColumnsError, match="duplicate column names"):
        load_raw_dataset(file_path=file_path)
