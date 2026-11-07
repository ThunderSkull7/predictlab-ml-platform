"""Integration tests for end-to-end data loading and validation workflow."""

from pathlib import Path

from predictlab.config import get_settings
from predictlab.data.load import load_raw_dataset
from predictlab.data.validate import validate_raw_data
from tests.fixtures.synthetic_data import get_synthetic_telco_data


def test_synthetic_end_to_end_pipeline(tmp_path: Path):
    """Verify loading synthetic data from disk and validating schema end-to-end."""
    df_synthetic = get_synthetic_telco_data(num_rows=25)
    csv_path = tmp_path / "test_pipeline.csv"
    df_synthetic.to_csv(csv_path, index=False)

    df_loaded = load_raw_dataset(file_path=csv_path)
    result = validate_raw_data(df_loaded)

    assert result.is_valid is True
    assert result.row_count == 25
    assert result.column_count == 21
    assert result.target_present is True
    assert len(result.errors) == 0


def test_actual_raw_dataset_pipeline():
    """Verify loading and validating the actual project raw dataset if present."""
    settings = get_settings()
    if not settings.raw_data_path.exists():
        return

    df_raw = load_raw_dataset()
    result = validate_raw_data(df_raw)

    assert result.is_valid is True
    assert result.row_count == 7043
    assert result.column_count == 21
    assert result.target_present is True
    assert set(result.target_classes) == {"No", "Yes"}
    assert result.duplicate_rows == 0
    assert result.duplicate_ids == 0
    assert len(result.missing_columns) == 0
    assert len(result.errors) == 0
