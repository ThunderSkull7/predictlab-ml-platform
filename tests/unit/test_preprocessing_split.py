"""Unit tests for dataset partitioning, stratification, and manifest persistence."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from predictlab.data.load import load_raw_dataset
from predictlab.data.validate import (
    DataValidationError,
    EmptyDatasetError,
    TargetColumnMissingError,
)
from predictlab.preprocessing.schema import ID_COLUMN, TARGET_COLUMN
from predictlab.preprocessing.split import (
    SplitManifest,
    load_split_manifest,
    split_dataset,
)


@pytest.fixture
def canonical_df() -> pd.DataFrame:
    """Load canonical raw dataset."""
    return load_raw_dataset()


def test_split_dataset_dimensions_and_proportions(canonical_df: pd.DataFrame) -> None:
    """Test that split produces exact 80/20 train/test proportions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = Path(tmp_dir) / "test_manifest.json"
        X_train, X_test, y_train, y_test, manifest = split_dataset(
            canonical_df,
            test_size=0.20,
            random_state=42,
            manifest_path=manifest_path,
        )

        assert len(X_train) == 5634
        assert len(X_test) == 1409
        assert len(y_train) == 5634
        assert len(y_test) == 1409
        assert len(X_train) + len(X_test) == len(canonical_df)

        # ID and Target must be excluded from feature matrices
        assert ID_COLUMN not in X_train.columns
        assert ID_COLUMN not in X_test.columns
        assert TARGET_COLUMN not in X_train.columns
        assert TARGET_COLUMN not in X_test.columns

        # Target must be binary integers {0, 1}
        assert set(y_train.unique()) == {0, 1}
        assert set(y_test.unique()) == {0, 1}


def test_split_dataset_stratification_fidelity(canonical_df: pd.DataFrame) -> None:
    """Test that train and test partitions preserve target churn proportions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = Path(tmp_dir) / "test_manifest.json"
        _, _, y_train, y_test, manifest = split_dataset(
            canonical_df,
            test_size=0.20,
            random_state=42,
            manifest_path=manifest_path,
        )

        overall_churn = (canonical_df[TARGET_COLUMN] == "Yes").mean()
        train_churn = y_train.mean()
        test_churn = y_test.mean()

        assert pytest.approx(train_churn, abs=0.005) == overall_churn
        assert pytest.approx(test_churn, abs=0.005) == overall_churn
        assert manifest.overall_churn_rate_pct == round(overall_churn * 100.0, 4)


def test_split_dataset_no_id_leakage_or_overlap(canonical_df: pd.DataFrame) -> None:
    """Test that train and test customer IDs have zero intersection."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = Path(tmp_dir) / "test_manifest.json"
        _, _, _, _, manifest = split_dataset(
            canonical_df,
            manifest_path=manifest_path,
        )

        train_ids = set(manifest.train_customer_ids)
        test_ids = set(manifest.test_customer_ids)

        assert len(train_ids) == 5634
        assert len(test_ids) == 1409
        assert len(train_ids.intersection(test_ids)) == 0


def test_split_dataset_reproducibility(canonical_df: pd.DataFrame) -> None:
    """Test that identical seeds yield bit-for-bit identical splits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p1 = Path(tmp_dir) / "m1.json"
        p2 = Path(tmp_dir) / "m2.json"
        X_tr1, X_te1, y_tr1, y_te1, _ = split_dataset(canonical_df, random_state=42, manifest_path=p1)
        X_tr2, X_te2, y_tr2, y_te2, _ = split_dataset(canonical_df, random_state=42, manifest_path=p2)

        pd.testing.assert_frame_equal(X_tr1, X_tr2)
        pd.testing.assert_frame_equal(X_te1, X_te2)
        pd.testing.assert_series_equal(y_tr1, y_tr2)
        pd.testing.assert_series_equal(y_te1, y_te2)


def test_split_manifest_persistence_and_loading(canonical_df: pd.DataFrame) -> None:
    """Test persisting and deserializing the lightweight SplitManifest."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = Path(tmp_dir) / "split_manifest.json"
        _, _, _, _, original_manifest = split_dataset(
            canonical_df,
            manifest_path=manifest_path,
        )

        assert manifest_path.exists()
        loaded_manifest = load_split_manifest(manifest_path)

        assert isinstance(loaded_manifest, SplitManifest)
        assert loaded_manifest.total_samples == original_manifest.total_samples
        assert loaded_manifest.train_samples == 5634
        assert loaded_manifest.test_samples == 1409
        assert loaded_manifest.train_customer_ids == original_manifest.train_customer_ids


def test_split_dataset_raises_on_invalid_inputs() -> None:
    """Test exception raising for empty data, missing target, and duplicate IDs."""
    with pytest.raises(EmptyDatasetError):
        split_dataset(pd.DataFrame())

    df_no_target = pd.DataFrame({"customerID": ["C1", "C2"], "tenure": [1, 2]})
    with pytest.raises(TargetColumnMissingError):
        split_dataset(df_no_target)

    df_dup_ids = pd.DataFrame(
        {
            "customerID": ["C1", "C1"],  # Duplicate ID
            "Churn": ["No", "Yes"],
            "tenure": [1, 2],
        }
    )
    with pytest.raises(DataValidationError):
        split_dataset(df_dup_ids)
