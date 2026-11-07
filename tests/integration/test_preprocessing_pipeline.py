"""Integration tests for leakage-free preprocessing pipeline and parameter invariance."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from predictlab.config import get_settings
from predictlab.data.load import load_raw_dataset
from predictlab.preprocessing.pipelines import (
    create_linear_preprocessor,
    create_tree_preprocessor,
    transform_to_dataframe,
)
from predictlab.preprocessing.split import split_dataset


@pytest.fixture
def partitioned_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load canonical dataset and create stratified 80/20 train/test partition."""
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, _ = split_dataset(df, save_manifest=False)
    return X_train, X_test, y_train, y_test


def test_leakage_invariance_extreme_test_values(partitioned_dataset: tuple) -> None:
    """Test that extreme test set values do not contaminate or modify training-fitted parameters.

    Leakage Prevention Proof:
    1. Fit preprocessor strictly on X_train.
    2. Record learned scaling means, variances, and imputer statistics.
    3. Transform standard X_train and record output.
    4. Construct a corrupted test set with extreme unphysical values (e.g. tenure=100,000, TotalCharges=10,000,000).
    5. Transform corrupted test set through the pipeline.
    6. Verify that learned training parameters remain bit-identical.
    7. Verify that re-transforming X_train produces identical outputs.
    """
    X_train, X_test, _, _ = partitioned_dataset

    pipeline = create_linear_preprocessor()
    pipeline.fit(X_train)

    # Extract learned scaler parameters
    col_trans = pipeline.named_steps["column_transform"]
    scaler = col_trans.named_transformers_["num"].named_steps["scaler"]
    imputer = col_trans.named_transformers_["num"].named_steps["imputer"]

    initial_mean = scaler.mean_.copy()
    initial_scale = scaler.scale_.copy()
    initial_imputer_stats = imputer.statistics_.copy()

    # Transform training data
    train_output_1 = transform_to_dataframe(pipeline, X_train)

    # Create corrupted test set with extreme outliers
    corrupted_test = X_test.copy()
    corrupted_test["tenure"] = 99999
    corrupted_test["MonthlyCharges"] = 88888.0
    corrupted_test["TotalCharges"] = "9999999.0"

    # Transform corrupted test data
    _ = transform_to_dataframe(pipeline, corrupted_test)

    # Assert learned parameters are unchanged
    np.testing.assert_array_equal(scaler.mean_, initial_mean)
    np.testing.assert_array_equal(scaler.scale_, initial_scale)
    np.testing.assert_array_equal(imputer.statistics_, initial_imputer_stats)

    # Assert re-transforming X_train produces bit-identical results
    train_output_2 = transform_to_dataframe(pipeline, X_train)
    pd.testing.assert_frame_equal(train_output_1, train_output_2)


def test_zero_tenure_whitespace_split_handling(partitioned_dataset: tuple) -> None:
    """Test that both train and test partitions properly clean zero-tenure whitespace records."""
    X_train, X_test, _, _ = partitioned_dataset

    # Verify that the 11 zero-tenure records exist across train and test
    train_ws_count = int((X_train["tenure"] == 0).sum())
    test_ws_count = int((X_test["tenure"] == 0).sum())

    assert train_ws_count + test_ws_count == 11
    assert train_ws_count > 0
    assert test_ws_count > 0

    # Fit and transform linear pipeline
    linear_prep = create_linear_preprocessor()
    linear_prep.fit(X_train)

    train_trans = transform_to_dataframe(linear_prep, X_train)
    test_trans = transform_to_dataframe(linear_prep, X_test)

    assert not train_trans.isna().any().any()
    assert not test_trans.isna().any().any()

    # Fit and transform tree pipeline
    tree_prep = create_tree_preprocessor()
    tree_prep.fit(X_train)

    tree_train_trans = transform_to_dataframe(tree_prep, X_train)
    tree_test_trans = transform_to_dataframe(tree_prep, X_test)

    assert not tree_train_trans.isna().any().any()
    assert not tree_test_trans.isna().any().any()


def test_raw_csv_immutability_verification() -> None:
    """Verify that end-to-end preprocessing execution leaves the raw dataset 100% byte-unmodified."""
    settings = get_settings()
    expected_sha256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

    hasher = hashlib.sha256()
    with open(settings.raw_data_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    assert hasher.hexdigest() == expected_sha256
