"""Unit tests for pipeline assembly, ColumnTransformer configuration, and serialization."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictlab.data.load import load_raw_dataset
from predictlab.preprocessing.pipelines import (
    create_experimental_preprocessor,
    create_linear_preprocessor,
    create_tree_preprocessor,
    get_preprocessor_feature_names,
    load_preprocessor,
    save_preprocessor,
    transform_to_dataframe,
)
from predictlab.preprocessing.split import split_dataset


@pytest.fixture
def train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Provide canonical partitioned dataset."""
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, _ = split_dataset(df, save_manifest=False)
    return X_train, X_test, y_train, y_test


def test_linear_preprocessor_pipeline(train_test_data: tuple) -> None:
    """Test linear-model preprocessor pipeline fitting, transformation, and dimensions."""
    X_train, X_test, _, _ = train_test_data

    preprocessor = create_linear_preprocessor()
    preprocessor.fit(X_train)

    X_train_trans = transform_to_dataframe(preprocessor, X_train)
    X_test_trans = transform_to_dataframe(preprocessor, X_test)

    assert X_train_trans.shape == (5634, 47)
    assert X_test_trans.shape == (1409, 47)
    assert not X_train_trans.isna().any().any()
    assert not X_test_trans.isna().any().any()

    # Verify scaling on numerical features
    assert pytest.approx(X_train_trans["num__tenure"].mean(), abs=1e-2) == 0.0
    assert pytest.approx(X_train_trans["num__tenure"].std(), abs=1e-2) == 1.0


def test_tree_preprocessor_pipeline(train_test_data: tuple) -> None:
    """Test tree-model preprocessor preserves natural numeric scales without standard scaling."""
    X_train, X_test, _, _ = train_test_data

    preprocessor = create_tree_preprocessor()
    preprocessor.fit(X_train)

    X_train_trans = transform_to_dataframe(preprocessor, X_train)
    X_test_trans = transform_to_dataframe(preprocessor, X_test)

    assert X_train_trans.shape == (5634, 47)
    assert X_test_trans.shape == (1409, 47)

    # Verify natural units are preserved
    assert X_train_trans["num__tenure"].min() == 0.0
    assert X_train_trans["num__tenure"].max() == 72.0
    assert X_train_trans["num__OptionalServiceCount"].min() == 0.0
    assert X_train_trans["num__OptionalServiceCount"].max() == 6.0


def test_contract_one_hot_encoding(train_test_data: tuple) -> None:
    """Test that Contract is strictly One-Hot encoded by default."""
    X_train, _, _, _ = train_test_data

    preprocessor = create_linear_preprocessor()
    preprocessor.fit(X_train)

    feature_names = get_preprocessor_feature_names(preprocessor)
    contract_features = [f for f in feature_names if "Contract" in f]

    assert len(contract_features) == 3
    assert "cat__Contract_Month-to-month" in contract_features
    assert "cat__Contract_One year" in contract_features
    assert "cat__Contract_Two year" in contract_features


def test_unknown_category_handling_at_inference(train_test_data: tuple) -> None:
    """Test that unseen categorical values in test/inference data do not crash the pipeline."""
    X_train, _, _, _ = train_test_data

    preprocessor = create_linear_preprocessor()
    preprocessor.fit(X_train)

    # Create synthetic inference record with an unseen category
    unseen_record = X_train.iloc[0:1].copy()
    unseen_record["PaymentMethod"] = "Unseen Payment Vendor"

    # Must transform without error (OneHotEncoder handle_unknown='ignore')
    transformed = transform_to_dataframe(preprocessor, unseen_record)
    assert transformed.shape == (1, 47)

    # PaymentMethod columns should all be 0 for this record
    pm_cols = [c for c in transformed.columns if "PaymentMethod" in c]
    assert (transformed[pm_cols].values == 0).all()


def test_preprocessor_serialization(train_test_data: tuple) -> None:
    """Test serializing and deserializing fitted preprocessor pipeline with joblib."""
    X_train, X_test, _, _ = train_test_data

    preprocessor = create_linear_preprocessor()
    preprocessor.fit(X_train)
    expected_output = preprocessor.transform(X_test)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "preprocessor.joblib"
        save_preprocessor(preprocessor, save_path)
        assert save_path.exists()

        loaded_pipeline = load_preprocessor(save_path)
        loaded_output = loaded_pipeline.transform(X_test)

        np.testing.assert_allclose(expected_output, loaded_output)


def test_experimental_preprocessor_pipeline(train_test_data: tuple) -> None:
    """Test experimental pipeline with ChargesRatio, ChargesRatioDefined, AutomaticPayment, and subtotals."""
    X_train, _, _, _ = train_test_data

    exp_pipeline = create_experimental_preprocessor(
        include_charges_ratio=True,
        include_automatic_payment=True,
        include_service_subtotals=True,
    )
    exp_pipeline.fit(X_train)
    transformed = transform_to_dataframe(exp_pipeline, X_train)

    assert transformed.shape == (5634, 52)
    feature_names = get_preprocessor_feature_names(exp_pipeline)
    assert "num__ChargesRatio" in feature_names
    assert "num__ChargesRatioDefined" in feature_names
    assert "num__AutomaticPayment" in feature_names
    assert "num__TechnicalProtectionCount" in feature_names
    assert "num__StreamingEntertainmentCount" in feature_names


def test_experimental_preprocessor_zero_tenure_imputation_and_leakage(train_test_data: tuple) -> None:
    """Test experimental pipeline handles zero-tenure undefined ratios via training-only imputer."""
    X_train, X_test, _, _ = train_test_data

    # Use unscaled experimental pipeline to inspect raw imputed values cleanly
    exp_pipeline = create_experimental_preprocessor(
        include_charges_ratio=True,
        include_automatic_payment=False,
        include_service_subtotals=False,
        scale_numeric=False,
    )
    exp_pipeline.fit(X_train)

    # Imputer statistics learned strictly from training partition
    col_trans = exp_pipeline.named_steps["column_transform"]
    imputer = col_trans.named_transformers_["num"].named_steps["imputer"]
    num_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "OptionalServiceCount",
        "ChargesRatio",
        "ChargesRatioDefined",
    ]
    imputer_stats = dict(zip(num_features, imputer.statistics_, strict=True))

    # The training median ChargesRatio is ~1.0000
    assert pytest.approx(imputer_stats["ChargesRatio"], abs=0.05) == 1.0
    assert imputer_stats["ChargesRatioDefined"] == 1.0  # Dominant class is defined

    # Transform test set (containing zero-tenure records)
    test_trans = transform_to_dataframe(exp_pipeline, X_test)
    assert not test_trans.isna().any().any()

    # Identify zero tenure test records
    test_zero_mask = X_test["tenure"] == 0
    assert test_zero_mask.sum() > 0

    # ChargesRatioDefined must be 0.0 for zero-tenure records and 1.0 for positive tenure
    assert (test_trans.loc[test_zero_mask, "num__ChargesRatioDefined"] == 0.0).all()
    assert (test_trans.loc[~test_zero_mask, "num__ChargesRatioDefined"] == 1.0).all()

    # The missing ChargesRatio at tenure==0 in test set was imputed with training median
    for val in test_trans.loc[test_zero_mask, "num__ChargesRatio"]:
        assert pytest.approx(val, abs=1e-4) == imputer_stats["ChargesRatio"]

    # Verify parameter invariance: transforming corrupted test data does not alter training imputer stats
    corrupted_test = X_test.copy()
    corrupted_test["tenure"] = 99999
    corrupted_test["MonthlyCharges"] = 99999.0
    corrupted_test["TotalCharges"] = "9999999.0"

    _ = transform_to_dataframe(exp_pipeline, corrupted_test)
    np.testing.assert_array_equal(imputer.statistics_, list(imputer_stats.values()))
