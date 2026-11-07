"""Pipeline construction module for PredictLab preprocessing."""

from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictlab.config import get_settings
from predictlab.logging_config import get_logger
from predictlab.preprocessing.schema import (
    RAW_CATEGORICAL_COLUMNS,
    RAW_NUMERICAL_COLUMNS,
)
from predictlab.preprocessing.transformers import (
    AutomaticPaymentTransformer,
    ColumnDropper,
    ExperimentalChargesRatioTransformer,
    OptionalServiceCounter,
    TotalChargesCleaner,
)

logger = get_logger(__name__)

# Baseline numerical features after cleaning and feature engineering
BASELINE_NUMERICAL_FEATURES: tuple[str, ...] = RAW_NUMERICAL_COLUMNS + ("OptionalServiceCount",)

# Baseline categorical features (16 nominal categoricals including Contract and SeniorCitizen)
BASELINE_CATEGORICAL_FEATURES: tuple[str, ...] = RAW_CATEGORICAL_COLUMNS


def build_column_transformer(
    numerical_cols: tuple[str, ...] = BASELINE_NUMERICAL_FEATURES,
    categorical_cols: tuple[str, ...] = BASELINE_CATEGORICAL_FEATURES,
    scale_numeric: bool = True,
    handle_unknown_cat: Literal["error", "ignore", "infrequent_if_exist"] = "ignore",
    drop_first_cat: bool = False,
    verbose_feature_names_out: bool = True,
) -> ColumnTransformer:
    """Build a ColumnTransformer for numerical and categorical feature processing.

    Parameters
    ----------
    numerical_cols : tuple[str, ...]
        Tuple of continuous/discrete numeric column names.
    categorical_cols : tuple[str, ...]
        Tuple of nominal categorical column names.
    scale_numeric : bool
        Whether to apply StandardScaler to numerical features.
    handle_unknown_cat : str
        Strategy for unseen categories at inference/test time ('ignore' prevents crashes).
    drop_first_cat : bool
        Whether to drop the first category in OneHotEncoder (useful for unregularized linear models).
    verbose_feature_names_out : bool
        Whether to prefix feature names with transformer branch names.

    Returns
    -------
    ColumnTransformer
        Configured scikit-learn ColumnTransformer.
    """
    # 1. Numerical branch
    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_pipeline = Pipeline(steps=num_steps)

    # 2. Categorical branch
    drop_strategy = "first" if drop_first_cat else None
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop=drop_strategy,
                    handle_unknown=handle_unknown_cat,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, list(numerical_cols)),
            ("cat", cat_pipeline, list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=verbose_feature_names_out,
    )
    return preprocessor


def create_linear_preprocessor(
    scale_numeric: bool = True,
    drop_first_cat: bool = False,
    include_customer_id_dropper: bool = True,
) -> Pipeline:
    """Factory function creating a linear-model-compatible preprocessing pipeline.

    Architecture:
    1. TotalChargesCleaner: Converts whitespace to NaN, validates zero-tenure missingness, imputes 0.0.
    2. OptionalServiceCounter: Engineers OptionalServiceCount (0 to 6).
    3. ColumnDropper (optional): Safeguard dropping customerID if passed into transform.
    4. ColumnTransformer:
       - Numerical: Median imputation + StandardScaler.
       - Categorical: Most-frequent imputation + OneHotEncoder(handle_unknown='ignore', sparse=False).

    Returns
    -------
    Pipeline
        Composed scikit-learn Pipeline.
    """
    steps: list[tuple[str, Any]] = [
        ("clean_total_charges", TotalChargesCleaner()),
        ("engineer_optional_services", OptionalServiceCounter()),
    ]
    if include_customer_id_dropper:
        steps.append(("drop_id", ColumnDropper(columns_to_drop=["customerID"])))

    column_trans = build_column_transformer(
        numerical_cols=BASELINE_NUMERICAL_FEATURES,
        categorical_cols=BASELINE_CATEGORICAL_FEATURES,
        scale_numeric=scale_numeric,
        drop_first_cat=drop_first_cat,
    )
    steps.append(("column_transform", column_trans))

    return Pipeline(steps=steps)


def create_tree_preprocessor(
    include_customer_id_dropper: bool = True,
) -> Pipeline:
    """Factory function creating a tree-model-compatible preprocessing pipeline.

    Architecture:
    1. TotalChargesCleaner: Converts whitespace to NaN, validates zero-tenure missingness, imputes 0.0.
    2. OptionalServiceCounter: Engineers OptionalServiceCount (0 to 6).
    3. ColumnDropper (optional): Safeguard dropping customerID if passed into transform.
    4. ColumnTransformer:
       - Numerical: Median imputation (NO standard scaling!).
       - Categorical: Most-frequent imputation + OneHotEncoder(handle_unknown='ignore', sparse=False).

    Returns
    -------
    Pipeline
        Composed scikit-learn Pipeline.
    """
    steps: list[tuple[str, Any]] = [
        ("clean_total_charges", TotalChargesCleaner()),
        ("engineer_optional_services", OptionalServiceCounter()),
    ]
    if include_customer_id_dropper:
        steps.append(("drop_id", ColumnDropper(columns_to_drop=["customerID"])))

    column_trans = build_column_transformer(
        numerical_cols=BASELINE_NUMERICAL_FEATURES,
        categorical_cols=BASELINE_CATEGORICAL_FEATURES,
        scale_numeric=False,  # No standard scaling for tree models
        drop_first_cat=False,
    )
    steps.append(("column_transform", column_trans))

    return Pipeline(steps=steps)


def create_experimental_preprocessor(
    include_charges_ratio: bool = True,
    include_automatic_payment: bool = True,
    include_service_subtotals: bool = False,
    scale_numeric: bool = True,
) -> Pipeline:
    """Factory function creating an experimental preprocessing pipeline for comparative evaluation.

    Allows isolated evaluation of experimental features:
    - ChargesRatio (with explicit zero-tenure handling)
    - AutomaticPayment binary indicator
    - TechnicalProtectionCount and StreamingEntertainmentCount subtotals

    Returns
    -------
    Pipeline
        Experimental scikit-learn Pipeline.
    """
    steps: list[tuple[str, Any]] = [
        ("clean_total_charges", TotalChargesCleaner()),
        (
            "engineer_optional_services",
            OptionalServiceCounter(include_subtotals=include_service_subtotals),
        ),
    ]

    exp_num_cols = list(BASELINE_NUMERICAL_FEATURES)
    if include_service_subtotals:
        exp_num_cols.extend(["TechnicalProtectionCount", "StreamingEntertainmentCount"])

    if include_charges_ratio:
        steps.append(
            (
                "engineer_charges_ratio",
                ExperimentalChargesRatioTransformer(include_defined_indicator=True),
            )
        )
        exp_num_cols.append("ChargesRatio")
        exp_num_cols.append("ChargesRatioDefined")

    exp_cat_cols = list(BASELINE_CATEGORICAL_FEATURES)
    if include_automatic_payment:
        steps.append(("engineer_auto_payment", AutomaticPaymentTransformer()))
        exp_num_cols.append("AutomaticPayment")

    steps.append(("drop_id", ColumnDropper(columns_to_drop=["customerID"])))

    column_trans = build_column_transformer(
        numerical_cols=tuple(exp_num_cols),
        categorical_cols=tuple(exp_cat_cols),
        scale_numeric=scale_numeric,
    )
    steps.append(("column_transform", column_trans))

    return Pipeline(steps=steps)


def get_preprocessor_feature_names(
    pipeline: Pipeline,
    strip_prefix: bool = False,
) -> list[str]:
    """Extract transformed feature names from a fitted preprocessing Pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted preprocessing pipeline.
    strip_prefix : bool
        Whether to strip 'num__' and 'cat__' branch prefixes.

    Returns
    -------
    list[str]
        List of output feature column names.
    """
    if "column_transform" not in pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'column_transform' step.")

    col_trans: ColumnTransformer = pipeline.named_steps["column_transform"]
    feature_names = col_trans.get_feature_names_out().tolist()

    if strip_prefix:
        feature_names = [f.split("__", 1)[-1] if "__" in f else f for f in feature_names]

    return feature_names


def transform_to_dataframe(
    pipeline: Pipeline,
    X: pd.DataFrame,
    index: pd.Index | None = None,
    strip_prefix: bool = False,
) -> pd.DataFrame:
    """Transform input DataFrame and return output as a labeled Pandas DataFrame.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted preprocessing pipeline.
    X : pd.DataFrame
        Input feature matrix.
    index : pd.Index, optional
        Index for output DataFrame (defaults to X.index).
    strip_prefix : bool
        Whether to strip column transformer prefixes.

    Returns
    -------
    pd.DataFrame
        Transformed data as a DataFrame with explicit feature column names.
    """
    transformed_array = pipeline.transform(X)
    feature_names = get_preprocessor_feature_names(pipeline, strip_prefix=strip_prefix)
    idx = index if index is not None else (X.index if isinstance(X, pd.DataFrame) else None)
    return pd.DataFrame(transformed_array, columns=feature_names, index=idx)


def save_preprocessor(pipeline: Pipeline, file_path: Path | str | None = None) -> Path:
    """Serialize fitted preprocessor pipeline to disk via joblib."""
    settings = get_settings()
    out_path = Path(file_path) if file_path is not None else settings.artifacts_dir / "preprocessor.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    logger.info("Saved serialized preprocessor pipeline to: %s", out_path.resolve())
    return out_path


def load_preprocessor(file_path: Path | str | None = None) -> Pipeline:
    """Deserialize fitted preprocessor pipeline from disk."""
    settings = get_settings()
    in_path = Path(file_path) if file_path is not None else settings.artifacts_dir / "preprocessor.joblib"
    if not in_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at: {in_path.resolve()}")
    pipeline = joblib.load(in_path)
    logger.info("Loaded preprocessor pipeline from: %s", in_path.resolve())
    return pipeline
