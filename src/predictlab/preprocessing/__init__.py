"""PredictLab Preprocessing Package.

Provides leakage-free dataset splitting, deterministic data cleaning, domain feature engineering,
scikit-learn compatible custom transformers, and production-ready ColumnTransformer pipelines.
"""

from predictlab.preprocessing.cleaning import (
    TotalChargesValidationError,
    clean_total_charges,
)
from predictlab.preprocessing.feature_engineering import (
    compute_automatic_payment_flag,
    compute_charges_ratio_defined,
    compute_experimental_charges_ratio,
    compute_optional_service_count,
    compute_streaming_entertainment_count,
    compute_technical_protection_count,
)
from predictlab.preprocessing.pipelines import (
    BASELINE_CATEGORICAL_FEATURES,
    BASELINE_NUMERICAL_FEATURES,
    build_column_transformer,
    create_experimental_preprocessor,
    create_linear_preprocessor,
    create_tree_preprocessor,
    get_preprocessor_feature_names,
    load_preprocessor,
    save_preprocessor,
    transform_to_dataframe,
)
from predictlab.preprocessing.schema import (
    ALLOWED_CATEGORICAL_VALUES,
    AUTOMATIC_PAYMENT_METHODS,
    ID_COLUMN,
    INVERSE_TARGET_MAPPING,
    NEGATIVE_CLASS,
    OPTIONAL_SERVICE_COLUMNS,
    POSITIVE_CLASS,
    RAW_CATEGORICAL_COLUMNS,
    RAW_FEATURE_COLUMNS,
    RAW_NUMERICAL_COLUMNS,
    STREAMING_ENTERTAINMENT_COLUMNS,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TECHNICAL_PROTECTION_COLUMNS,
    InvalidCategoryValueError,
    TargetContractViolationError,
    validate_dataset_contract,
)
from predictlab.preprocessing.split import (
    SplitManifest,
    load_split_manifest,
    split_dataset,
)
from predictlab.preprocessing.transformers import (
    AutomaticPaymentTransformer,
    ColumnDropper,
    ExperimentalChargesRatioTransformer,
    OptionalServiceCounter,
    TotalChargesCleaner,
)

__all__ = [
    # Schema & Contracts
    "ID_COLUMN",
    "TARGET_COLUMN",
    "POSITIVE_CLASS",
    "NEGATIVE_CLASS",
    "TARGET_MAPPING",
    "INVERSE_TARGET_MAPPING",
    "RAW_NUMERICAL_COLUMNS",
    "RAW_CATEGORICAL_COLUMNS",
    "RAW_FEATURE_COLUMNS",
    "OPTIONAL_SERVICE_COLUMNS",
    "TECHNICAL_PROTECTION_COLUMNS",
    "STREAMING_ENTERTAINMENT_COLUMNS",
    "AUTOMATIC_PAYMENT_METHODS",
    "ALLOWED_CATEGORICAL_VALUES",
    "validate_dataset_contract",
    "InvalidCategoryValueError",
    "TargetContractViolationError",
    # Cleaning
    "clean_total_charges",
    "TotalChargesValidationError",
    # Feature Engineering
    "compute_optional_service_count",
    "compute_technical_protection_count",
    "compute_streaming_entertainment_count",
    "compute_automatic_payment_flag",
    "compute_experimental_charges_ratio",
    "compute_charges_ratio_defined",
    # Transformers
    "TotalChargesCleaner",
    "OptionalServiceCounter",
    "AutomaticPaymentTransformer",
    "ExperimentalChargesRatioTransformer",
    "ColumnDropper",
    # Splitting
    "split_dataset",
    "load_split_manifest",
    "SplitManifest",
    # Pipelines
    "BASELINE_NUMERICAL_FEATURES",
    "BASELINE_CATEGORICAL_FEATURES",
    "build_column_transformer",
    "create_linear_preprocessor",
    "create_tree_preprocessor",
    "create_experimental_preprocessor",
    "get_preprocessor_feature_names",
    "transform_to_dataframe",
    "save_preprocessor",
    "load_preprocessor",
]
