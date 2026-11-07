"""Schema contracts and feature group definitions for PredictLab preprocessing."""

from collections.abc import Sequence

import pandas as pd

from predictlab.data.validate import (
    DataValidationError,
    DuplicateColumnsError,
    EmptyDatasetError,
    SchemaValidationError,
    TargetColumnMissingError,
)

# Core identifier and target definitions
ID_COLUMN: str = "customerID"
TARGET_COLUMN: str = "Churn"
POSITIVE_CLASS: str = "Yes"
NEGATIVE_CLASS: str = "No"

# Canonical target binary encoding mapping
TARGET_MAPPING: dict[str, int] = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}
INVERSE_TARGET_MAPPING: dict[int, str] = {0: NEGATIVE_CLASS, 1: POSITIVE_CLASS}

# Raw numerical features present in input data
RAW_NUMERICAL_COLUMNS: tuple[str, ...] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

# Raw categorical features present in input data
RAW_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)

# All 19 raw predictive features (excluding customerID and Churn)
RAW_FEATURE_COLUMNS: tuple[str, ...] = RAW_NUMERICAL_COLUMNS + RAW_CATEGORICAL_COLUMNS

# Feature engineering input column sets
OPTIONAL_SERVICE_COLUMNS: tuple[str, ...] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

TECHNICAL_PROTECTION_COLUMNS: tuple[str, ...] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
)

STREAMING_ENTERTAINMENT_COLUMNS: tuple[str, ...] = (
    "StreamingTV",
    "StreamingMovies",
)

AUTOMATIC_PAYMENT_METHODS: tuple[str, ...] = (
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)

# Allowed category value sets for deterministic domain validation
ALLOWED_CATEGORICAL_VALUES: dict[str, set[str | int]] = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1, "0", "1"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    TARGET_COLUMN: {POSITIVE_CLASS, NEGATIVE_CLASS},
}


class InvalidCategoryValueError(DataValidationError):
    """Raised when a feature contains values outside its defined categorical domain."""


class TargetContractViolationError(DataValidationError):
    """Raised when target values violate binary contract requirements."""


def validate_dataset_contract(
    df: pd.DataFrame,
    check_target: bool = True,
    expected_features: Sequence[str] | None = None,
) -> None:
    """Validate that an input dataset strictly respects the expected preprocessing contract.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    check_target : bool
        Whether to require and validate the ground-truth target column.
    expected_features : Sequence[str], optional
        Expected feature columns. If None, defaults to RAW_FEATURE_COLUMNS.

    Raises
    ------
    EmptyDatasetError
        If df is empty (0 rows or 0 columns).
    DuplicateColumnsError
        If df has duplicated column headers.
    TargetColumnMissingError
        If check_target is True and TARGET_COLUMN is missing.
    TargetContractViolationError
        If TARGET_COLUMN contains unexpected values.
    SchemaValidationError
        If required feature columns are missing.
    InvalidCategoryValueError
        If categorical columns contain unexpected values.
    """
    if df.empty:
        raise EmptyDatasetError("Cannot preprocess an empty DataFrame.")

    if len(df.columns) != len(set(df.columns)):
        dups = df.columns[df.columns.duplicated()].tolist()
        raise DuplicateColumnsError(f"Dataset contains duplicate column names: {dups}")

    # Check target if requested
    if check_target:
        if TARGET_COLUMN not in df.columns:
            raise TargetColumnMissingError(
                f"Target column '{TARGET_COLUMN}' is required but not present in dataset."
            )
        observed_target_values = set(df[TARGET_COLUMN].dropna().unique())
        allowed_target_values = ALLOWED_CATEGORICAL_VALUES[TARGET_COLUMN]
        if not observed_target_values.issubset(allowed_target_values):
            unexpected = observed_target_values - allowed_target_values
            raise TargetContractViolationError(
                f"Target column contains unexpected values: {unexpected}. Allowed: {allowed_target_values}"
            )

    # Check required feature columns
    req_cols = expected_features if expected_features is not None else RAW_FEATURE_COLUMNS
    missing_cols = [col for col in req_cols if col not in df.columns]
    if missing_cols:
        raise SchemaValidationError(
            f"Missing required schema columns for preprocessing: {missing_cols}"
        )

    # Check categorical domains on training/input data
    for col in RAW_CATEGORICAL_COLUMNS:
        if col in df.columns:
            # Drop NaN if any (imputation handles NaN, domain check handles malformed categories)
            observed = set(df[col].dropna().unique())
            # For SeniorCitizen, normalize string/int representation
            allowed = ALLOWED_CATEGORICAL_VALUES[col]
            if not observed.issubset(allowed):
                unexpected = observed - allowed
                raise InvalidCategoryValueError(
                    f"Column '{col}' contains unexpected category values: {unexpected}. "
                    f"Allowed: {allowed}"
                )
