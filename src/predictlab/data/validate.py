"""Data validation foundation module for PredictLab."""

from collections.abc import Sequence

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from predictlab.config import get_settings
from predictlab.logging_config import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Base exception for data validation failures."""


class EmptyDatasetError(DataValidationError):
    """Raised when a dataset contains zero rows or columns."""


class TargetColumnMissingError(DataValidationError):
    """Raised when the specified target column is missing from the dataset."""


class DuplicateColumnsError(DataValidationError):
    """Raised when duplicate column names exist in the dataset."""


class SchemaValidationError(DataValidationError):
    """Raised when required schema columns are missing."""


class ValidationResult(BaseModel):
    """Structured report detailing dataset validation findings with active integrity validation."""

    is_valid: bool = Field(description="Whether the dataset passed all mandatory checks")
    row_count: int = Field(ge=0, description="Number of rows in the dataset")
    column_count: int = Field(ge=0, description="Number of columns in the dataset")
    missing_columns: list[str] = Field(
        default_factory=list, description="Expected columns absent from dataset"
    )
    extra_columns: list[str] = Field(
        default_factory=list, description="Columns present beyond the expected schema"
    )
    duplicate_columns: list[str] = Field(
        default_factory=list, description="Column names occurring more than once"
    )
    target_present: bool = Field(description="Whether the ground truth target column is present")
    target_classes: list[str] = Field(
        default_factory=list, description="Unique values observed in target column"
    )
    duplicate_rows: int = Field(default=0, ge=0, description="Count of duplicate rows across all columns")
    duplicate_ids: int = Field(default=0, ge=0, description="Count of duplicate values in ID column")
    errors: list[str] = Field(default_factory=list, description="Critical validation error messages")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warning notices")

    @model_validator(mode="after")
    def verify_validation_consistency(self) -> "ValidationResult":
        """Validate logical consistency between validity flag, error lists, and row counts."""
        expected_validity = len(self.errors) == 0
        if self.is_valid != expected_validity:
            raise ValueError(
                f"ValidationResult consistency violation: is_valid={self.is_valid} but error count is {len(self.errors)}."
            )
        if self.duplicate_rows > self.row_count:
            raise ValueError(
                f"duplicate_rows ({self.duplicate_rows}) cannot exceed row_count ({self.row_count})."
            )
        if not self.target_present and len(self.target_classes) > 0:
            raise ValueError(
                "target_classes cannot be populated when target_present is False."
            )
        return self


def validate_raw_data(
    df: pd.DataFrame,
    target_column: str | None = None,
    id_column: str | None = None,
    expected_columns: Sequence[str] | None = None,
    raise_on_error: bool = False,
) -> ValidationResult:
    """Validate foundational structural assumptions and schema expectations for a raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    target_column : Optional[str]
        Target column name (defaults to configured target_column).
    id_column : Optional[str]
        Unique identifier column name (defaults to configured id_column).
    expected_columns : Optional[Sequence[str]]
        List of expected column names (defaults to configured expected_columns).
    raise_on_error : bool
        If True, raises the first fatal validation error encountered.
        If False, accumulates errors and returns a ValidationResult.

    Returns
    -------
    ValidationResult
        Structured validation results.

    Raises
    ------
    EmptyDatasetError
        If dataset has zero rows or zero columns and raise_on_error is True.
    DuplicateColumnsError
        If duplicate column headers exist and raise_on_error is True.
    TargetColumnMissingError
        If target column is absent and raise_on_error is True.
    SchemaValidationError
        If expected columns are missing and raise_on_error is True.
    """
    settings = get_settings()
    target_col = target_column if target_column is not None else settings.target_column
    id_col = id_column if id_column is not None else settings.id_column
    expected = list(expected_columns if expected_columns is not None else settings.expected_columns)

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Check for empty dataframe
    if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
        msg = f"Dataset is empty: shape={df.shape}."
        errors.append(msg)
        logger.error(msg)
        if raise_on_error:
            raise EmptyDatasetError(msg)
        return ValidationResult(
            is_valid=False,
            row_count=df.shape[0],
            column_count=df.shape[1],
            target_present=False,
            errors=errors,
            warnings=warnings,
        )

    # 2. Check for duplicate column names
    col_names = list(df.columns)
    duplicate_cols = [col for col in set(col_names) if col_names.count(col) > 1]
    if duplicate_cols:
        msg = f"Duplicate column headers detected: {sorted(duplicate_cols)}"
        errors.append(msg)
        logger.error(msg)
        if raise_on_error:
            raise DuplicateColumnsError(msg)

    # 3. Check for target column existence
    target_present = target_col in df.columns
    target_classes: list[str] = []
    if not target_present:
        msg = f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}"
        errors.append(msg)
        logger.error(msg)
        if raise_on_error:
            raise TargetColumnMissingError(msg)
    else:
        target_series = df[target_col].dropna()
        target_classes = sorted([str(val) for val in target_series.unique()])
        if len(target_classes) == 0:
            msg = f"Target column '{target_col}' contains only missing values."
            errors.append(msg)
            logger.error(msg)
            if raise_on_error:
                raise TargetColumnMissingError(msg)
        elif len(target_classes) == 1:
            warnings.append(
                f"Target column '{target_col}' is degenerate (contains only one class: {target_classes[0]})."
            )

    # 4. Check expected schema / missing columns
    present_cols = set(df.columns)
    missing_cols = [c for c in expected if c not in present_cols]
    extra_cols = [c for c in present_cols if c not in expected]

    if missing_cols:
        msg = f"Dataset is missing {len(missing_cols)} expected columns: {missing_cols}"
        errors.append(msg)
        logger.error(msg)
        if raise_on_error:
            raise SchemaValidationError(msg)

    if extra_cols:
        warnings.append(f"Dataset contains {len(extra_cols)} unexpected extra columns: {sorted(extra_cols)}")

    # 5. Identifier and duplicate checks
    duplicate_rows_count = int(df.duplicated().sum())
    if duplicate_rows_count > 0:
        warnings.append(f"Dataset contains {duplicate_rows_count} completely duplicated rows.")

    duplicate_ids_count = 0
    if id_col in df.columns:
        duplicate_ids_count = int(df[id_col].duplicated().sum())
        if duplicate_ids_count > 0:
            warnings.append(f"ID column '{id_col}' contains {duplicate_ids_count} duplicate keys.")
    else:
        warnings.append(f"Configured identifier column '{id_col}' is not present in dataset.")

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        row_count=df.shape[0],
        column_count=df.shape[1],
        missing_columns=missing_cols,
        extra_columns=sorted(extra_cols),
        duplicate_columns=sorted(duplicate_cols),
        target_present=target_present,
        target_classes=target_classes,
        duplicate_rows=duplicate_rows_count,
        duplicate_ids=duplicate_ids_count,
        errors=errors,
        warnings=warnings,
    )
