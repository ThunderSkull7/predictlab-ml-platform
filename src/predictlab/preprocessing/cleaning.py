"""Data cleaning and structural validation module for PredictLab preprocessing."""

import numpy as np
import pandas as pd

from predictlab.data.validate import DataValidationError
from predictlab.logging_config import get_logger

logger = get_logger(__name__)


class TotalChargesValidationError(DataValidationError):
    """Raised when TotalCharges contains missingness or anomalies that violate domain rules."""


def clean_total_charges(
    df: pd.DataFrame,
    tenure_col: str = "tenure",
    total_charges_col: str = "TotalCharges",
    validate_zero_tenure: bool = True,
) -> pd.DataFrame:
    """Clean TotalCharges by converting whitespace to NaN, validating zero tenure, and imputing 0.0.

    This function implements a mandatory data-quality contract:
    1. Detects whitespace-only strings in TotalCharges and converts them to NaN.
    2. Converts valid numeric strings to float64, raising on malformed non-numeric strings.
    3. Validates that any missing TotalCharges values occur strictly and exclusively where tenure == 0.
    4. If missing TotalCharges occurs where tenure > 0, raises an informative TotalChargesValidationError.
    5. Imputes validated zero-tenure missing values with 0.0.
    6. Ensures the resulting TotalCharges feature is strictly numeric and finite.
    7. Guarantees input DataFrame immutability by operating on an explicit copy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing tenure and TotalCharges columns.
    tenure_col : str
        Column name for customer tenure (defaults to 'tenure').
    total_charges_col : str
        Column name for TotalCharges (defaults to 'TotalCharges').
    validate_zero_tenure : bool
        Whether to enforce the strict zero-tenure missingness check (defaults to True).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with cleaned and imputed TotalCharges as float64.

    Raises
    ------
    DataValidationError
        If required columns are absent or df is empty.
    TotalChargesValidationError
        If missing TotalCharges occurs where tenure > 0, or if unparseable non-numeric text exists.
    """
    if df.empty:
        raise DataValidationError("Cannot clean TotalCharges on an empty DataFrame.")

    if total_charges_col not in df.columns:
        raise DataValidationError(f"Required column '{total_charges_col}' not found in DataFrame.")

    if validate_zero_tenure and tenure_col not in df.columns:
        raise DataValidationError(
            f"Required column '{tenure_col}' not found in DataFrame for zero-tenure validation."
        )

    # Guarantee input immutability
    df_out = df.copy()

    # Step 1: Detect whitespace-only strings and strip strings
    raw_col = df_out[total_charges_col]
    if pd.api.types.is_object_dtype(raw_col) or pd.api.types.is_string_dtype(raw_col):
        # Identify whitespace strings
        is_whitespace = raw_col.astype(str).str.strip() == ""
        # Convert whitespace to NaN
        series_clean = raw_col.mask(is_whitespace, np.nan)
    else:
        series_clean = raw_col

    # Step 2: Convert to numeric float64, detecting malformed non-numeric strings
    # Check if there are any non-numeric strings other than NaN/whitespace
    numeric_series = pd.to_numeric(series_clean, errors="coerce")

    # If errors='coerce' produced new NaNs that were NOT originally NaN or whitespace, raise error
    original_nan_mask = series_clean.isna()
    coerced_nan_mask = numeric_series.isna() & ~original_nan_mask
    if coerced_nan_mask.any():
        bad_indices = df_out.index[coerced_nan_mask].tolist()
        bad_values = df_out.loc[coerced_nan_mask, total_charges_col].tolist()
        raise TotalChargesValidationError(
            f"TotalCharges contains unparseable non-numeric values at indices {bad_indices[:5]}: {bad_values[:5]}"
        )

    # Step 3 & 4: Validate zero-tenure missingness
    if validate_zero_tenure:
        tenure_series = pd.to_numeric(df_out[tenure_col], errors="coerce")
        missing_mask = numeric_series.isna()
        if missing_mask.any():
            invalid_missing = missing_mask & (tenure_series > 0)
            if invalid_missing.any():
                violation_indices = df_out.index[invalid_missing].tolist()
                violation_tenures = df_out.loc[invalid_missing, tenure_col].tolist()
                msg = (
                    f"TotalCharges validation failed: {int(invalid_missing.sum())} records have "
                    f"missing TotalCharges with positive tenure (tenure > 0). "
                    f"Sample indices: {violation_indices[:5]}, sample tenures: {violation_tenures[:5]}. "
                    f"Domain rule strictly mandates that TotalCharges missingness is only permitted for newly enrolled signups at tenure == 0."
                )
                logger.error(msg)
                raise TotalChargesValidationError(msg)

            # Check for negative tenure or NaN tenure on missing records
            invalid_tenure = missing_mask & (tenure_series < 0 | tenure_series.isna())
            if invalid_tenure.any():
                msg = "TotalCharges missing on records with negative or NaN tenure."
                logger.error(msg)
                raise TotalChargesValidationError(msg)

    # Step 5: Impute validated zero-tenure missing values with 0.0
    imputed_series = numeric_series.fillna(0.0).astype(np.float64)

    # Step 6: Verify all values are finite float64
    if not np.isfinite(imputed_series).all():
        raise TotalChargesValidationError("TotalCharges contains non-finite values after imputation.")

    df_out[total_charges_col] = imputed_series
    return df_out
