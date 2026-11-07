"""Feature engineering functions and domain transformations for PredictLab."""

import numpy as np
import pandas as pd

from predictlab.preprocessing.schema import (
    ALLOWED_CATEGORICAL_VALUES,
    AUTOMATIC_PAYMENT_METHODS,
    OPTIONAL_SERVICE_COLUMNS,
    STREAMING_ENTERTAINMENT_COLUMNS,
    TECHNICAL_PROTECTION_COLUMNS,
    InvalidCategoryValueError,
)


def compute_optional_service_count(
    df: pd.DataFrame,
    columns: tuple[str, ...] = OPTIONAL_SERVICE_COLUMNS,
    output_col: str = "OptionalServiceCount",
    validate_categories: bool = True,
) -> pd.Series:
    """Compute the integer count of active optional internet services (0 to 6).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the optional service columns.
    columns : tuple[str, ...]
        Columns to evaluate (defaults to all 6 optional services).
    output_col : str
        Name for the generated Series.
    validate_categories : bool
        Whether to enforce that values belong strictly to {'Yes', 'No', 'No internet service'}.

    Returns
    -------
    pd.Series
        Integer Series with values in range [0, len(columns)].

    Raises
    ------
    KeyError
        If any required service column is absent from df.
    InvalidCategoryValueError
        If any service column contains an unrecognized category value.
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Required service column '{col}' not found in DataFrame.")

        if validate_categories:
            allowed = ALLOWED_CATEGORICAL_VALUES[col]
            observed = set(df[col].dropna().unique())
            if not observed.issubset(allowed):
                unexpected = observed - allowed
                raise InvalidCategoryValueError(
                    f"Column '{col}' contains unrecognized values {unexpected}. "
                    f"Expected subset of {allowed}."
                )

    # Clean boolean indicator for each column ("Yes" == 1, all others == 0)
    service_matrix = pd.DataFrame(index=df.index)
    for col in columns:
        service_matrix[col] = (df[col] == "Yes").astype(int)

    count_series = service_matrix.sum(axis=1).astype(int)
    count_series.name = output_col
    return count_series


def compute_technical_protection_count(
    df: pd.DataFrame,
    output_col: str = "TechnicalProtectionCount",
) -> pd.Series:
    """Compute count of active security/protection services (0 to 4)."""
    return compute_optional_service_count(
        df,
        columns=TECHNICAL_PROTECTION_COLUMNS,
        output_col=output_col,
    )


def compute_streaming_entertainment_count(
    df: pd.DataFrame,
    output_col: str = "StreamingEntertainmentCount",
) -> pd.Series:
    """Compute count of active streaming services (0 to 2)."""
    return compute_optional_service_count(
        df,
        columns=STREAMING_ENTERTAINMENT_COLUMNS,
        output_col=output_col,
    )


def compute_automatic_payment_flag(
    df: pd.DataFrame,
    payment_col: str = "PaymentMethod",
    output_col: str = "AutomaticPayment",
    validate_categories: bool = True,
) -> pd.Series:
    """Compute binary indicator for automated payment methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing payment_col.
    payment_col : str
        Column name for payment method.
    output_col : str
        Name for generated binary feature.
    validate_categories : bool
        Whether to enforce that payment methods match the canonical schema.

    Returns
    -------
    pd.Series
        Binary integer Series (1 if automated payment, 0 otherwise).

    Raises
    ------
    KeyError
        If payment_col is missing.
    InvalidCategoryValueError
        If unrecognized payment methods are detected.
    """
    if payment_col not in df.columns:
        raise KeyError(f"Payment column '{payment_col}' not found in DataFrame.")

    if validate_categories:
        allowed = ALLOWED_CATEGORICAL_VALUES[payment_col]
        observed = set(df[payment_col].dropna().unique())
        if not observed.issubset(allowed):
            unexpected = observed - allowed
            raise InvalidCategoryValueError(
                f"Column '{payment_col}' contains unrecognized payment methods {unexpected}. "
                f"Expected subset of {allowed}."
            )

    flag_series = df[payment_col].isin(AUTOMATIC_PAYMENT_METHODS).astype(int)
    flag_series.name = output_col
    return flag_series


def compute_experimental_charges_ratio(
    df: pd.DataFrame,
    total_charges_col: str = "TotalCharges",
    tenure_col: str = "tenure",
    monthly_charges_col: str = "MonthlyCharges",
    output_col: str = "ChargesRatio",
    zero_tenure_fallback: float = 1.0,
) -> pd.Series:
    """Compute the experimental ChargesRatio feature with explicit zero-tenure branching.

    NOTE: This feature is strictly experimental and excluded from the default baseline pipeline.
    It evaluates TotalCharges / (tenure * MonthlyCharges). When tenure == 0, it avoids arbitrary
    epsilon denominators by explicitly branching to zero_tenure_fallback (default: 1.0).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    total_charges_col : str
        Total charges column.
    tenure_col : str
        Tenure column.
    monthly_charges_col : str
        Monthly charges column.
    output_col : str
        Name for generated ratio feature.
    zero_tenure_fallback : float
        Value assigned when tenure == 0 (defaults to 1.0).

    Returns
    -------
    pd.Series
        Numeric Series of computed charge ratios.
    """
    for col in (total_charges_col, tenure_col, monthly_charges_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' for ChargesRatio not found in DataFrame.")

    tc = pd.to_numeric(df[total_charges_col], errors="coerce")
    tenure = pd.to_numeric(df[tenure_col], errors="coerce")
    mc = pd.to_numeric(df[monthly_charges_col], errors="coerce")

    # Explicit condition without arbitrary epsilon
    ratio = pd.Series(index=df.index, dtype=np.float64, name=output_col)

    # For positive tenure, compute standard ratio
    positive_mask = tenure > 0
    expected_charges = tenure * mc

    # Avoid zero division if monthly charges is 0
    valid_denom_mask = positive_mask & (expected_charges > 0)
    ratio.loc[valid_denom_mask] = tc.loc[valid_denom_mask] / expected_charges.loc[valid_denom_mask]

    # Explicit handling for tenure == 0
    zero_tenure_mask = tenure == 0
    ratio.loc[zero_tenure_mask] = zero_tenure_fallback

    # Handle remaining non-positive tenure or zero denominator gracefully
    other_mask = ratio.isna()
    ratio.loc[other_mask] = zero_tenure_fallback

    return ratio
