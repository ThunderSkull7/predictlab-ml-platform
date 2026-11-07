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


def compute_charges_ratio_defined(
    df: pd.DataFrame,
    tenure_col: str = "tenure",
    monthly_charges_col: str = "MonthlyCharges",
    output_col: str = "ChargesRatioDefined",
) -> pd.Series:
    """Compute deterministic binary indicator for whether ChargesRatio is mathematically defined.

    ChargesRatio = TotalCharges / (tenure * MonthlyCharges) is mathematically undefined when
    tenure == 0 because both numerator and denominator evaluate to 0 (0 / 0).
    This function generates:
    - 1: ChargesRatio is mathematically defined (tenure > 0 and MonthlyCharges > 0).
    - 0: ChargesRatio is undefined because tenure is 0 (or non-positive denominator).

    This indicator is derived strictly from predictor columns without target access.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    tenure_col : str
        Tenure column name.
    monthly_charges_col : str
        Monthly charges column name.
    output_col : str
        Name for generated indicator feature.

    Returns
    -------
    pd.Series
        Binary integer Series (1 if defined, 0 if undefined).
    """
    for col in (tenure_col, monthly_charges_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' for ChargesRatioDefined not found in DataFrame.")

    tenure = pd.to_numeric(df[tenure_col], errors="coerce")
    mc = pd.to_numeric(df[monthly_charges_col], errors="coerce")

    # Defined strictly when tenure > 0 and monthly charges > 0
    is_defined = (tenure > 0) & (mc > 0) & tenure.notna() & mc.notna()
    indicator = is_defined.astype(int)
    indicator.name = output_col
    return indicator


def compute_experimental_charges_ratio(
    df: pd.DataFrame,
    total_charges_col: str = "TotalCharges",
    tenure_col: str = "tenure",
    monthly_charges_col: str = "MonthlyCharges",
    output_col: str = "ChargesRatio",
    validate_finite: bool = True,
) -> pd.Series:
    """Compute the experimental ChargesRatio feature with mathematically valid zero-tenure NaN handling.

    NOTE: This feature is strictly experimental and excluded from the default baseline pipeline.
    It evaluates TotalCharges / (tenure * MonthlyCharges).
    When tenure == 0, the ratio is mathematically undefined (0 / 0) and is explicitly represented as NaN.
    Never substitutes 1.0, an arbitrary epsilon, or another fabricated numeric ratio for zero-tenure records.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    total_charges_col : str
        Total charges column name.
    tenure_col : str
        Tenure column name.
    monthly_charges_col : str
        Monthly charges column name.
    output_col : str
        Name for generated ratio feature.
    validate_finite : bool
        Whether to enforce that inputs and non-zero tenure denominators are finite and valid.

    Returns
    -------
    pd.Series
        Float64 Series of computed charge ratios, with NaN at tenure == 0.

    Raises
    ------
    KeyError
        If required columns are absent from DataFrame.
    ValueError
        If inputs are non-finite, denominators for positive tenure are non-positive,
        or computed ratios for positive tenure are non-finite.
    """
    for col in (total_charges_col, tenure_col, monthly_charges_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' for ChargesRatio not found in DataFrame.")

    tc = pd.to_numeric(df[total_charges_col], errors="coerce")
    tenure = pd.to_numeric(df[tenure_col], errors="coerce")
    mc = pd.to_numeric(df[monthly_charges_col], errors="coerce")

    if validate_finite:
        # Check for unparseable or inf values in input series for positive tenure
        pos_mask = tenure > 0
        if pos_mask.any():
            for s, name in ((tc, total_charges_col), (tenure, tenure_col), (mc, monthly_charges_col)):
                if s.loc[pos_mask].isna().any():
                    raise ValueError(f"Column '{name}' contains non-numeric or NaN values for positive tenure.")
                if np.isinf(s).any():
                    raise ValueError(f"Column '{name}' contains non-finite infinite values.")

    ratio = pd.Series(np.nan, index=df.index, dtype=np.float64, name=output_col)

    pos_mask = tenure > 0
    if pos_mask.any():
        denom = tenure.loc[pos_mask] * mc.loc[pos_mask]
        if (denom <= 0).any():
            raise ValueError("Non-positive denominator encountered for positive tenure.")

        computed = tc.loc[pos_mask] / denom
        if not np.isfinite(computed).all():
            raise ValueError("Calculated ratio contains non-finite values.")

        ratio.loc[pos_mask] = computed

    # Zero tenure is explicitly represented as NaN
    zero_mask = tenure == 0
    ratio.loc[zero_mask] = np.nan

    return ratio
