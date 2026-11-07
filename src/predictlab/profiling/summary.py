"""Univariate statistical profiling module for PredictLab."""

from typing import Any

import numpy as np
import pandas as pd


def compute_dataset_overview(df: pd.DataFrame, id_col: str = "customerID") -> dict[str, Any]:
    """Compute structural metrics and dimensions for the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to profile.
    id_col : str
        Name of identifier column.

    Returns
    -------
    dict[str, Any]
        Dictionary of overview metrics.
    """
    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])
    mem_bytes = int(df.memory_usage(deep=True).sum())

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != id_col]

    return {
        "total_observations": total_rows,
        "total_features": total_cols,
        "numeric_features_count": len(numeric_cols),
        "categorical_features_count": len(categorical_cols),
        "identifier_column": id_col if id_col in df.columns else None,
        "unique_customers": int(df[id_col].nunique()) if id_col in df.columns else total_rows,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_identifiers": int(df[id_col].duplicated().sum()) if id_col in df.columns else 0,
        "memory_usage_mb": round(mem_bytes / (1024 * 1024), 2),
    }


def compute_numeric_summary(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute comprehensive descriptive distribution statistics for numerical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numerical variables.
    columns : Optional[list[str]]
        Specific columns to summarize. If None, automatically detects numeric columns
        and handles TotalCharges if present as an object.

    Returns
    -------
    pd.DataFrame
        Table with column, count, missing_count, whitespace_count, mean, std, min,
        q25, median, q75, max, iqr, skewness.
    """
    if columns is not None:
        target_cols = columns
    else:
        # Detect numeric columns and TotalCharges
        target_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if "TotalCharges" in df.columns and "TotalCharges" not in target_cols:
            target_cols.append("TotalCharges")

    records = []
    for col in target_cols:
        if col not in df.columns:
            continue

        raw_series = df[col]
        # Count explicit whitespace strings
        ws_count = int(raw_series.astype(str).str.strip().eq("").sum()) if raw_series.dtype == "object" else 0
        null_count = int(raw_series.isna().sum())

        # Safely convert to numeric for statistical analysis without mutating raw data
        s = pd.to_numeric(raw_series.replace(r"^\s*$", np.nan, regex=True), errors="coerce").dropna()
        valid_count = int(len(s))

        if valid_count == 0:
            records.append(
                {
                    "feature": col,
                    "valid_count": 0,
                    "missing_count": null_count,
                    "whitespace_count": ws_count,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "max": np.nan,
                    "iqr": np.nan,
                    "skew": np.nan,
                }
            )
            continue

        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        iqr = float(q75 - q25)

        records.append(
            {
                "feature": col,
                "valid_count": valid_count,
                "missing_count": null_count,
                "whitespace_count": ws_count,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "q25": q25,
                "median": float(s.median()),
                "q75": q75,
                "max": float(s.max()),
                "iqr": iqr,
                "skew": float(s.skew()),
            }
        )

    return pd.DataFrame(records)


def compute_categorical_summary(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    target_col: str = "Churn",
) -> dict[str, pd.DataFrame]:
    """Compute category frequencies, percentages, and cardinality for categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    columns : Optional[list[str]]
        Specific columns. If None, selects all object/string/boolean columns except target.
    target_col : str
        Target column name (excluded from standard categorical feature list if default).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from feature name to a summary DataFrame with category, frequency, and percentage.
    """
    if columns is not None:
        target_cols = columns
    else:
        target_cols = [
            c
            for c in df.columns
            if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
            and c != target_col
            and c != "customerID"
            and c != "TotalCharges"
        ]
        # Include SeniorCitizen if present
        if "SeniorCitizen" in df.columns and "SeniorCitizen" not in target_cols:
            target_cols.append("SeniorCitizen")

    summaries = {}
    total_len = len(df)

    for col in target_cols:
        if col not in df.columns:
            continue

        s = df[col]
        # Check for whitespace
        ws_count = int(s.astype(str).str.strip().eq("").sum()) if s.dtype == "object" else 0
        null_count = int(s.isna().sum())

        counts = s.value_counts(dropna=False)
        pcts = s.value_counts(dropna=False, normalize=True) * 100

        col_df = pd.DataFrame(
            {
                "category": [str(c) if pd.notna(c) else "<NULL>" for c in counts.index],
                "frequency": counts.values.astype(int),
                "percentage": pcts.values.round(2),
            }
        )
        col_df.attrs["whitespace_count"] = ws_count
        col_df.attrs["null_count"] = null_count
        col_df.attrs["cardinality"] = int(s.nunique(dropna=False))
        col_df.attrs["total_observations"] = total_len

        summaries[col] = col_df

    return summaries


def compute_target_distribution(
    df: pd.DataFrame,
    target_col: str = "Churn",
    positive_class: str = "Yes",
    negative_class: str = "No",
) -> dict[str, Any]:
    """Calculate target class frequencies, percentages, and class balance ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Target column name.
    positive_class : str
        Label representing the positive churn class.
    negative_class : str
        Label representing the negative retained class.

    Returns
    -------
    dict[str, Any]
        Dictionary detailing target distribution metrics.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    counts = df[target_col].value_counts(dropna=False)
    pcts = df[target_col].value_counts(dropna=False, normalize=True) * 100

    pos_count = int(counts.get(positive_class, 0))
    neg_count = int(counts.get(negative_class, 0))
    total_count = int(len(df[target_col]))

    pos_pct = float(pcts.get(positive_class, 0.0))
    neg_pct = float(pcts.get(negative_class, 0.0))

    neg_to_pos_ratio = round(neg_count / pos_count, 4) if pos_count > 0 else np.nan
    pos_to_neg_ratio = round(pos_count / neg_count, 4) if neg_count > 0 else np.nan

    return {
        "target_column": target_col,
        "total_records": total_count,
        "positive_class": positive_class,
        "negative_class": negative_class,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_percentage": round(pos_pct, 2),
        "negative_percentage": round(neg_pct, 2),
        "negative_to_positive_ratio": neg_to_pos_ratio,
        "positive_to_negative_ratio": pos_to_neg_ratio,
        "class_balance_description": (
            f"Moderate class imbalance (~{neg_to_pos_ratio:.2f}:1 retained vs churned accounts). "
            f"Positive churn rate is {pos_pct:.2f}%."
        ),
    }
