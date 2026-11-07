"""Scikit-learn compatible custom transformers for PredictLab preprocessing."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from predictlab.preprocessing.cleaning import clean_total_charges
from predictlab.preprocessing.feature_engineering import (
    compute_automatic_payment_flag,
    compute_charges_ratio_defined,
    compute_experimental_charges_ratio,
    compute_optional_service_count,
    compute_streaming_entertainment_count,
    compute_technical_protection_count,
)
from predictlab.preprocessing.schema import OPTIONAL_SERVICE_COLUMNS


class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """Transformer that validates and cleans TotalCharges with zero-tenure domain imputation."""

    def __init__(
        self,
        tenure_col: str = "tenure",
        total_charges_col: str = "TotalCharges",
        validate_zero_tenure: bool = True,
    ) -> None:
        self.tenure_col = tenure_col
        self.total_charges_col = total_charges_col
        self.validate_zero_tenure = validate_zero_tenure
        self.feature_names_in_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TotalChargesCleaner":
        """Fit transformer (records input column names; learns no test or target statistics)."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=str)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean TotalCharges, validate zero-tenure missingness, and impute 0.0."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return clean_total_charges(
            X,
            tenure_col=self.tenure_col,
            total_charges_col=self.total_charges_col,
            validate_zero_tenure=self.validate_zero_tenure,
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names."""
        if input_features is not None:
            return np.array(input_features, dtype=str)
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        raise ValueError("Transformer has not been fitted and input_features not provided.")


class OptionalServiceCounter(BaseEstimator, TransformerMixin):
    """Transformer that engineers OptionalServiceCount and optional service sub-aggregates."""

    def __init__(
        self,
        service_columns: tuple[str, ...] = OPTIONAL_SERVICE_COLUMNS,
        output_col: str = "OptionalServiceCount",
        include_subtotals: bool = False,
        drop_source_columns: bool = False,
        validate_categories: bool = True,
    ) -> None:
        self.service_columns = service_columns
        self.output_col = output_col
        self.include_subtotals = include_subtotals
        self.drop_source_columns = drop_source_columns
        self.validate_categories = validate_categories
        self.feature_names_in_: np.ndarray | None = None
        self.feature_names_out_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OptionalServiceCounter":
        """Fit transformer (records schema; learns no target or test statistics)."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=str)
            cols = list(X.columns)
            if self.drop_source_columns:
                for col in self.service_columns:
                    if col in cols:
                        cols.remove(col)
            cols.append(self.output_col)
            if self.include_subtotals:
                cols.extend(["TechnicalProtectionCount", "StreamingEntertainmentCount"])
            self.feature_names_out_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute OptionalServiceCount and append to output DataFrame."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        df_out = X.copy()
        df_out[self.output_col] = compute_optional_service_count(
            df_out,
            columns=self.service_columns,
            output_col=self.output_col,
            validate_categories=self.validate_categories,
        )

        if self.include_subtotals:
            df_out["TechnicalProtectionCount"] = compute_technical_protection_count(df_out)
            df_out["StreamingEntertainmentCount"] = compute_streaming_entertainment_count(df_out)

        if self.drop_source_columns:
            df_out = df_out.drop(columns=list(self.service_columns))

        return df_out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names."""
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_, dtype=str)
        if input_features is not None:
            cols = list(input_features)
            if self.drop_source_columns:
                for col in self.service_columns:
                    if col in cols:
                        cols.remove(col)
            cols.append(self.output_col)
            if self.include_subtotals:
                cols.extend(["TechnicalProtectionCount", "StreamingEntertainmentCount"])
            return np.array(cols, dtype=str)
        raise ValueError("Transformer has not been fitted and input_features not provided.")


class AutomaticPaymentTransformer(BaseEstimator, TransformerMixin):
    """Transformer that creates binary AutomaticPayment indicator."""

    def __init__(
        self,
        payment_col: str = "PaymentMethod",
        output_col: str = "AutomaticPayment",
        drop_payment_col: bool = False,
        validate_categories: bool = True,
    ) -> None:
        self.payment_col = payment_col
        self.output_col = output_col
        self.drop_payment_col = drop_payment_col
        self.validate_categories = validate_categories
        self.feature_names_in_: np.ndarray | None = None
        self.feature_names_out_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AutomaticPaymentTransformer":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=str)
            cols = list(X.columns)
            if self.drop_payment_col and self.payment_col in cols:
                cols.remove(self.payment_col)
            cols.append(self.output_col)
            self.feature_names_out_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        df_out = X.copy()
        df_out[self.output_col] = compute_automatic_payment_flag(
            df_out,
            payment_col=self.payment_col,
            output_col=self.output_col,
            validate_categories=self.validate_categories,
        )
        if self.drop_payment_col and self.payment_col in df_out.columns:
            df_out = df_out.drop(columns=[self.payment_col])
        return df_out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_, dtype=str)
        if input_features is not None:
            cols = list(input_features)
            if self.drop_payment_col and self.payment_col in cols:
                cols.remove(self.payment_col)
            cols.append(self.output_col)
            return np.array(cols, dtype=str)
        raise ValueError("Transformer has not been fitted.")


class ExperimentalChargesRatioTransformer(BaseEstimator, TransformerMixin):
    """Experimental transformer generating mathematically valid ChargesRatio (NaN at tenure 0)
    and deterministic ChargesRatioDefined indicator."""

    def __init__(
        self,
        total_charges_col: str = "TotalCharges",
        tenure_col: str = "tenure",
        monthly_charges_col: str = "MonthlyCharges",
        output_col: str = "ChargesRatio",
        defined_indicator_col: str = "ChargesRatioDefined",
        include_defined_indicator: bool = True,
    ) -> None:
        self.total_charges_col = total_charges_col
        self.tenure_col = tenure_col
        self.monthly_charges_col = monthly_charges_col
        self.output_col = output_col
        self.defined_indicator_col = defined_indicator_col
        self.include_defined_indicator = include_defined_indicator
        self.feature_names_in_: np.ndarray | None = None
        self.feature_names_out_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ExperimentalChargesRatioTransformer":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=str)
            cols = list(X.columns)
            cols.append(self.output_col)
            if self.include_defined_indicator:
                cols.append(self.defined_indicator_col)
            self.feature_names_out_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        df_out = X.copy()
        df_out[self.output_col] = compute_experimental_charges_ratio(
            df_out,
            total_charges_col=self.total_charges_col,
            tenure_col=self.tenure_col,
            monthly_charges_col=self.monthly_charges_col,
            output_col=self.output_col,
        )
        if self.include_defined_indicator:
            df_out[self.defined_indicator_col] = compute_charges_ratio_defined(
                df_out,
                tenure_col=self.tenure_col,
                monthly_charges_col=self.monthly_charges_col,
                output_col=self.defined_indicator_col,
            )
        return df_out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_, dtype=str)
        if input_features is not None:
            cols = list(input_features)
            cols.append(self.output_col)
            if self.include_defined_indicator:
                cols.append(self.defined_indicator_col)
            return np.array(cols, dtype=str)
        raise ValueError("Transformer has not been fitted.")


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops specified columns if present in DataFrame."""

    def __init__(self, columns_to_drop: Sequence[str] | None = None) -> None:
        self.columns_to_drop = columns_to_drop
        self.feature_names_in_: np.ndarray | None = None
        self.feature_names_out_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnDropper":
        if isinstance(X, pd.DataFrame):
            drop_cols = list(self.columns_to_drop) if self.columns_to_drop is not None else []
            self.feature_names_in_ = np.array(X.columns, dtype=str)
            self.feature_names_out_ = [c for c in X.columns if c not in drop_cols]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        drop_cols = list(self.columns_to_drop) if self.columns_to_drop is not None else []
        existing_drops = [c for c in drop_cols if c in X.columns]
        if existing_drops:
            return X.drop(columns=existing_drops)
        return X.copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_, dtype=str)
        if input_features is not None:
            drop_cols = list(self.columns_to_drop) if self.columns_to_drop is not None else []
            return np.array([c for c in input_features if c not in drop_cols], dtype=str)
        raise ValueError("Transformer has not been fitted.")
