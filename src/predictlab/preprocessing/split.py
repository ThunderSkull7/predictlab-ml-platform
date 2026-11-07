"""Dataset partitioning module with reproducible stratification and lightweight manifest persistence."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from predictlab.config import get_settings
from predictlab.data.validate import (
    DataValidationError,
    EmptyDatasetError,
    TargetColumnMissingError,
    validate_raw_data,
)
from predictlab.logging_config import get_logger
from predictlab.preprocessing.schema import (
    ID_COLUMN,
    POSITIVE_CLASS,
    TARGET_COLUMN,
    TARGET_MAPPING,
)

logger = get_logger(__name__)


class SplitManifest(BaseModel):
    """Pydantic model recording reproducible dataset partitioning metadata and customer identifiers."""

    created_at_utc: str = Field(description="ISO 8601 timestamp when partition was generated")
    raw_data_sha256: str = Field(description="SHA-256 hash of raw source dataset")
    total_samples: int = Field(ge=0, description="Total sample count in raw dataset")
    train_samples: int = Field(ge=0, description="Count of training samples")
    test_samples: int = Field(ge=0, description="Count of test holdout samples")
    test_size_fraction: float = Field(gt=0.0, lt=1.0, description="Test holdout fraction")
    random_state: int = Field(description="Deterministic random seed used for splitting")
    stratify_column: str = Field(description="Column used for stratification")
    overall_churn_rate_pct: float = Field(ge=0.0, le=100.0)
    train_churn_rate_pct: float = Field(ge=0.0, le=100.0)
    test_churn_rate_pct: float = Field(ge=0.0, le=100.0)
    train_customer_ids: list[str] = Field(description="List of customerIDs allocated to training")
    test_customer_ids: list[str] = Field(description="List of customerIDs allocated to holdout test")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_dataset(
    df: pd.DataFrame,
    test_size: float | None = None,
    random_state: int | None = None,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
    exclude_id_from_features: bool = True,
    map_target_to_binary: bool = True,
    save_manifest: bool = True,
    manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, SplitManifest]:
    """Partition dataset into stratified training and testing sets with manifest persistence.

    Enforces strict leakage-prevention contracts:
    - Verifies raw dataset integrity.
    - Excludes customerID from the feature matrices (X_train, X_test).
    - Extracts and stratifies on Churn.
    - Encodes Churn as binary integer (Yes=1, No=0).
    - Guarantees zero sample overlap between train and test sets.
    - Persists a lightweight reproducible split manifest containing IDs and distribution audits.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical input DataFrame.
    test_size : float, optional
        Holdout fraction (defaults to settings.test_size = 0.20).
    random_state : int, optional
        Random seed (defaults to settings.random_state = 42).
    target_col : str
        Target column name (defaults to 'Churn').
    id_col : str
        Identifier column name (defaults to 'customerID').
    exclude_id_from_features : bool
        Whether to drop customerID from feature matrices (defaults to True).
    map_target_to_binary : bool
        Whether to map target strings to {0, 1} (defaults to True).
    save_manifest : bool
        Whether to persist the split manifest to disk (defaults to True).
    manifest_path : Path, optional
        Custom path for saving the manifest.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, SplitManifest]
        (X_train, X_test, y_train, y_test, manifest)

    Raises
    ------
    EmptyDatasetError
        If df is empty.
    TargetColumnMissingError
        If target_col is missing.
    DataValidationError
        If id_col is missing or contains duplicate values.
    """
    settings = get_settings()
    test_size = test_size if test_size is not None else settings.test_size
    random_state = random_state if random_state is not None else settings.random_state

    if df.empty:
        raise EmptyDatasetError("Cannot split an empty DataFrame.")

    if target_col not in df.columns:
        raise TargetColumnMissingError(f"Target column '{target_col}' not found in DataFrame.")

    if id_col not in df.columns:
        raise DataValidationError(f"Identifier column '{id_col}' not found in DataFrame.")

    if df[id_col].duplicated().any():
        dup_count = int(df[id_col].duplicated().sum())
        raise DataValidationError(
            f"Dataset contains {dup_count} duplicate customer identifiers. Cannot split reliably."
        )

    # Validate overall dataset schema integrity
    val_report = validate_raw_data(df)
    if not val_report.is_valid:
        raise DataValidationError(f"Dataset validation failed before splitting: {val_report.errors}")

    # Compute raw data checksum if available
    raw_path = settings.raw_data_path
    raw_sha256 = compute_sha256(raw_path) if raw_path.exists() else "unknown"

    # Separate target
    y_raw = df[target_col].copy()
    if map_target_to_binary:
        y = y_raw.map(TARGET_MAPPING).astype(int)
    else:
        y = y_raw

    # Prepare feature matrix X
    drop_cols = [target_col]
    if exclude_id_from_features:
        drop_cols.append(id_col)
    X = df.drop(columns=drop_cols).copy()

    # Preserve customer IDs for the manifest
    customer_ids = df[id_col].values

    # Perform stratified split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        customer_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw,
    )

    # Audit churn proportions
    overall_churn_pct = float((y_raw == POSITIVE_CLASS).mean() * 100.0)
    train_churn_pct = float(
        ((y_train == 1) if map_target_to_binary else (y_train == POSITIVE_CLASS)).mean() * 100.0
    )
    test_churn_pct = float(
        ((y_test == 1) if map_target_to_binary else (y_test == POSITIVE_CLASS)).mean() * 100.0
    )

    manifest = SplitManifest(
        created_at_utc=datetime.now(UTC).isoformat(),
        raw_data_sha256=raw_sha256,
        total_samples=len(df),
        train_samples=len(X_train),
        test_samples=len(X_test),
        test_size_fraction=test_size,
        random_state=random_state,
        stratify_column=target_col,
        overall_churn_rate_pct=round(overall_churn_pct, 4),
        train_churn_rate_pct=round(train_churn_pct, 4),
        test_churn_rate_pct=round(test_churn_pct, 4),
        train_customer_ids=list(ids_train),
        test_customer_ids=list(ids_test),
    )

    if save_manifest:
        target_manifest_path = (
            manifest_path
            if manifest_path is not None
            else settings.splits_dir / "split_manifest.json"
        )
        target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        logger.info("Persisted split manifest to: %s", target_manifest_path.resolve())

    logger.info(
        "Dataset split complete: Train=%d (%.2f%% churn), Test=%d (%.2f%% churn)",
        len(X_train),
        train_churn_pct,
        len(X_test),
        test_churn_pct,
    )

    return X_train, X_test, y_train, y_test, manifest


def load_split_manifest(manifest_path: Path | None = None) -> SplitManifest:
    """Load and validate a persisted split manifest from disk."""
    settings = get_settings()
    path = manifest_path if manifest_path is not None else settings.splits_dir / "split_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found at: {path.resolve()}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return SplitManifest(**data)
