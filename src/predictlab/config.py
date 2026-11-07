"""Centralized configuration module for PredictLab."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for the repository, resolved dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard canonical columns for the verified IBM Telco Customer Churn dataset
TELCO_COLUMNS: tuple[str, ...] = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)


class Settings(BaseSettings):
    """Application settings and configuration parameters with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="PREDICTLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment & Logging
    env: str = Field(default="development", description="Application runtime environment")
    log_level: str = Field(default="INFO", description="Default logging level")

    # Project Paths (relative to PROJECT_ROOT by default)
    project_root: Path = Field(default=PROJECT_ROOT, description="Absolute project root path")
    data_dir_name: str = Field(default="data", description="Data directory name")
    models_dir_name: str = Field(default="models", description="Models directory name")
    artifacts_dir_name: str = Field(default="artifacts", description="Artifacts directory name")
    reports_dir_name: str = Field(default="reports", description="Reports directory name")

    # Dataset Parameters
    raw_dataset_filename: str = Field(
        default="Telco-Customer-Churn.csv",
        description="Raw dataset file name",
    )
    target_column: str = Field(
        default="Churn",
        description="Name of the ground-truth target variable",
    )
    positive_class: str = Field(
        default="Yes",
        description="Label representing the positive churn class",
    )
    negative_class: str = Field(
        default="No",
        description="Label representing the negative retained class",
    )
    id_column: str = Field(
        default="customerID",
        description="Unique identifier column name",
    )
    expected_columns: tuple[str, ...] = Field(
        default=TELCO_COLUMNS,
        description="Expected canonical feature set in raw dataset",
    )
    test_size: float = Field(
        default=0.20,
        description="Default stratified holdout test split fraction",
    )
    random_state: int = Field(
        default=42,
        description="Global deterministic random seed for data splitting and sampling",
    )
    splits_dir_name: str = Field(
        default="splits",
        description="Directory name for dataset partitioning manifests under data directory",
    )

    @property
    def data_dir(self) -> Path:
        """Directory path for all data assets."""
        return self.project_root / self.data_dir_name

    @property
    def splits_dir(self) -> Path:
        """Directory path for train/test partition manifests."""
        return self.data_dir / self.splits_dir_name

    @property
    def raw_data_dir(self) -> Path:
        """Directory path for raw, immutable data files."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Directory path for processed datasets and features."""
        return self.data_dir / "processed"

    @property
    def raw_data_path(self) -> Path:
        """Full file path to the canonical raw dataset."""
        return self.raw_data_dir / self.raw_dataset_filename

    @property
    def models_dir(self) -> Path:
        """Directory path for serialized model artifacts."""
        return self.project_root / self.models_dir_name

    @property
    def artifacts_dir(self) -> Path:
        """Directory path for pipeline artifacts (e.g., encoders, metrics)."""
        return self.project_root / self.artifacts_dir_name

    @property
    def reports_dir(self) -> Path:
        """Directory path for analysis reports and generated figures."""
        return self.project_root / self.reports_dir_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
