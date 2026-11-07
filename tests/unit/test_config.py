"""Unit tests for centralized configuration module."""

from pathlib import Path

from predictlab.config import Settings, get_settings


def test_default_settings_paths():
    """Verify that default settings resolve relative paths accurately without hardcoded machine paths."""
    settings = get_settings()

    assert settings.project_root.exists()
    assert settings.project_root.is_dir()
    assert (settings.project_root / "pyproject.toml").is_file()

    assert settings.data_dir == settings.project_root / "data"
    assert settings.raw_data_dir == settings.project_root / "data" / "raw"
    assert settings.processed_data_dir == settings.project_root / "data" / "processed"
    assert settings.models_dir == settings.project_root / "models"
    assert settings.artifacts_dir == settings.project_root / "artifacts"
    assert settings.reports_dir == settings.project_root / "reports"


def test_default_constants():
    """Verify default problem and schema constants."""
    settings = get_settings()

    assert settings.raw_dataset_filename == "Telco-Customer-Churn.csv"
    assert settings.target_column == "Churn"
    assert settings.positive_class == "Yes"
    assert settings.negative_class == "No"
    assert settings.id_column == "customerID"
    assert len(settings.expected_columns) == 21
    assert "Churn" in settings.expected_columns
    assert "customerID" in settings.expected_columns


def test_settings_env_override(monkeypatch):
    """Verify environment variables override default settings."""
    monkeypatch.setenv("PREDICTLAB_ENV", "testing")
    monkeypatch.setenv("PREDICTLAB_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("PREDICTLAB_TARGET_COLUMN", "Exited")

    custom_settings = Settings()
    assert custom_settings.env == "testing"
    assert custom_settings.log_level == "DEBUG"
    assert custom_settings.target_column == "Exited"


def test_settings_custom_root(tmp_path: Path):
    """Verify custom project root resolves child directories accordingly."""
    custom_settings = Settings(project_root=tmp_path)
    assert custom_settings.project_root == tmp_path
    assert custom_settings.data_dir == tmp_path / "data"
    assert custom_settings.raw_data_dir == tmp_path / "data" / "raw"
    assert custom_settings.raw_data_path == tmp_path / "data" / "raw" / "Telco-Customer-Churn.csv"
