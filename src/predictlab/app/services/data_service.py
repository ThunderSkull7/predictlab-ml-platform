"""Data access and static reporting service for PredictLab Streamlit application."""

import io
import json
from typing import Any

import pandas as pd
import streamlit as st

from predictlab.config import get_settings
from predictlab.logging_config import get_logger

logger = get_logger(__name__)


@st.cache_data
def load_dataset_overview_stats() -> dict[str, Any]:
    """Return verified IBM Telco source dataset statistics."""
    return {
        "dataset_name": "IBM Telco Customer Churn",
        "total_records": 7043,
        "input_features": 19,
        "target_column": "Churn",
        "churned_records": 1869,
        "retained_records": 5174,
        "churn_prevalence": 0.26537,
        "train_samples": 5634,
        "holdout_samples": 1409,
    }


@st.cache_data
def load_cv_model_metrics() -> dict[str, Any]:
    """Load validated 5-fold cross-validation metrics for the tuned HistGradientBoosting model."""
    settings = get_settings()
    path = settings.reports_dir / "experiments" / "tuned_model_cv_results.json"
    if not path.exists():
        logger.warning("CV results file not found at: %s", path)
        return {}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return {
        "model_name": data.get("model_name", "Hist Gradient Boosting (Tuned)"),
        "mean_metrics": data.get("mean_metrics", {}),
        "std_metrics": data.get("std_metrics", {}),
        "n_splits": data.get("n_splits", 5),
    }


@st.cache_data
def load_threshold_benchmark_data() -> pd.DataFrame:
    """Load cost and threshold analysis grid evaluated on out-of-fold calibrated probabilities."""
    settings = get_settings()
    path = settings.reports_dir / "experiments" / "threshold_cost_analysis.csv"
    if not path.exists():
        logger.warning("Threshold analysis CSV not found at: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_risk_tier_summary() -> pd.DataFrame:
    """Load validated risk tier distribution on the training cohort."""
    settings = get_settings()
    path = settings.reports_dir / "experiments" / "risk_tier_summary.csv"
    if not path.exists():
        logger.warning("Risk tier summary CSV not found at: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_permutation_importance_data() -> pd.DataFrame:
    """Load 5-fold cross-validated permutation importance results."""
    settings = get_settings()
    path = settings.reports_dir / "experiments" / "permutation_importance_detailed.csv"
    if not path.exists():
        logger.warning("Permutation importance CSV not found at: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def get_sample_customer_presets() -> dict[str, dict[str, Any]]:
    """Return dictionary of representative customer presets for form auto-fill."""
    return {
        "High Risk (Short Tenure, Fiber, Month-to-Month)": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.85,
            "TotalCharges": 179.70,
        },
        "Low Risk (Long Tenure, Two-Year Contract)": {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 65,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 79.50,
            "TotalCharges": 5167.50,
        },
        "Medium Risk (Mid Tenure, Fiber, Month-to-Month)": {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.00,
            "TotalCharges": 2040.00,
        },
        "New Customer (Tenure = 0, Two-Year Contract)": {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 0,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 52.55,
            "TotalCharges": 0.0,
        },
    }


def generate_sample_batch_csv(include_invalid: bool = True) -> str:
    """Generate sample batch CSV string for demonstration and download."""
    records = [
        {
            "customerID": "DEMO-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.85,
            "TotalCharges": 269.55,
        },
        {
            "customerID": "DEMO-002",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 68,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 84.10,
            "TotalCharges": 5718.80,
        },
        {
            "customerID": "DEMO-003",
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.50,
            "TotalCharges": 1146.00,
        },
        {
            "customerID": "DEMO-004",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 45,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 20.00,
            "TotalCharges": 900.00,
        },
        {
            "customerID": "DEMO-005",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 0,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 49.95,
            "TotalCharges": " ",
        },
    ]

    if include_invalid:
        # Add 1 record with negative tenure and 1 with missing column for validation test
        records.append(
            {
                "customerID": "DEMO-ERR1",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": -5,  # Invalid negative tenure
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 45.00,
                "TotalCharges": 100.00,
            }
        )

    df = pd.DataFrame(records)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()
