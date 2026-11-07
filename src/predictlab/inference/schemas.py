"""Schemas, data models, and validation definitions for production inference."""

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

RAW_PREDICTORS = [
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
]

VALID_CATEGORIES = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1, "0", "1"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


class InferenceError(Exception):
    """Base exception for inference engine failures."""


class InferenceValidationError(InferenceError):
    """Raised when input records violate schema or domain constraints."""


class ModelArtifactError(InferenceError):
    """Raised when loading or validating a serialized model artifact fails."""


class CustomerInputRecord(BaseModel):
    """Pydantic validation model for raw customer predictors."""

    model_config = ConfigDict(extra="forbid")

    gender: str = Field(description="Customer gender ('Female' or 'Male')")
    SeniorCitizen: int | str = Field(description="Senior citizen status (0 or 1)")
    Partner: str = Field(description="Partner status ('Yes' or 'No')")
    Dependents: str = Field(description="Dependents status ('Yes' or 'No')")
    tenure: int | float = Field(ge=0, description="Months customer has stayed with company (>= 0)")
    PhoneService: str = Field(description="Phone service ('Yes' or 'No')")
    MultipleLines: str = Field(description="Multiple lines ('Yes', 'No', or 'No phone service')")
    InternetService: str = Field(description="Internet service provider ('DSL', 'Fiber optic', or 'No')")
    OnlineSecurity: str = Field(description="Online security ('Yes', 'No', or 'No internet service')")
    OnlineBackup: str = Field(description="Online backup ('Yes', 'No', or 'No internet service')")
    DeviceProtection: str = Field(description="Device protection ('Yes', 'No', or 'No internet service')")
    TechSupport: str = Field(description="Tech support ('Yes', 'No', or 'No internet service')")
    StreamingTV: str = Field(description="Streaming TV ('Yes', 'No', or 'No internet service')")
    StreamingMovies: str = Field(description="Streaming movies ('Yes', 'No', or 'No internet service')")
    Contract: str = Field(description="Contract term ('Month-to-month', 'One year', or 'Two year')")
    PaperlessBilling: str = Field(description="Paperless billing ('Yes' or 'No')")
    PaymentMethod: str = Field(description="Payment method")
    MonthlyCharges: float = Field(ge=0.0, description="Monthly charges billed to customer (>= 0)")
    TotalCharges: float | str = Field(description="Total charges billed (float >= 0 or whitespace string for 0 tenure)")

    @field_validator("SeniorCitizen")
    @classmethod
    def validate_senior(cls, v: int | str) -> int:
        if v in (0, "0"):
            return 0
        if v in (1, "1"):
            return 1
        raise ValueError(f"SeniorCitizen must be 0 or 1, got {v}")

    @field_validator("tenure")
    @classmethod
    def validate_tenure_finite(cls, v: int | float) -> float:
        val = float(v)
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"tenure must be a non-negative finite number, got {v}")
        return val

    @field_validator("MonthlyCharges")
    @classmethod
    def validate_monthly_charges(cls, v: float) -> float:
        val = float(v)
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"MonthlyCharges must be a non-negative finite number, got {v}")
        return val

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v: float | str) -> float | str:
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return " "
            try:
                num = float(s)
                if not np.isfinite(num) or num < 0:
                    raise ValueError(f"TotalCharges must be non-negative, got {v}")
                return num
            except ValueError as e:
                raise ValueError(f"TotalCharges string cannot be parsed as numeric: '{v}'") from e
        num = float(v)
        if not np.isfinite(num) or num < 0:
            raise ValueError(f"TotalCharges must be non-negative, got {v}")
        return num


class SinglePredictionOutput(BaseModel):
    """Structured output for a single customer churn prediction."""

    model_version: str = Field(description="Loaded model artifact identifier / version")
    model_name: str = Field(description="Descriptive name of the model")
    calibration_method: str = Field(description="Calibration strategy applied ('sigmoid', 'isotonic', 'uncalibrated')")
    estimated_probability: float = Field(description="Model-estimated churn probability in [0, 1]")
    predicted_class: int = Field(description="Predicted binary class (1 = Churned, 0 = Retained)")
    predicted_label: str = Field(description="'Churned' or 'Retained'")
    applied_threshold: float = Field(description="Decision threshold applied")
    risk_tier: str | None = Field(default=None, description="Assigned risk tier ('Low', 'Medium', 'High') if tiering enabled")
    explanation: dict[str, Any] | None = Field(default=None, description="Local SHAP feature attributions if requested")


class BatchPredictionRecord(BaseModel):
    """Individual record result within a batch prediction."""

    record_index: int = Field(description="Original 0-based index of the input record")
    customer_id: str | None = Field(default=None, description="Optional customerID if provided in raw payload")
    is_valid: bool = Field(description="True if record successfully predicted, False if validation failed")
    prediction: SinglePredictionOutput | None = Field(default=None, description="Prediction output if successful")
    error_message: str | None = Field(default=None, description="Validation error message if failed")


class BatchPredictionOutput(BaseModel):
    """Structured output for batch customer churn predictions."""

    model_version: str = Field(description="Loaded model artifact identifier")
    total_records: int = Field(description="Number of records submitted")
    successful_predictions: int = Field(description="Number of successfully scored records")
    failed_predictions: int = Field(description="Number of invalid records")
    error_handling_mode: Literal["atomic", "per_record"] = Field(description="Applied batch validation mode")
    predictions: list[BatchPredictionRecord] = Field(description="Ordered list of prediction records matching input order")

