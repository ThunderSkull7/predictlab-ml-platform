"""Synthetic test data fixtures for PredictLab unit and integration tests."""

import pandas as pd


def get_synthetic_telco_data(num_rows: int = 5) -> pd.DataFrame:
    """Generate a minimal synthetic Telco Customer Churn dataset for unit testing."""
    records = [
        {
            "customerID": f"TEST-{i:04d}",
            "gender": "Female" if i % 2 == 0 else "Male",
            "SeniorCitizen": 0 if i % 3 != 0 else 1,
            "Partner": "Yes" if i % 2 == 0 else "No",
            "Dependents": "No",
            "tenure": i * 10 + 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic" if i % 2 == 0 else "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month" if i % 2 == 0 else "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.0 + float(i),
            "TotalCharges": f"{(i * 10 + 1) * (70.0 + float(i)):.2f}",
            "Churn": "Yes" if i % 2 == 0 else "No",
        }
        for i in range(num_rows)
    ]
    return pd.DataFrame(records)
