"""Profiling, data quality audit, and analytical insights package for PredictLab."""

from predictlab.profiling.insights import (
    benjamini_hochberg_correction,
    compute_categorical_associations,
    compute_correlation_matrix,
    compute_cramers_v,
    compute_numerical_churn_tests,
    compute_segment_churn_rates,
)
from predictlab.profiling.quality import (
    ColumnQualityProfile,
    DataQualityReport,
    TotalChargesInvestigation,
    detect_whitespace_entries,
    investigate_total_charges,
    profile_data_quality,
)
from predictlab.profiling.summary import (
    compute_categorical_summary,
    compute_dataset_overview,
    compute_numeric_summary,
    compute_target_distribution,
)
from predictlab.profiling.visuals import generate_all_eda_figures

__all__ = [
    "profile_data_quality",
    "detect_whitespace_entries",
    "investigate_total_charges",
    "ColumnQualityProfile",
    "TotalChargesInvestigation",
    "DataQualityReport",
    "compute_dataset_overview",
    "compute_numeric_summary",
    "compute_categorical_summary",
    "compute_target_distribution",
    "compute_segment_churn_rates",
    "compute_categorical_associations",
    "compute_numerical_churn_tests",
    "compute_correlation_matrix",
    "compute_cramers_v",
    "benjamini_hochberg_correction",
    "generate_all_eda_figures",
]
