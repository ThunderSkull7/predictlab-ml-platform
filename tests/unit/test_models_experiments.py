"""Unit tests for feature ablation studies and comparative experiment execution."""

from sklearn.dummy import DummyClassifier

from predictlab.data.load import load_raw_dataset
from predictlab.models.experiments import run_feature_ablation_study
from predictlab.preprocessing.split import split_dataset


def test_feature_ablation_study_structure() -> None:
    """Test feature ablation study runs through all 5 configurations on a subsample."""
    df = load_raw_dataset()
    X_train, _, y_train, _, _ = split_dataset(df, save_manifest=False)

    # Use small subset and dummy classifier for instantaneous execution
    X_sub = X_train.iloc[:100]
    y_sub = y_train.iloc[:100]

    def fast_estimator() -> DummyClassifier:
        return DummyClassifier(strategy="prior")

    results, df_deltas = run_feature_ablation_study(
        X_train=X_sub,
        y_train=y_sub,
        estimator_factory=fast_estimator,
        n_splits=2,
    )

    assert len(results) == 5
    assert len(df_deltas) == 5

    expected_configs = [
        "Baseline (47 features)",
        "Full Experimental (52 features)",
        "Ablate ChargesRatio (50 features)",
        "Ablate AutomaticPayment (51 features)",
        "Ablate Service Subtotals (50 features)",
    ]

    observed_configs = [r.model_name for r in results]
    assert observed_configs == expected_configs

    # Verify delta columns exist
    assert "Δ PR-AUC" in df_deltas.columns
    assert "Δ ROC-AUC" in df_deltas.columns
    assert "Δ Log Loss" in df_deltas.columns
    # Baseline delta against itself must be 0.0
    assert df_deltas.iloc[0]["Δ PR-AUC"] == 0.0

