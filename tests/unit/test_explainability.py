"""Unit tests for global permutation importance and local TreeSHAP explanations."""

import numpy as np

from predictlab.data.load import load_raw_dataset
from predictlab.explainability.global_importance import compute_training_permutation_importance
from predictlab.explainability.local_attribution import TreeShapExplainer
from predictlab.models.estimators import create_hist_gradient_boosting_pipeline
from predictlab.preprocessing.split import split_dataset


def test_training_permutation_importance() -> None:
    """Test global permutation importance on training sample."""
    df = load_raw_dataset()
    X_train, _, y_train, _, _ = split_dataset(df, save_manifest=False)

    pipe = create_hist_gradient_boosting_pipeline(max_iter=30)
    pipe.fit(X_train.iloc[:150], y_train.iloc[:150])

    res, df_ranked = compute_training_permutation_importance(
        pipeline=pipe,
        X_train=X_train.iloc[:150],
        y_train=y_train.iloc[:150],
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
    )

    assert len(res.feature_names) == 47
    assert len(df_ranked) == 47
    assert "importance_mean" in df_ranked.columns
    assert "importance_std" in df_ranked.columns
    assert "normalized_importance" in df_ranked.columns
    assert "rank" in df_ranked.columns
    assert df_ranked["rank"].iloc[0] == 1


def test_tree_shap_local_explanation() -> None:
    """Test TreeShapExplainer on fitted pipeline generates consistent local attribution."""
    df = load_raw_dataset()
    X_train, _, y_train, _, _ = split_dataset(df, save_manifest=False)

    pipe = create_hist_gradient_boosting_pipeline(max_iter=30)
    pipe.fit(X_train.iloc[:150], y_train.iloc[:150])

    explainer = TreeShapExplainer(pipe)

    sample = X_train.iloc[0]
    explanation = explainer.explain_record(sample, threshold=0.50, top_n=5)

    assert 0.0 <= explanation.model_estimated_probability <= 1.0
    assert explanation.predicted_class in (0, 1)
    assert len(explanation.all_attributions) == 47
    assert len(explanation.top_churn_drivers) <= 5
    assert len(explanation.top_retention_drivers) <= 5

    # Check additive efficiency: sum(shap_values) + base_value approx model_output_margin
    total_shap = sum(a.shap_value for a in explanation.all_attributions)
    expected_margin = explanation.base_value_margin + total_shap
    assert np.isclose(expected_margin, explanation.model_output_margin, atol=1e-3)


def test_consistent_explanations_for_repeated_inputs() -> None:
    """Test deterministic explanations for identical customer records."""
    df = load_raw_dataset()
    X_train, _, y_train, _, _ = split_dataset(df, save_manifest=False)

    pipe = create_hist_gradient_boosting_pipeline(max_iter=30)
    pipe.fit(X_train.iloc[:150], y_train.iloc[:150])

    explainer = TreeShapExplainer(pipe)
    sample = X_train.iloc[5]

    expl_1 = explainer.explain_record(sample, threshold=0.45)
    expl_2 = explainer.explain_record(sample, threshold=0.45)

    assert expl_1.model_estimated_probability == expl_2.model_estimated_probability
    assert expl_1.predicted_class == expl_2.predicted_class
    assert [a.shap_value for a in expl_1.all_attributions] == [
        a.shap_value for a in expl_2.all_attributions
    ]
