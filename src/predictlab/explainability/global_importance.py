"""Global feature importance evaluation via training-only permutation importance."""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from predictlab.logging_config import get_logger

logger = get_logger(__name__)


class GlobalImportanceResult(BaseModel):
    """Encapsulates permutation importance rankings and methodological metadata."""

    metric_used: str = Field(default="average_precision", description="Scoring metric used for permutation tests")
    n_repeats: int = Field(default=5, description="Number of permutation iterations per feature per fold")
    n_splits: int = Field(default=5, description="Number of cross-validation folds")
    evaluation_type: str = Field(
        default="cross_validated_out_of_fold",
        description="Type of evaluation ('cross_validated_out_of_fold' vs 'in_sample_training')",
    )
    feature_names: list[str] = Field(description="Names of evaluated features")
    importances_mean: dict[str, float] = Field(description="Mean score decrease per feature across all fold repeats")
    importances_std: dict[str, float] = Field(description="Standard deviation of score decrease across repeats")
    normalized_importances: dict[str, float] = Field(description="Normalized relative importance scores (sum=1.0)")
    methodological_notes: str = Field(description="Scientific interpretation rules and limitations")


def compute_cross_validated_permutation_importance(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    scoring: str = "average_precision",
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> tuple[GlobalImportanceResult, pd.DataFrame]:
    """Compute permutation feature importances using leakage-free cross-validation.

    Methodological Definition & Limitations:
    1. Cross-Validated Validation Scoring: The pipeline is fitted within each of `n_splits` training
       folds, and permutation importance is evaluated strictly on the corresponding held-out validation
       split. This prevents optimistic bias inherent in in-sample feature shuffling.
    2. Model Reliance vs Causality: Permutation importance measures model reliance through predictive-score
       degradation after feature values are randomly permuted. It does NOT establish real-world causality,
       treatment effects, or intrinsic feature value.
    3. Collinear Feature Dilution: When correlated predictors are present (e.g. tenure and TotalCharges,
       or internet add-on service flags), predictive signal is shared across features, diluting individual
       importance scores. Lower importance for a collinear feature does not mean it lacks predictive power.
    4. Non-Directional Metric: Permutation importance is strictly non-negative (score decrease) and does
       NOT indicate whether a feature increases or decreases churn risk. Directional interpretation must
       be derived from separate evidence (e.g. bivariate risk tables or local Shapley attributions).
    5. Partition Isolation: Evaluated strictly on the training partition (N=5,634). The holdout test set
       (N=1,409) is strictly untouched.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted or fitted base pipeline (cloned for each fold).
    X_train : pd.DataFrame
        Training feature matrix (N=5,634).
    y_train : pd.Series or np.ndarray
        Training target vector.
    scoring : str, default='average_precision'
        Evaluation metric (Average Precision / PR-AUC).
    n_splits : int, default=5
        Number of cross-validation folds.
    n_repeats : int, default=5
        Number of shuffling iterations per feature within each fold.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    tuple[GlobalImportanceResult, pd.DataFrame]
        Pydantic result object and ranked DataFrame.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    logger.info(
        "Computing %d-fold cross-validated permutation importance (%d repeats/fold, metric='%s')...",
        n_splits,
        n_repeats,
        scoring,
    )
    y_arr = np.asarray(y_train).astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_importances: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_arr)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_arr[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_arr[va_idx]

        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_tr, y_tr)

        if "preprocessor" in fold_pipe.named_steps:
            preprocessor = fold_pipe.named_steps["preprocessor"]
            classifier = fold_pipe.named_steps["classifier"]
            X_va_trans = preprocessor.transform(X_va)
            if feature_names is None:
                if hasattr(preprocessor, "get_feature_names_out"):
                    feature_names = list(preprocessor.get_feature_names_out())
                else:
                    feature_names = [f"feature_{i}" for i in range(X_va_trans.shape[1])]
            estimator_to_test = classifier
            X_eval = X_va_trans
        else:
            if feature_names is None:
                feature_names = list(X_train.columns)
            estimator_to_test = fold_pipe
            X_eval = X_va

        perm = permutation_importance(
            estimator=estimator_to_test,
            X=X_eval,
            y=y_va,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )
        fold_importances.append(perm.importances)

    assert feature_names is not None
    # Concatenate across all folds: shape (n_features, n_splits * n_repeats)
    all_repeats = np.concatenate(fold_importances, axis=1)
    means = np.mean(all_repeats, axis=1)
    stds = np.std(all_repeats, axis=1)

    # Clip negative values resulting from noise to 0.0 for normalized percentages
    clipped_means = np.maximum(0.0, means)
    sum_means = float(np.sum(clipped_means))
    norm_means = (clipped_means / sum_means) if sum_means > 0 else clipped_means

    records = []
    for name, m, s, norm in zip(feature_names, means, stds, norm_means, strict=True):
        clean_name = name.replace("num__", "").replace("cat__", "")
        records.append(
            {
                "feature": name,
                "clean_feature": clean_name,
                "importance_mean": round(float(m), 5),
                "importance_std": round(float(s), 5),
                "normalized_importance": round(float(norm), 5),
            }
        )

    df_ranked = (
        pd.DataFrame(records)
        .sort_values(by="importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    df_ranked["rank"] = df_ranked.index + 1

    notes = (
        f"Cross-validated permutation importance ({n_splits} folds, {n_repeats} repeats/fold) "
        "evaluating Average Precision drop strictly on held-out validation splits. "
        "Permutation importance quantifies model reliance, NOT causal mechanism or intrinsic feature value. "
        "Collinear features share predictive signal and dilute individual importance rankings. "
        "Permutation importance is non-directional; directional associations must be derived separately."
    )

    result = GlobalImportanceResult(
        metric_used=scoring,
        n_repeats=n_repeats,
        n_splits=n_splits,
        evaluation_type="cross_validated_out_of_fold",
        feature_names=feature_names,
        importances_mean={r["feature"]: r["importance_mean"] for r in records},
        importances_std={r["feature"]: r["importance_std"] for r in records},
        normalized_importances={r["feature"]: r["normalized_importance"] for r in records},
        methodological_notes=notes,
    )

    return result, df_ranked


def compute_training_permutation_importance(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[GlobalImportanceResult, pd.DataFrame]:
    """Backward-compatible wrapper executing cross-validated permutation importance."""
    return compute_cross_validated_permutation_importance(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        scoring=scoring,
        n_splits=5,
        n_repeats=max(2, n_repeats // 2),
        random_state=random_state,
    )

