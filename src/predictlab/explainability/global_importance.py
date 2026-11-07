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
    n_repeats: int = Field(default=10, description="Number of permutation iterations per feature")
    feature_names: list[str] = Field(description="Names of evaluated features")
    importances_mean: dict[str, float] = Field(description="Mean score decrease per feature")
    importances_std: dict[str, float] = Field(description="Standard deviation of score decrease across repeats")
    normalized_importances: dict[str, float] = Field(description="Normalized relative importance scores (sum=1.0)")
    methodological_notes: str = Field(description="Scientific interpretation rules and limitations")


def compute_training_permutation_importance(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[GlobalImportanceResult, pd.DataFrame]:
    """Compute permutation feature importances strictly on the training partition.

    Methodological Definition & Limitations:
    1. Permutation importance measures the decrease in a model's predictive score after a feature's
       values are shuffled. It estimates model reliance on that feature under the selected scoring procedure;
       it does not establish causality or conditional dependence.
    2. Collinear Dilution: When correlated predictors are present (e.g. tenure and TotalCharges, or
       component service flags and OptionalServiceCount), correlated predictors can dilute or distort
       individual feature importance estimates because the information remains partially accessible to
       the model through collinear unpermuted features.
    3. Partition Isolation: Evaluated STRICTLY on X_train, y_train. Holdout test labels are never used.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted scikit-learn pipeline (preprocessing + estimator).
    X_train : pd.DataFrame
        Training feature matrix (N=5,634).
    y_train : pd.Series or np.ndarray
        Training target vector.
    scoring : str, default='average_precision'
        Evaluation metric (Average Precision / PR-AUC).
    n_repeats : int, default=10
        Number of shuffling iterations per feature.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    tuple[GlobalImportanceResult, pd.DataFrame]
        Pydantic result object and ranked DataFrame.
    """
    logger.info(
        "Computing training permutation importance (%d repeats, metric='%s')...",
        n_repeats,
        scoring,
    )
    y_arr = np.asarray(y_train).astype(int)

    # Transform X_train through pipeline preprocessor to evaluate transformed feature space
    if "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        X_trans = preprocessor.transform(X_train)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = list(preprocessor.get_feature_names_out())
        else:
            feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]
        estimator_to_test = classifier
        X_eval = X_trans
    else:
        feature_names = list(X_train.columns)
        estimator_to_test = pipeline
        X_eval = X_train

    perm = permutation_importance(
        estimator=estimator_to_test,
        X=X_eval,
        y=y_arr,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    means = perm.importances_mean
    stds = perm.importances_std

    # Clip negative values resulting from noise to 0.0
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
        "Permutation importance measures the decrease in a model's predictive score after a feature's values "
        "are shuffled. It estimates model reliance on that feature under the selected scoring procedure; "
        "it does not establish causality or conditional dependence. Furthermore, when correlated predictors "
        "are present, correlated predictors can dilute or distort individual feature importance estimates. "
        "Evaluated strictly on training partition (N=5,634)."
    )

    result = GlobalImportanceResult(
        metric_used=scoring,
        n_repeats=n_repeats,
        feature_names=feature_names,
        importances_mean={r["feature"]: r["importance_mean"] for r in records},
        importances_std={r["feature"]: r["importance_std"] for r in records},
        normalized_importances={r["feature"]: r["normalized_importance"] for r in records},
        methodological_notes=notes,
    )

    return result, df_ranked

