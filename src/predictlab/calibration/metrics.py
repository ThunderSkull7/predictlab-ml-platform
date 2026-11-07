"""Calibration evaluation metrics, Expected Calibration Error, and Brier score decomposition."""

import warnings
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


class CalibrationMetrics(BaseModel):
    """Encapsulates comprehensive probabilistic calibration and discrimination metrics."""

    average_precision: float = Field(description="Area under the Precision-Recall curve")
    roc_auc: float = Field(description="Area under the ROC curve")
    log_loss: float = Field(description="Cross-entropy / Logarithmic loss")
    brier_score: float = Field(description="Overall Brier score loss")
    ece_uniform: float = Field(description="Expected Calibration Error with equal-width binning")
    ece_quantile: float = Field(description="Expected Calibration Error with equal-frequency binning")
    mce_uniform: float = Field(description="Maximum Calibration Error with equal-width binning")
    brier_reliability: float = Field(description="Brier score reliability component (calibration error)")
    brier_resolution: float = Field(description="Brier score resolution component (sorting ability)")
    brier_uncertainty: float = Field(description="Brier score uncertainty component (inherent variance)")
    calibration_intercept: float = Field(description="Calibration intercept from logistic recalibration regression")
    calibration_slope: float = Field(description="Calibration slope from logistic recalibration regression")


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> float:
    """Calculate the Expected Calibration Error (ECE) for binary classification.

    ECE measures the weighted average absolute difference between predicted probabilities
    and empirical observed event frequencies across discretization bins:
    ECE = sum_m (N_m / N) * |acc(B_m) - conf(B_m)|

    Methodological Note on Binning Sensitivity:
    - 'uniform' (equal-width): Creates equal intervals in [0, 1]. In models with low base rates,
      higher-probability bins often suffer from sparse sample support.
    - 'quantile' (equal-frequency): Places equal numbers of samples in each bin, ensuring uniform
      statistical power per bin at the expense of non-uniform interval widths.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Model-estimated positive-class probabilities in [0, 1].
    n_bins : int, default=10
        Number of bins for probability grouping.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Binning strategy.

    Returns
    -------
    float
        Expected Calibration Error in [0, 1].
    """
    y_true_arr = np.asarray(y_true).astype(float)
    y_prob_arr = np.asarray(y_prob).astype(float)

    if len(y_true_arr) == 0:
        return 0.0

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob_arr, bins[1:-1])
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.percentile(y_prob_arr, quantiles * 100)
        bins = np.unique(bins)
        if len(bins) <= 1:
            return float(np.abs(np.mean(y_prob_arr) - np.mean(y_true_arr)))
        bin_ids = np.digitize(y_prob_arr, bins[1:-1])
    else:
        raise ValueError(f"Unknown binning strategy: '{strategy}'. Use 'uniform' or 'quantile'.")

    ece = 0.0
    total_samples = len(y_prob_arr)
    unique_bins = np.unique(bin_ids)

    for b in unique_bins:
        mask = bin_ids == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true_arr[mask])
            bin_conf = np.mean(y_prob_arr[mask])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> float:
    """Calculate the Maximum Calibration Error (MCE) across probability bins."""
    y_true_arr = np.asarray(y_true).astype(float)
    y_prob_arr = np.asarray(y_prob).astype(float)

    if len(y_true_arr) == 0:
        return 0.0

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob_arr, bins[1:-1])
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.percentile(y_prob_arr, quantiles * 100)
        bins = np.unique(bins)
        if len(bins) <= 1:
            return float(np.abs(np.mean(y_prob_arr) - np.mean(y_true_arr)))
        bin_ids = np.digitize(y_prob_arr, bins[1:-1])
    else:
        raise ValueError(f"Unknown binning strategy: '{strategy}'.")

    max_err = 0.0
    for b in np.unique(bin_ids):
        mask = bin_ids == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true_arr[mask])
            bin_conf = np.mean(y_prob_arr[mask])
            err = np.abs(bin_acc - bin_conf)
            if err > max_err:
                max_err = err

    return float(max_err)


def compute_brier_score_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Calculate the Murphy (1973) Brier Score decomposition into Reliability, Resolution, and Uncertainty.

    Brier Score = Reliability - Resolution + Uncertainty (plus within-bin variance).

    Components:
    - Uncertainty = base_rate * (1 - base_rate): The inherent unpredictability of the event,
      dependent solely on sample prevalence.
    - Reliability (Calibration Error) = sum_m (N_m / N) * (mean_prob_m - mean_obs_m)^2:
      Measures deviation of predictions from actual observed proportions within bins (closer to 0 is better).
    - Resolution = sum_m (N_m / N) * (mean_obs_m - base_rate)^2:
      Measures the model's ability to sort instances into distinct bins differing from the base rate
      (higher is better).

    Methodological Limitation:
    In discrete binning, predictions within each bin exhibit variance, so:
    Brier = Reliability - Resolution + Uncertainty + Within_Bin_Variance.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Model-estimated positive-class probabilities.
    n_bins : int, default=10
        Number of equal-width bins in [0, 1].

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'brier_score', 'reliability', 'resolution', 'uncertainty'.
    """
    y_true_arr = np.asarray(y_true).astype(float)
    y_prob_arr = np.asarray(y_prob).astype(float)
    n = len(y_true_arr)

    if n == 0:
        return {
            "brier_score": 0.0,
            "reliability": 0.0,
            "resolution": 0.0,
            "uncertainty": 0.0,
        }

    base_rate = float(np.mean(y_true_arr))
    uncertainty = base_rate * (1.0 - base_rate)
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins[1:-1])

    reliability = 0.0
    resolution = 0.0

    for b in np.unique(bin_ids):
        mask = bin_ids == b
        count = np.sum(mask)
        if count > 0:
            p_k = np.mean(y_prob_arr[mask])
            y_k = np.mean(y_true_arr[mask])
            reliability += (count / n) * ((p_k - y_k) ** 2)
            resolution += (count / n) * ((y_k - base_rate) ** 2)

    return {
        "brier_score": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def compute_calibration_intercept_and_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Calculate the calibration intercept (calibration-in-the-large) and calibration slope.

    Evaluated by fitting a univariable logistic regression of outcomes on log-odds:
    logit(P(Y=1)) = alpha + beta * logit(p)

    Interpretation:
    - alpha (Intercept): Ideally 0.0. alpha > 0 indicates systematic underprediction; alpha < 0 indicates overprediction.
    - beta (Slope): Ideally 1.0. beta < 1 indicates predictions are too extreme (overconfident);
      beta > 1 indicates predictions are too conservative (underconfident).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities in [0, 1].

    Returns
    -------
    tuple[float, float]
        (intercept, slope)
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    # Avoid infinite logit values
    eps = 1e-7
    clipped_prob = np.clip(y_prob_arr, eps, 1.0 - eps)
    log_odds = logit(clipped_prob).reshape(-1, 1)

    if len(np.unique(y_true_arr)) < 2:
        return 0.0, 1.0

    # Unregularized logistic regression for unbiased parameter estimation
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(log_odds, y_true_arr)
            intercept = float(lr.intercept_[0])
            slope = float(lr.coef_[0][0])
            return intercept, slope
    except Exception:
        return 0.0, 1.0


def evaluate_full_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute comprehensive calibration, discrimination, and scoring metrics for predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int, default=10
        Number of bins for ECE and Brier decomposition.

    Returns
    -------
    CalibrationMetrics
        Pydantic model containing all metrics.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    eps = 1e-15
    clipped_prob = np.clip(y_prob_arr, eps, 1.0 - eps)

    ap = float(average_precision_score(y_true_arr, y_prob_arr))
    auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    ll = float(log_loss(y_true_arr, clipped_prob, labels=[0, 1]))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    ece_uniform = compute_expected_calibration_error(y_true_arr, y_prob_arr, n_bins=n_bins, strategy="uniform")
    ece_quantile = compute_expected_calibration_error(y_true_arr, y_prob_arr, n_bins=n_bins, strategy="quantile")
    mce_uniform = compute_maximum_calibration_error(y_true_arr, y_prob_arr, n_bins=n_bins, strategy="uniform")

    brier_decomp = compute_brier_score_decomposition(y_true_arr, y_prob_arr, n_bins=n_bins)
    intercept, slope = compute_calibration_intercept_and_slope(y_true_arr, y_prob_arr)

    return CalibrationMetrics(
        average_precision=ap,
        roc_auc=auc,
        log_loss=ll,
        brier_score=brier,
        ece_uniform=ece_uniform,
        ece_quantile=ece_quantile,
        mce_uniform=mce_uniform,
        brier_reliability=brier_decomp["reliability"],
        brier_resolution=brier_decomp["resolution"],
        brier_uncertainty=brier_decomp["uncertainty"],
        calibration_intercept=intercept,
        calibration_slope=slope,
    )

