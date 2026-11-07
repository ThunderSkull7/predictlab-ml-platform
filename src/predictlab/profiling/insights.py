"""Diagnostic insights, segmentation, and inferential hypothesis testing module."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg_correction(p_values: Sequence[float]) -> list[float]:
    """Apply the Benjamini-Hochberg False Discovery Rate (FDR) procedure to adjust p-values.

    Parameters
    ----------
    p_values : Sequence[float]
        Raw p-values from multiple hypothesis tests.

    Returns
    -------
    list[float]
        Adjusted p-values (q-values) constrained to [0.0, 1.0].
    """
    n = len(p_values)
    if n == 0:
        return []

    p_arr = np.asarray(p_values, dtype=float)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]

    # Cumulative minimum from the right for monotonicity
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj_val = (sorted_p[i] * n) / rank
        running_min = min(running_min, adj_val)
        adjusted[i] = min(1.0, max(0.0, running_min))

    # Revert to original order
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return [float(x) for x in out]


def compute_cramers_v(contingency_table: pd.DataFrame) -> float:
    """Compute bias-corrected Cramér's V statistic for categorical association.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Two-way contingency table of frequencies.

    Returns
    -------
    float
        Bias-corrected Cramér's V coefficient in range [0.0, 1.0].
    """
    chi2, _, _, _ = stats.chi2_contingency(contingency_table, correction=False)
    n = contingency_table.to_numpy().sum()
    if n <= 1:
        return 0.0

    phi2 = chi2 / n
    r, k = contingency_table.shape

    # Bias correction (Bergsma, 2013)
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    min_dim = min(k_corr - 1, r_corr - 1)
    if min_dim <= 0:
        return 0.0

    return float(np.sqrt(phi2_corr / min_dim))


def compute_segment_churn_rates(
    df: pd.DataFrame,
    group_cols: list[str] | str,
    target_col: str = "Churn",
    positive_label: str = "Yes",
) -> pd.DataFrame:
    """Calculate segment volume, churn volume, churn rate, and relative risk vs overall baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    group_cols : Union[list[str], str]
        Column name or list of column names defining the segment.
    target_col : str
        Target column name.
    positive_label : str
        Label representing the churned state.

    Returns
    -------
    pd.DataFrame
        Aggregated table with segment definitions, customer counts, churn counts,
        segment churn rate (%), and relative risk ratio.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Grouping column '{c}' not found in dataframe.")

    baseline_churn_rate = float((df[target_col] == positive_label).mean())

    grouped = df.groupby(cols, observed=False)[target_col]
    total_counts = grouped.count()
    churn_counts = grouped.apply(lambda s: (s == positive_label).sum())

    records = []
    for idx in total_counts.index:
        total_n = int(total_counts[idx])
        churn_n = int(churn_counts[idx])
        rate_pct = float((churn_n / total_n) * 100) if total_n > 0 else 0.0
        rel_risk = float((rate_pct / 100) / baseline_churn_rate) if baseline_churn_rate > 0 else 1.0

        row = {}
        if len(cols) == 1:
            row[cols[0]] = str(idx)
        else:
            for i, col_name in enumerate(cols):
                row[col_name] = str(idx[i])

        row.update(
            {
                "total_customers": total_n,
                "churned_customers": churn_n,
                "retained_customers": total_n - churn_n,
                "churn_rate_pct": round(rate_pct, 2),
                "relative_risk": round(rel_risk, 3),
            }
        )
        records.append(row)

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        res_df = res_df.sort_values(by="churn_rate_pct", ascending=False).reset_index(drop=True)
    return res_df


def compute_categorical_associations(
    df: pd.DataFrame,
    cat_cols: list[str],
    target_col: str = "Churn",
    alpha: float = 0.05,
    apply_fdr: bool = True,
) -> pd.DataFrame:
    """Run Chi-Square tests of independence and compute bias-corrected Cramér's V.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    cat_cols : list[str]
        Categorical columns to evaluate against target.
    target_col : str
        Target column name.
    alpha : float
        Nominal significance threshold.
    apply_fdr : bool
        Whether to compute Benjamini-Hochberg FDR adjusted p-values.

    Returns
    -------
    pd.DataFrame
        Results table with feature, chi2, dof, p_value, fdr_p_value, cramers_v,
        and significant flag.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    results = []
    for col in cat_cols:
        if col not in df.columns or col == target_col or col == "customerID":
            continue

        ct = pd.crosstab(df[col], df[target_col])
        if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        chi2, p_val, dof, _ = stats.chi2_contingency(ct)
        cramers_v = compute_cramers_v(ct)

        results.append(
            {
                "feature": col,
                "chi2_statistic": round(float(chi2), 3),
                "degrees_of_freedom": int(dof),
                "raw_p_value": float(p_val),
                "cramers_v": round(float(cramers_v), 4),
            }
        )

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return res_df

    if apply_fdr:
        res_df["fdr_p_value"] = benjamini_hochberg_correction(res_df["raw_p_value"].tolist())
        res_df["significant"] = [bool(p < alpha) for p in res_df["fdr_p_value"]]
    else:
        res_df["fdr_p_value"] = res_df["raw_p_value"]
        res_df["significant"] = [bool(p < alpha) for p in res_df["raw_p_value"]]

    return res_df.sort_values(by="cramers_v", ascending=False).reset_index(drop=True)


def compute_numerical_churn_tests(
    df: pd.DataFrame,
    num_cols: list[str],
    target_col: str = "Churn",
    positive_label: str = "Yes",
    negative_label: str = "No",
    apply_fdr: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Perform non-parametric (Mann-Whitney U) and parametric (Welch's t) tests on numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    num_cols : list[str]
        List of numerical feature names.
    target_col : str
        Target column name.
    positive_label : str
        Churn label.
    negative_label : str
        Retained label.
    apply_fdr : bool
        Whether to adjust p-values using the Benjamini-Hochberg procedure.
    alpha : float
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        Table with means, medians, test statistics, effect sizes, and p-values.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    records = []
    for col in num_cols:
        if col not in df.columns:
            continue

        raw = df[col]
        s = pd.to_numeric(raw.replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        mask_pos = (df[target_col] == positive_label) & s.notna()
        mask_neg = (df[target_col] == negative_label) & s.notna()

        pos_vals = s[mask_pos]
        neg_vals = s[mask_neg]

        n_pos = len(pos_vals)
        n_neg = len(neg_vals)

        if n_pos == 0 or n_neg == 0:
            continue

        # Mann-Whitney U test
        u_res = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
        # Rank-biserial correlation effect size: r = 1 - (2U / (n1 * n2))
        rank_biserial = 1.0 - (2.0 * float(u_res.statistic) / (n_pos * n_neg))

        # Welch's t-test (unequal variance)
        t_res = stats.ttest_ind(pos_vals, neg_vals, equal_var=False)

        # Pooled standard deviation for Cohen's d
        pooled_std = np.sqrt(
            ((n_pos - 1) * pos_vals.var() + (n_neg - 1) * neg_vals.var()) / (n_pos + n_neg - 2)
        )
        cohens_d = float((pos_vals.mean() - neg_vals.mean()) / pooled_std) if pooled_std > 0 else 0.0

        records.append(
            {
                "feature": col,
                "retained_count": n_neg,
                "churned_count": n_pos,
                "retained_mean": round(float(neg_vals.mean()), 2),
                "churned_mean": round(float(pos_vals.mean()), 2),
                "retained_median": round(float(neg_vals.median()), 2),
                "churned_median": round(float(pos_vals.median()), 2),
                "mann_whitney_u": float(u_res.statistic),
                "mann_whitney_p": float(u_res.pvalue),
                "rank_biserial_r": round(float(rank_biserial), 4),
                "welch_t_stat": round(float(t_res.statistic), 3),
                "welch_t_p": float(t_res.pvalue),
                "cohens_d": round(cohens_d, 4),
            }
        )

    res_df = pd.DataFrame(records)
    if res_df.empty:
        return res_df

    if apply_fdr:
        res_df["mann_whitney_fdr_p"] = benjamini_hochberg_correction(res_df["mann_whitney_p"].tolist())
        res_df["welch_t_fdr_p"] = benjamini_hochberg_correction(res_df["welch_t_p"].tolist())
        res_df["significant"] = [bool(p < alpha) for p in res_df["mann_whitney_fdr_p"]]
    else:
        res_df["mann_whitney_fdr_p"] = res_df["mann_whitney_p"]
        res_df["welch_t_fdr_p"] = res_df["welch_t_p"]
        res_df["significant"] = [bool(p < alpha) for p in res_df["mann_whitney_p"]]

    return res_df.sort_values(by="rank_biserial_r", key=abs, ascending=False).reset_index(drop=True)


def compute_correlation_matrix(
    df: pd.DataFrame,
    num_cols: list[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute pairwise rank or linear correlation between continuous/discrete numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    num_cols : list[str]
        List of numerical feature names.
    method : str
        Correlation method ('spearman' or 'pearson').

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.
    """
    clean_dict = {}
    for col in num_cols:
        if col in df.columns:
            clean_dict[col] = pd.to_numeric(
                df[col].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
            )

    sub_df = pd.DataFrame(clean_dict).dropna()
    return sub_df.corr(method=method).round(4)
