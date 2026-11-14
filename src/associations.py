"""Association measures for the Q2 associated-factors analysis.

All functions use observed pairs only: a record is used when both values are
present. Nothing is imputed. Each association function returns a dict with:

- ``estimate``: the association statistic, or NaN when it is undefined
  (for example, a constant variable or too few usable pairs);
- ``n_used``: the number of complete pairs used;
- ``missing_pair_fraction``: the fraction of input records dropped because
  at least one value was missing.

These functions compute measures only. They do not choose predictors.
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd


def _observed_pairs(x, y) -> tuple[pd.Series, pd.Series, int, float]:
    """Align x and y, keep complete pairs, and return them with n_used and the dropped fraction."""
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length ({len(x)} != {len(y)})")
    keep = x.notna() & y.notna()
    n_total = len(x)
    n_used = int(keep.sum())
    missing_pair_fraction = float(1 - n_used / n_total) if n_total else float("nan")
    return x[keep], y[keep], n_used, missing_pair_fraction


def _result(estimate, n_used, missing_pair_fraction, **extra) -> dict:
    out = {
        "estimate": float(estimate) if estimate is not None else float("nan"),
        "n_used": int(n_used),
        "missing_pair_fraction": missing_pair_fraction,
    }
    out.update(extra)
    return out


def _to_numeric(values: pd.Series, ordinal_map: Mapping | None, name: str) -> pd.Series:
    """Return numeric values, applying an explicit ordinal mapping to non-numeric input."""
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    if ordinal_map is None:
        raise ValueError(f"{name} is not numeric; pass an explicit ordinal_map")
    unmapped = sorted(set(values.astype(str)) - {str(k) for k in ordinal_map})
    if unmapped:
        raise ValueError(f"{name} has categories missing from ordinal_map: {unmapped}")
    lookup = {str(k): v for k, v in ordinal_map.items()}
    return values.astype(str).map(lookup).astype(float)


def spearman_assoc(x, y, ordinal_map: Mapping | None = None) -> dict:
    """Spearman rank correlation for observed pairs.

    Numeric inputs are used as they are. A non-numeric input (for example an
    ordered category such as ``GeneralHealth``) is converted with
    ``ordinal_map``, a dict from category to rank, for example
    ``{"Poor": 1, "Fair": 2, "Good": 3, "Very good": 4, "Excellent": 5}``.
    The mapping fixes the direction, so the sign of the estimate is kept and
    can be interpreted: positive means higher mapped values go with higher
    values of the other variable. A mapping is required for non-numeric input,
    and every observed category must appear in it.

    Ties receive average ranks. The estimate is NaN if fewer than 3 pairs
    remain or either variable is constant among them.
    """
    x_obs, y_obs, n_used, missing_fraction = _observed_pairs(x, y)
    x_num = _to_numeric(x_obs, ordinal_map, "x")
    y_num = _to_numeric(y_obs, ordinal_map, "y")
    if n_used < 3 or x_num.nunique() < 2 or y_num.nunique() < 2:
        return _result(np.nan, n_used, missing_fraction)
    rx = x_num.rank(method="average").to_numpy()
    ry = y_num.rank(method="average").to_numpy()
    estimate = np.corrcoef(rx, ry)[0, 1]
    return _result(estimate, n_used, missing_fraction)


def cramers_v_bc(x, y) -> dict:
    """Bias-corrected Cramér's V (Bergsma, 2013) for two categorical variables.

    With ``n`` observed pairs, ``r`` rows and ``k`` columns in the contingency
    table and ``phi2 = chi2 / n``::

        phi2_corr = max(0, phi2 - (k - 1)(r - 1) / (n - 1))
        r_corr = r - (r - 1)^2 / (n - 1)
        k_corr = k - (k - 1)^2 / (n - 1)
        V = sqrt(phi2_corr / min(k_corr - 1, r_corr - 1))

    V lies between 0 and 1 and has no sign. It is NaN when either variable
    has fewer than two observed categories or the denominator is not positive.
    The returned dict also contains ``table``, the observed contingency table.

    Reference: Bergsma, W. (2013). A bias-correction for Cramér's V and
    Tschuprow's T. Journal of the Korean Statistical Society, 42(3), 323-328.
    """
    x_obs, y_obs, n_used, missing_fraction = _observed_pairs(x, y)
    table = pd.crosstab(x_obs.astype(str).rename("x"), y_obs.astype(str).rename("y"))
    r, k = table.shape
    if n_used < 2 or r < 2 or k < 2:
        return _result(np.nan, n_used, missing_fraction, table=table)

    observed = table.to_numpy(dtype=float)
    expected = observed.sum(axis=1, keepdims=True) * observed.sum(axis=0, keepdims=True) / n_used
    chi2 = float(((observed - expected) ** 2 / expected).sum())

    phi2_corr = max(0.0, chi2 / n_used - (k - 1) * (r - 1) / (n_used - 1))
    r_corr = r - (r - 1) ** 2 / (n_used - 1)
    k_corr = k - (k - 1) ** 2 / (n_used - 1)
    denominator = min(k_corr - 1, r_corr - 1)
    estimate = np.sqrt(phi2_corr / denominator) if denominator > 0 else np.nan
    return _result(estimate, n_used, missing_fraction, table=table)


def correlation_ratio(categories, values) -> dict:
    """Correlation ratio (eta) of a numeric response across categories.

    ``values`` is the numeric response and ``categories`` the grouping::

        eta = sqrt(between-group sum of squares / total sum of squares)

    Eta lies between 0 and 1 and has no sign. It is NaN when fewer than two
    categories are observed or the numeric values are constant.
    """
    cat_obs, val_obs, n_used, missing_fraction = _observed_pairs(categories, values)
    val_obs = val_obs.astype(float)
    if n_used < 2 or cat_obs.nunique() < 2:
        return _result(np.nan, n_used, missing_fraction)
    total_ss = float(((val_obs - val_obs.mean()) ** 2).sum())
    if total_ss == 0:
        return _result(np.nan, n_used, missing_fraction)
    groups = val_obs.groupby(cat_obs.astype(str).to_numpy())
    between_ss = float((groups.size() * (groups.mean() - val_obs.mean()) ** 2).sum())
    return _result(np.sqrt(between_ss / total_ss), n_used, missing_fraction)


def sparse_groups(x, min_n: int = 30) -> pd.DataFrame:
    """Count usable (non-missing) records per category and flag sparse ones.

    Returns a DataFrame with columns ``category``, ``n`` and ``sparse``
    (True when ``n < min_n``), sorted by ``n`` ascending. Missing values are
    not a category here because the association functions drop them.
    """
    counts = pd.Series(x).dropna().astype(str).value_counts()
    out = counts.rename_axis("category").reset_index(name="n")
    out["sparse"] = out["n"] < min_n
    return out.sort_values(["n", "category"]).reset_index(drop=True)


def group_bootstrap(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], float | dict],
    group_col: str = "duplicate_group",
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap interval that resamples whole groups.

    Records sharing ``group_col`` (identical survey responses) are resampled
    together, so repeated records never appear in a resample without the rest
    of their group. A record with a missing group id is treated as its own
    group. ``func`` takes a DataFrame and returns a number, or a dict with an
    ``estimate`` key (so the association functions above can be used
    directly). Resamples where ``func`` returns NaN are skipped.

    Returns ``estimate`` (on the full data), ``ci_lower`` and ``ci_upper``
    (the alpha/2 and 1 - alpha/2 percentiles), ``n_boot_valid`` and
    ``n_groups``.
    """

    def evaluate(frame: pd.DataFrame) -> float:
        value = func(frame)
        return float(value["estimate"] if isinstance(value, dict) else value)

    codes, _ = pd.factorize(df[group_col], use_na_sentinel=True)
    missing = codes < 0
    if missing.any():
        codes = codes.copy()
        codes[missing] = codes.max() + 1 + np.arange(missing.sum())
    n_groups = int(codes.max() + 1) if len(codes) else 0

    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        chosen = rng.integers(0, n_groups, size=n_groups)
        lengths = sizes[chosen]
        # Row positions of every chosen group, in order.
        offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        rows = order[np.repeat(starts[chosen], lengths) + offsets]
        value = evaluate(df.iloc[rows])
        if not np.isnan(value):
            estimates.append(value)

    if estimates:
        lower, upper = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    else:
        lower = upper = np.nan
    return {
        "estimate": evaluate(df),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_boot_valid": len(estimates),
        "n_groups": n_groups,
    }
