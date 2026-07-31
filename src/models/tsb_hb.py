from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm


GLOBAL_GROUP = "__global__"

# Column carrying each observation's lag behind the forecast origin of the full
# initialization window. Present only on frames produced by
# :func:`split_for_hyper_estimation`, so that a sub-sample keeps the recency
# weights it would have received inside the undivided window.
LAG_COLUMN = "_origin_lag"


@dataclass
class TSBHBParams:
    # Posterior terms (indexed by unique_id)
    p_posterior: pd.Series
    shrunk_mean_log: pd.Series
    posterior_var_mu: pd.Series
    sigma_sq_process: pd.Series

    # Grouping/hyperparameters
    group_labels: pd.Series
    alpha_by_group: pd.Series
    beta_by_group: pd.Series
    size_global_mean_by_group: pd.Series
    size_sigma_sq_by_group: pd.Series
    size_tau_sq_by_group: pd.Series

    # Sufficient statistics for online updates / uncertainty propagation
    n_obs: pd.Series
    s_obs: pd.Series
    n_pos: pd.Series
    sum_log: pd.Series
    sum_sq_log: pd.Series

    # Optional bootstrap hyperparameter draws keyed by:
    # alpha, beta, size_mu, size_sigma, size_tau
    # Each value is DataFrame indexed by group, columns draw_*
    bootstrap_group_hypers: Optional[Dict[str, pd.DataFrame]] = None
    # Configuration metadata
    group_shrink_strength: float = 0.0
    dynamic_occurrence: bool = False
    occurrence_discount: float = 1.0
    adaptive_prior_strength: bool = False
    prior_strength_min: float = 1.0
    prior_strength_max: float = 1.0
    prior_strength_power: float = 1.0
    item_variance_mode: str = "group"
    item_variance_shrink_strength: float = 20.0
    item_var_log: Optional[pd.Series] = None
    fit_discount: float = 1.0


@dataclass
class TSBHBOnlineState:
    n_obs: pd.Series
    s_obs: pd.Series
    n_eff_occ: pd.Series
    s_eff_occ: pd.Series
    n_pos: pd.Series
    sum_log: pd.Series
    sum_sq_log: pd.Series
    group_labels: pd.Series
    alpha_by_group: pd.Series
    beta_by_group: pd.Series
    size_global_mean_by_group: pd.Series
    size_sigma_sq_by_group: pd.Series
    size_tau_sq_by_group: pd.Series
    bootstrap_group_hypers: Optional[Dict[str, pd.DataFrame]] = None
    dynamic_occurrence: bool = False
    occurrence_discount: float = 1.0
    adaptive_prior_strength: bool = False
    prior_strength_min: float = 1.0
    prior_strength_max: float = 1.0
    prior_strength_power: float = 1.0
    item_variance_mode: str = "group"
    item_variance_shrink_strength: float = 20.0


def _beta_binom_log_marginal(s: float, n: float, alpha: float, beta: float) -> float:
    # Accepts real-valued (s, n): discounted sufficient statistics yield
    # fractional pseudo-counts, for which this is the weighted-likelihood
    # (power-prior) generalization of the Beta-Binomial marginal.
    if alpha <= 0 or beta <= 0 or s < 0 or n < s:
        return -np.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(s + 1)
        - math.lgamma(n - s + 1)
        + math.lgamma(s + alpha)
        + math.lgamma(n - s + beta)
        - math.lgamma(n + alpha + beta)
        - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
    )


def _estimate_beta_hyperparams(counts: pd.DataFrame) -> tuple[float, float]:
    s_arr = counts["s"].astype(float).to_numpy()
    n_arr = counts["n"].astype(float).to_numpy()
    valid = (n_arr > 0) & (s_arr <= n_arr)
    s_arr, n_arr = s_arr[valid], n_arr[valid]
    if len(s_arr) == 0:
        return 1.0, 1.0

    def objective(params: np.ndarray) -> float:
        alpha, beta = float(params[0]), float(params[1])
        if alpha <= 0 or beta <= 0:
            return np.inf
        ll = [_beta_binom_log_marginal(float(si), float(ni), alpha, beta) for si, ni in zip(s_arr, n_arr)]
        return -float(np.sum(ll))

    try:
        result = opt.minimize(
            objective,
            x0=[1.0, 10.0],
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, None)],
        )
    except Exception:
        return 1.0, 1.0
    if not result.success:
        return 1.0, 1.0
    return float(result.x[0]), float(result.x[1])


def _normalize_group_labels(
    unique_ids: pd.Index,
    group_labels: Optional[pd.Series | Dict[str, str]] = None,
) -> pd.Series:
    if group_labels is None:
        return pd.Series(GLOBAL_GROUP, index=unique_ids, dtype=object)

    if isinstance(group_labels, dict):
        mapped = pd.Series(group_labels, dtype=object)
        out = mapped.reindex(unique_ids)
    elif isinstance(group_labels, pd.Series):
        if group_labels.index.equals(unique_ids):
            out = group_labels.copy()
        else:
            out = group_labels.reindex(unique_ids)
            if out.isna().all() and len(group_labels) == len(unique_ids):
                out = pd.Series(group_labels.to_numpy(), index=unique_ids, dtype=object)
    else:
        raise TypeError("group_labels must be None, dict, or pandas Series.")

    out = out.fillna(GLOBAL_GROUP).astype(str)
    return out


def _clip_occurrence_discount(discount: float) -> float:
    try:
        val = float(discount)
    except Exception:
        val = 1.0
    if not np.isfinite(val):
        val = 1.0
    return float(np.clip(val, 1e-6, 1.0))


def _clip_fit_discount(discount: float) -> float:
    try:
        val = float(discount)
    except Exception:
        val = 1.0
    if not np.isfinite(val):
        val = 1.0
    return float(np.clip(val, 1e-6, 1.0))


def _compute_series_stats(
    train_df: pd.DataFrame,
    group_labels: Optional[pd.Series | Dict[str, str]] = None,
    fit_discount: float = 1.0,
) -> pd.DataFrame:
    """Per-series sufficient statistics from the initialization window.

    When ``fit_discount`` < 1, observations are exponentially down-weighted by
    recency: an observation lagged by ``d`` periods behind the forecast origin
    receives weight ``fit_discount ** d``. The resulting weighted pseudo-counts
    give TSB-HB temporal forgetting (obsolescence tracking) while keeping every
    downstream posterior update closed form.
    """
    fit_discount = _clip_fit_discount(fit_discount)
    cols = ["unique_id", "ds", "y"]
    if LAG_COLUMN in train_df.columns:
        cols = cols + [LAG_COLUMN]
    data = train_df[cols].copy()
    data["occ"] = (data["y"] > 0).astype(int)
    data["log_y"] = np.nan
    pos_mask = data["y"] > 0
    data.loc[pos_mask, "log_y"] = np.log(data.loc[pos_mask, "y"].astype(float))

    if fit_discount < 1.0:
        data = data.sort_values(["unique_id", "ds"], kind="stable")
        if LAG_COLUMN in data.columns:
            # Frame is a sub-sample of a longer window: use the lags recorded
            # against the original forecast origin, not positions in this frame.
            lag_from_origin = data[LAG_COLUMN].to_numpy(dtype=float)
        else:
            t_idx = data.groupby("unique_id", sort=False).cumcount()
            series_len = data.groupby("unique_id", sort=False)["y"].transform("size")
            lag_from_origin = (series_len - 1 - t_idx).to_numpy(dtype=float)
        data["w"] = np.power(fit_discount, lag_from_origin)
    else:
        data["w"] = 1.0

    data["w_occ"] = data["w"] * data["occ"]
    data["w_log"] = data["w"] * data["log_y"].fillna(0.0) * data["occ"]
    data["w_sq_log"] = data["w"] * np.square(data["log_y"].fillna(0.0)) * data["occ"]

    g = data.groupby("unique_id", sort=False)
    n_obs = g["w"].sum().astype(float)
    s_obs = g["w_occ"].sum().astype(float)
    n_pos = g["w_occ"].sum().astype(float)
    sum_log = g["w_log"].sum().astype(float)
    sum_sq_log = g["w_sq_log"].sum().astype(float)

    mean_log = (sum_log / n_pos.replace(0, np.nan)).fillna(np.nan)
    var_num = sum_sq_log - n_pos * (mean_log.fillna(0.0) ** 2)
    var_log = (var_num / (n_pos - 1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    groups = _normalize_group_labels(n_obs.index, group_labels)
    stats = pd.DataFrame(
        {
            "n_obs": n_obs,
            "s_obs": s_obs,
            "n_pos": n_pos,
            "sum_log": sum_log,
            "sum_sq_log": sum_sq_log,
            "mean_log": mean_log,
            "var_log": var_log,
            "group": groups.reindex(n_obs.index).astype(str),
        },
        index=n_obs.index,
    )
    return stats


def split_for_hyper_estimation(
    train_df: pd.DataFrame,
    mode: str = "parity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the initialization window into a *structure* half and a *hyper* half.

    Learning the pooling partition and estimating the group hyperparameters on
    the same observations is a post-selection problem: the partition is chosen
    so that within-group items look alike, and the subsequent moment/REML
    estimate of the between-item variance ``tau^2`` then reads that induced
    homogeneity as genuine, collapsing it toward zero. A collapsed ``tau^2``
    drives the credibility weight ``n/(n + sigma^2/tau^2)`` to zero, i.e. every
    item is replaced by its group mean. Estimating the hyperparameters on
    observations that took no part in forming the partition removes the
    dependence.

    Modes:
      - ``parity``: interleaved (even/odd positions). Both halves span the whole
        window, so the recency profile and any drift are preserved in each.
        Correct under the model's within-series exchangeability given
        ``(p_i, mu_i, sigma_i^2)``, and the default for that reason.
      - ``chrono``: first half / second half. Simpler to state, but the halves
        differ in recency, which confounds the split with the forgetting axis.

    Both frames carry :data:`LAG_COLUMN` so discounting still refers to the
    original forecast origin. Returns ``(structure_df, hyper_df)``.
    """
    mode = str(mode).lower()
    if mode not in {"parity", "chrono"}:
        raise ValueError(f"Unknown hyper-split mode: {mode!r} (expected 'parity' or 'chrono')")

    df = train_df.sort_values(["unique_id", "ds"], kind="stable").copy()
    grouped = df.groupby("unique_id", sort=False)
    t_idx = grouped.cumcount()
    series_len = grouped["y"].transform("size")
    df[LAG_COLUMN] = (series_len - 1 - t_idx).astype(float)

    if mode == "parity":
        structure_mask = (t_idx % 2 == 0).to_numpy()
    else:
        structure_mask = (t_idx < np.ceil(series_len.to_numpy(dtype=float) / 2.0)).to_numpy()

    return df.loc[structure_mask].copy(), df.loc[~structure_mask].copy()


def _estimate_size_hyper_from_stats(
    item_stats: pd.DataFrame,
    fallback_mu: float = 0.0,
    fallback_sigma: float = 1.0,
    fallback_tau: float = 1.0,
) -> tuple[float, float, float]:
    pos = item_stats[item_stats["n_pos"] > 0].copy()
    if pos.empty:
        return float(fallback_mu), float(max(fallback_sigma, 1e-6)), float(max(fallback_tau, 1e-6))

    with_var = pos[pos["n_pos"] > 1]
    numerator = float(np.sum((with_var["n_pos"] - 1) * with_var["var_log"]))
    denominator = float(np.sum(with_var["n_pos"] - 1))
    if denominator > 0:
        sigma_sq = numerator / denominator
    else:
        sigma_sq = float(pos["var_log"].mean()) if not pos["var_log"].empty else float(fallback_sigma)
    sigma_sq = float(max(sigma_sq, 1e-6))

    y_i = pos["mean_log"].to_numpy(dtype=float)
    n_i = pos["n_pos"].to_numpy(dtype=float)
    if len(y_i) == 0:
        return float(fallback_mu), sigma_sq, float(max(fallback_tau, 1e-6))
    if len(y_i) == 1:
        mu_hat = float(y_i[0]) if np.isfinite(y_i[0]) else float(fallback_mu)
        return mu_hat, sigma_sq, float(max(fallback_tau, 1e-6))

    observed_var = float(np.var(y_i, ddof=1))
    avg_sampling_var = float(np.mean(sigma_sq / np.maximum(n_i, 1.0)))
    tau_sq_mom = max(observed_var - avg_sampling_var, 1e-6)

    def reml_neg_log_likelihood(tau_sq: float) -> float:
        if tau_sq <= 0:
            return np.inf
        v_i = tau_sq + sigma_sq / np.maximum(n_i, 1.0)
        weights = 1.0 / v_i
        mu_hat = float(np.sum(weights * y_i) / np.sum(weights))
        return float(np.sum(np.log(v_i)) + np.sum((y_i - mu_hat) ** 2 / v_i) + np.log(np.sum(weights)))

    try:
        res = opt.minimize(
            lambda x: reml_neg_log_likelihood(float(x[0])),
            x0=[tau_sq_mom],
            method="L-BFGS-B",
            bounds=[(1e-9, None)],
        )
    except Exception:
        res = None

    if res is None or (not res.success):
        tau_sq = tau_sq_mom
    else:
        tau_sq = max(float(res.x[0]), 1e-6)

    v_i = tau_sq + sigma_sq / np.maximum(n_i, 1.0)
    w_i = 1.0 / v_i
    mu_hat = float(np.sum(w_i * y_i) / np.sum(w_i)) if np.sum(w_i) > 0 else float(np.nanmean(y_i))
    if not np.isfinite(mu_hat):
        mu_hat = float(fallback_mu)
    return mu_hat, sigma_sq, tau_sq


def _credibility_shrink_tau(
    stats: pd.DataFrame,
    size_sigma_by_group: pd.Series,
    size_tau_by_group: pd.Series,
    tau_global: float,
) -> pd.Series:
    """Partially pool the group variance components toward the global one.

    The plug-in REML estimate of the between-item variance ``tau_g^2`` sits on
    the boundary whenever a group's observed spread of item means falls below
    the average sampling variance ``sigma^2/n_i``. Recency discounting makes
    that common: it shrinks the effective ``n_i``, inflating the sampling term,
    while a learned partition shrinks the observed spread by construction. The
    plug-in then reports ``tau_g^2 = 0``, i.e. "the items in this group are
    identical", and the credibility weight ``n/(n + sigma^2/tau^2)`` collapses
    to zero, discarding every item's own history in favour of the group mean.

    The fix is the operation the model already applies to item means, applied
    one level up: treat the per-group estimates as noisy draws around a common
    value and pool them by Buhlmann credibility,

        tau_g^2  <-  w_g * tau_g^2 + (1 - w_g) * tau_0^2,
        w_g      =  V_0 / (V_0 + Var(tau_g^2)),

    with ``Var(tau_g^2)`` the asymptotic variance of the variance-component
    estimator, ``1 / (0.5 * sum_i (tau^2 + sigma^2/n_i)^{-2})``, evaluated at
    the stabilized ``tau_0^2`` so it stays finite on the boundary, and ``V_0``
    the between-group spread net of that estimation noise. Nothing is tuned:
    both quantities are read off the same initialization window. A group whose
    variance component is poorly determined is pulled to the global value; a
    well-determined one keeps its own. With a single pool the map is the
    identity, so global-pool results are untouched.
    """
    labels = [g for g in size_tau_by_group.index if g != GLOBAL_GROUP]
    if len(labels) < 2:
        return size_tau_by_group

    tau_global = float(max(tau_global, 1e-6))
    est_var: dict[str, float] = {}
    for grp in labels:
        sub = stats[(stats["group"].astype(str) == grp) & (stats["n_pos"] > 0)]
        if len(sub) < 2:
            est_var[grp] = np.inf
            continue
        sigma_g = float(max(size_sigma_by_group.get(grp, 1.0), 1e-9))
        s_i = sigma_g / np.maximum(sub["n_pos"].to_numpy(dtype=float), 1.0)
        info = 0.5 * float(np.sum(1.0 / np.square(tau_global + s_i)))
        est_var[grp] = 1.0 / info if info > 1e-12 else np.inf

    tau_hat = np.array([float(size_tau_by_group[g]) for g in labels], dtype=float)
    var_hat = np.array([est_var[g] for g in labels], dtype=float)
    finite = np.isfinite(var_hat)
    if finite.sum() < 2:
        return size_tau_by_group

    # Between-group signal, net of the noise in the per-group estimates.
    between = float(np.var(tau_hat[finite], ddof=1))
    v0 = max(between - float(np.mean(var_hat[finite])), 0.0)

    out = {g: float(v) for g, v in size_tau_by_group.items()}
    for grp, v_g in zip(labels, var_hat):
        if not np.isfinite(v_g):
            out[grp] = tau_global
            continue
        omega = v0 / (v0 + v_g) if (v0 + v_g) > 0 else 0.0
        out[grp] = max(omega * float(size_tau_by_group[grp]) + (1.0 - omega) * tau_global, 1e-6)
    return pd.Series(out, dtype=float)


def _estimate_group_hypers(
    stats: pd.DataFrame,
    fallback: Optional[tuple[float, float, float, float, float]] = None,
    group_shrink_strength: float = 0.0,
    hyper_shrink: str = "off",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    # Fallback tuple: (alpha, beta, size_mu, size_sigma, size_tau)
    if fallback is None:
        alpha_g, beta_g = _estimate_beta_hyperparams(
            pd.DataFrame(
                {
                    "s": stats["s_obs"].astype(int),
                    "n": stats["n_obs"].astype(int),
                }
            )
        )
        size_mu_g, size_sigma_g, size_tau_g = _estimate_size_hyper_from_stats(stats)
    else:
        alpha_g, beta_g, size_mu_g, size_sigma_g, size_tau_g = fallback

    groups = pd.Index(sorted(stats["group"].astype(str).unique()))
    if GLOBAL_GROUP not in groups:
        groups = groups.append(pd.Index([GLOBAL_GROUP]))

    alpha_by_group: dict[str, float] = {}
    beta_by_group: dict[str, float] = {}
    size_mu_by_group: dict[str, float] = {}
    size_sigma_by_group: dict[str, float] = {}
    size_tau_by_group: dict[str, float] = {}

    group_shrink_strength = float(max(group_shrink_strength, 0.0))

    for grp in groups:
        sub = stats[stats["group"] == grp]
        if sub.empty:
            alpha_by_group[grp] = float(alpha_g)
            beta_by_group[grp] = float(beta_g)
            size_mu_by_group[grp] = float(size_mu_g)
            size_sigma_by_group[grp] = float(size_sigma_g)
            size_tau_by_group[grp] = float(size_tau_g)
            continue

        a_hat, b_hat = _estimate_beta_hyperparams(
            pd.DataFrame({"s": sub["s_obs"].astype(int), "n": sub["n_obs"].astype(int)})
        )
        if not np.isfinite(a_hat) or not np.isfinite(b_hat):
            a_hat, b_hat = float(alpha_g), float(beta_g)

        mu_hat, sigma_hat, tau_hat = _estimate_size_hyper_from_stats(
            sub,
            fallback_mu=float(size_mu_g),
            fallback_sigma=float(size_sigma_g),
            fallback_tau=float(size_tau_g),
        )
        # Optional shrink from group-level estimates back to global estimates.
        if grp == GLOBAL_GROUP or group_shrink_strength <= 0.0:
            omega = 1.0
        else:
            n_series = float(len(sub))
            omega = n_series / (n_series + group_shrink_strength)
            omega = float(np.clip(omega, 0.0, 1.0))

        alpha_final = omega * float(max(a_hat, 1e-6)) + (1.0 - omega) * float(max(alpha_g, 1e-6))
        beta_final = omega * float(max(b_hat, 1e-6)) + (1.0 - omega) * float(max(beta_g, 1e-6))
        size_mu_final = omega * float(mu_hat) + (1.0 - omega) * float(size_mu_g)
        size_sigma_final = omega * float(max(sigma_hat, 1e-6)) + (1.0 - omega) * float(max(size_sigma_g, 1e-6))
        size_tau_final = omega * float(max(tau_hat, 1e-6)) + (1.0 - omega) * float(max(size_tau_g, 1e-6))

        alpha_by_group[grp] = float(max(alpha_final, 1e-6))
        beta_by_group[grp] = float(max(beta_final, 1e-6))
        size_mu_by_group[grp] = float(size_mu_final)
        size_sigma_by_group[grp] = float(max(size_sigma_final, 1e-6))
        size_tau_by_group[grp] = float(max(size_tau_final, 1e-6))

    alpha_s = pd.Series(alpha_by_group, dtype=float)
    beta_s = pd.Series(beta_by_group, dtype=float)
    size_mu_s = pd.Series(size_mu_by_group, dtype=float)
    size_sigma_s = pd.Series(size_sigma_by_group, dtype=float)
    size_tau_s = pd.Series(size_tau_by_group, dtype=float)

    if str(hyper_shrink).lower() == "credibility":
        size_tau_s = _credibility_shrink_tau(stats, size_sigma_s, size_tau_s, size_tau_g)

    return alpha_s, beta_s, size_mu_s, size_sigma_s, size_tau_s


def _compute_posteriors_from_stats(
    stats: pd.DataFrame,
    alpha_by_group: pd.Series,
    beta_by_group: pd.Series,
    size_mu_by_group: pd.Series,
    size_sigma_by_group: pd.Series,
    size_tau_by_group: pd.Series,
    adaptive_prior_strength: bool = False,
    prior_strength_min: float = 1.0,
    prior_strength_max: float = 1.0,
    prior_strength_power: float = 1.0,
    item_variance_mode: str = "group",
    item_variance_shrink_strength: float = 20.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    prior_min = float(max(prior_strength_min, 1e-6))
    prior_max = float(max(prior_strength_max, 1e-6))
    if prior_min > prior_max:
        prior_min, prior_max = prior_max, prior_min
    prior_power = float(max(prior_strength_power, 0.0))
    variance_mode = str(item_variance_mode).lower()
    if variance_mode == "shrink_item":
        # Backward-compatible alias: previous heuristic mode now maps to conjugate variance shrinkage.
        variance_mode = "conjugate"
    if variance_mode not in {"group", "conjugate"}:
        variance_mode = "group"
    variance_prior_df = float(max(item_variance_shrink_strength, 2.1))

    groups = stats["group"].astype(str)

    alpha = groups.map(alpha_by_group).fillna(alpha_by_group.get(GLOBAL_GROUP, alpha_by_group.iloc[0]))
    beta = groups.map(beta_by_group).fillna(beta_by_group.get(GLOBAL_GROUP, beta_by_group.iloc[0]))
    size_mu = groups.map(size_mu_by_group).fillna(size_mu_by_group.get(GLOBAL_GROUP, size_mu_by_group.iloc[0]))
    size_sigma = groups.map(size_sigma_by_group).fillna(size_sigma_by_group.get(GLOBAL_GROUP, size_sigma_by_group.iloc[0]))
    size_tau = groups.map(size_tau_by_group).fillna(size_tau_by_group.get(GLOBAL_GROUP, size_tau_by_group.iloc[0]))

    s_obs = stats["s_obs"].astype(float)
    n_obs = stats["n_obs"].astype(float)
    n_pos = stats["n_pos"].astype(float)
    sum_log = stats["sum_log"].astype(float)
    if "var_log" in stats.columns:
        item_var_log = stats["var_log"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    elif "sum_sq_log" in stats.columns:
        sum_sq_log = stats["sum_sq_log"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean_log = (sum_log / n_pos.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        var_num = sum_sq_log - n_pos * (mean_log.fillna(0.0) ** 2)
        item_var_log = (var_num / (n_pos - 1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    else:
        item_var_log = pd.Series(0.0, index=stats.index, dtype=float)

    n_obs_denom = n_obs.clip(lower=1.0)
    n_pos_denom = n_pos.clip(lower=1.0)
    occ_scale = pd.Series(1.0, index=stats.index, dtype=float)
    size_scale = pd.Series(1.0, index=stats.index, dtype=float)
    if adaptive_prior_strength:
        group_n_obs = stats.groupby("group", sort=False)["n_obs"].median()
        global_ref_n_obs = float(np.nanmedian(n_obs.to_numpy())) if len(n_obs) > 0 else 1.0
        if not np.isfinite(global_ref_n_obs) or global_ref_n_obs <= 0:
            global_ref_n_obs = 1.0
        ref_n_obs = groups.map(group_n_obs).astype(float).replace([np.inf, -np.inf], np.nan).fillna(global_ref_n_obs).clip(lower=1.0)
        occ_scale = ((ref_n_obs / n_obs_denom) ** prior_power).clip(lower=prior_min, upper=prior_max)

        group_n_pos = stats.groupby("group", sort=False)["n_pos"].median()
        positive_n_pos = n_pos[n_pos > 0]
        global_ref_n_pos = float(np.nanmedian(positive_n_pos.to_numpy())) if len(positive_n_pos) > 0 else 1.0
        if not np.isfinite(global_ref_n_pos) or global_ref_n_pos <= 0:
            global_ref_n_pos = 1.0
        ref_n_pos = groups.map(group_n_pos).astype(float).replace([np.inf, -np.inf], np.nan).fillna(global_ref_n_pos)
        ref_n_pos = ref_n_pos.where(ref_n_pos > 0, global_ref_n_pos)
        size_scale = ((ref_n_pos / n_pos_denom) ** prior_power).clip(lower=prior_min, upper=prior_max)

    alpha_eff = (alpha * occ_scale).clip(lower=1e-9)
    beta_eff = (beta * occ_scale).clip(lower=1e-9)

    p_post = (alpha_eff + s_obs) / (alpha_eff + beta_eff + n_obs)
    p_post = p_post.clip(lower=0.0, upper=1.0)

    mean_mle = (sum_log / n_pos.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if variance_mode == "conjugate":
        # Conjugate update for sigma_i^2 under scaled-inv-chi-square prior:
        # E[sigma_i^2 | x] = (nu*s0^2 + (n_i-1)*s_i^2) / (nu + n_i - 3)
        # Treat size_sigma as prior mean and back out s0^2 = size_sigma * (nu-2)/nu.
        nu = variance_prior_df
        prior_numer = (nu - 2.0) * size_sigma
        sse_term = (n_pos - 1.0).clip(lower=0.0) * item_var_log
        denom = nu + n_pos - 3.0
        sigma_post = (prior_numer + sse_term) / denom.replace(0.0, np.nan)
        sigma_sq_process = sigma_post.where((n_pos > 1.0) & (denom > 1e-9), size_sigma)
        sigma_sq_process = sigma_sq_process.fillna(size_sigma).clip(lower=1e-9)
    else:
        sigma_sq_process = size_sigma.clip(lower=1e-9)

    sigma_for_mu = sigma_sq_process
    k_base = (sigma_for_mu / size_tau.clip(lower=1e-9)).clip(lower=1e-9)
    k_eff = (k_base * size_scale).clip(lower=1e-9)

    credibility = n_pos / (n_pos + k_eff)
    credibility = credibility.fillna(0.0).clip(lower=0.0, upper=1.0)

    mean_filled = mean_mle.fillna(size_mu)
    shrunk_mean_log = credibility * mean_filled + (1.0 - credibility) * size_mu

    posterior_var_mu = sigma_for_mu / (n_pos + k_eff)
    posterior_var_mu = posterior_var_mu.where(n_pos > 0, sigma_for_mu / k_eff)
    posterior_var_mu = posterior_var_mu.fillna(sigma_for_mu / k_eff)
    posterior_var_mu = posterior_var_mu.clip(lower=1e-9)

    idx = stats.index
    return (
        p_post.reindex(idx),
        shrunk_mean_log.reindex(idx),
        posterior_var_mu.reindex(idx),
        sigma_sq_process.reindex(idx),
    )


def _fit_bootstrap_group_hypers(
    stats: pd.DataFrame,
    n_draws: int,
    seed: Optional[int],
    base_alpha_by_group: pd.Series,
    base_beta_by_group: pd.Series,
    base_size_mu_by_group: pd.Series,
    base_size_sigma_by_group: pd.Series,
    base_size_tau_by_group: pd.Series,
    group_shrink_strength: float = 0.0,
    hyper_shrink: str = "off",
) -> Optional[Dict[str, pd.DataFrame]]:
    if n_draws <= 0:
        return None

    rng = np.random.default_rng(seed)
    groups = base_alpha_by_group.index
    cols = [f"draw_{i}" for i in range(n_draws)]

    alpha_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)
    beta_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)
    size_mu_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)
    size_sigma_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)
    size_tau_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)

    if stats.empty:
        for col in cols:
            alpha_draws[col] = base_alpha_by_group
            beta_draws[col] = base_beta_by_group
            size_mu_draws[col] = base_size_mu_by_group
            size_sigma_draws[col] = base_size_sigma_by_group
            size_tau_draws[col] = base_size_tau_by_group
        return {
            "alpha": alpha_draws,
            "beta": beta_draws,
            "size_mu": size_mu_draws,
            "size_sigma": size_sigma_draws,
            "size_tau": size_tau_draws,
        }

    for col in cols:
        sample_idx = rng.integers(0, len(stats), size=len(stats))
        sampled = stats.iloc[sample_idx].copy().reset_index(drop=True)
        (
            alpha_b,
            beta_b,
            size_mu_b,
            size_sigma_b,
            size_tau_b,
        ) = _estimate_group_hypers(
            sampled,
            fallback=(
                float(base_alpha_by_group.get(GLOBAL_GROUP, base_alpha_by_group.iloc[0])),
                float(base_beta_by_group.get(GLOBAL_GROUP, base_beta_by_group.iloc[0])),
                float(base_size_mu_by_group.get(GLOBAL_GROUP, base_size_mu_by_group.iloc[0])),
                float(base_size_sigma_by_group.get(GLOBAL_GROUP, base_size_sigma_by_group.iloc[0])),
                float(base_size_tau_by_group.get(GLOBAL_GROUP, base_size_tau_by_group.iloc[0])),
            ),
            group_shrink_strength=group_shrink_strength,
            hyper_shrink=hyper_shrink,
        )
        alpha_draws[col] = alpha_b.reindex(groups).fillna(base_alpha_by_group)
        beta_draws[col] = beta_b.reindex(groups).fillna(base_beta_by_group)
        size_mu_draws[col] = size_mu_b.reindex(groups).fillna(base_size_mu_by_group)
        size_sigma_draws[col] = size_sigma_b.reindex(groups).fillna(base_size_sigma_by_group)
        size_tau_draws[col] = size_tau_b.reindex(groups).fillna(base_size_tau_by_group)

    return {
        "alpha": alpha_draws,
        "beta": beta_draws,
        "size_mu": size_mu_draws,
        "size_sigma": size_sigma_draws,
        "size_tau": size_tau_draws,
    }


def fit_tsb_hb(
    train_df: pd.DataFrame,
    group_labels: Optional[pd.Series | Dict[str, str]] = None,
    bootstrap_draws: int = 0,
    bootstrap_seed: Optional[int] = None,
    group_shrink_strength: float = 0.0,
    dynamic_occurrence: bool = False,
    occurrence_discount: float = 1.0,
    adaptive_prior_strength: bool = False,
    prior_strength_min: float = 1.0,
    prior_strength_max: float = 1.0,
    prior_strength_power: float = 1.0,
    item_variance_mode: str = "group",
    item_variance_shrink_strength: float = 20.0,
    fit_discount: float = 1.0,
    hyper_train_df: Optional[pd.DataFrame] = None,
    hyper_shrink: str = "off",
    fixed_group_hypers: Optional[dict[str, pd.Series]] = None,
) -> TSBHBParams:
    """Fit TSB-HB with optional group-aware priors and hyperparameter bootstrap.

    - `group_labels`: maps unique_id -> group.
      Use this for regime-aware (ADI/CV^2 groups) or hierarchy-aware (M5 group) priors.
    - `bootstrap_draws`: if >0, estimates group-level hyperparameter uncertainty via
      bootstrap resampling across series.
    - `fit_discount`: exponential recency weight applied to initialization-window
      observations (1.0 reproduces the undiscounted released model).
    - `hyper_train_df`: when given, the group-level hyperparameters (and their
      bootstrap) are estimated from this frame instead of `train_df`, while the
      per-item sufficient statistics still come from the full `train_df`. Pass
      the hyper half of :func:`split_for_hyper_estimation` to keep learned
      partitions from collapsing their own variance components.
    """
    fit_discount = _clip_fit_discount(fit_discount)
    stats = _compute_series_stats(train_df, group_labels=group_labels, fit_discount=fit_discount)
    if hyper_train_df is None:
        hyper_stats = stats
    else:
        hyper_stats = _compute_series_stats(
            hyper_train_df, group_labels=group_labels, fit_discount=fit_discount
        )
    (
        alpha_by_group,
        beta_by_group,
        size_mu_by_group,
        size_sigma_by_group,
        size_tau_by_group,
    ) = _estimate_group_hypers(
        hyper_stats,
        group_shrink_strength=group_shrink_strength,
        hyper_shrink=hyper_shrink,
    )

    if fixed_group_hypers is not None:
        # Oracle path: use supplied hyperparameters instead of estimating them.
        # Only meaningful on simulated data where the truth is known; it isolates
        # "the partition carries no information" from "the partition's
        # hyperparameters are hard to estimate".
        alpha_by_group = fixed_group_hypers["alpha"].reindex(alpha_by_group.index).fillna(alpha_by_group)
        beta_by_group = fixed_group_hypers["beta"].reindex(beta_by_group.index).fillna(beta_by_group)
        size_mu_by_group = fixed_group_hypers["size_mu"].reindex(size_mu_by_group.index).fillna(size_mu_by_group)
        size_sigma_by_group = fixed_group_hypers["size_sigma"].reindex(size_sigma_by_group.index).fillna(size_sigma_by_group)
        size_tau_by_group = fixed_group_hypers["size_tau"].reindex(size_tau_by_group.index).fillna(size_tau_by_group)

    p_post, shrunk_mean_log, posterior_var_mu, sigma_sq_process = _compute_posteriors_from_stats(
        stats=stats,
        alpha_by_group=alpha_by_group,
        beta_by_group=beta_by_group,
        size_mu_by_group=size_mu_by_group,
        size_sigma_by_group=size_sigma_by_group,
        size_tau_by_group=size_tau_by_group,
        adaptive_prior_strength=adaptive_prior_strength,
        prior_strength_min=prior_strength_min,
        prior_strength_max=prior_strength_max,
        prior_strength_power=prior_strength_power,
        item_variance_mode=item_variance_mode,
        item_variance_shrink_strength=item_variance_shrink_strength,
    )

    bootstrap_group_hypers = _fit_bootstrap_group_hypers(
        stats=hyper_stats,
        n_draws=int(max(bootstrap_draws, 0)),
        seed=bootstrap_seed,
        base_alpha_by_group=alpha_by_group,
        base_beta_by_group=beta_by_group,
        base_size_mu_by_group=size_mu_by_group,
        base_size_sigma_by_group=size_sigma_by_group,
        base_size_tau_by_group=size_tau_by_group,
        group_shrink_strength=group_shrink_strength,
        hyper_shrink=hyper_shrink,
    )

    return TSBHBParams(
        p_posterior=p_post,
        shrunk_mean_log=shrunk_mean_log,
        posterior_var_mu=posterior_var_mu,
        sigma_sq_process=sigma_sq_process,
        group_labels=stats["group"].astype(str),
        alpha_by_group=alpha_by_group,
        beta_by_group=beta_by_group,
        size_global_mean_by_group=size_mu_by_group,
        size_sigma_sq_by_group=size_sigma_by_group,
        size_tau_sq_by_group=size_tau_by_group,
        n_obs=stats["n_obs"],
        s_obs=stats["s_obs"],
        n_pos=stats["n_pos"],
        sum_log=stats["sum_log"],
        sum_sq_log=stats["sum_sq_log"],
        bootstrap_group_hypers=bootstrap_group_hypers,
        group_shrink_strength=float(max(group_shrink_strength, 0.0)),
        dynamic_occurrence=bool(dynamic_occurrence),
        occurrence_discount=_clip_occurrence_discount(occurrence_discount),
        adaptive_prior_strength=bool(adaptive_prior_strength),
        prior_strength_min=float(prior_strength_min),
        prior_strength_max=float(prior_strength_max),
        prior_strength_power=float(max(prior_strength_power, 0.0)),
        item_variance_mode=str(item_variance_mode).lower(),
        item_variance_shrink_strength=float(max(item_variance_shrink_strength, 1e-6)),
        item_var_log=stats["var_log"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0),
        fit_discount=fit_discount,
    )


def _resolve_uid_series_value(series: pd.Series, uid: str, default: float) -> float:
    val = series.get(uid, default)
    try:
        out = float(val)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    return out


def _resolve_uid_group(params: TSBHBParams, uid: str) -> str:
    grp = params.group_labels.get(uid, GLOBAL_GROUP)
    if pd.isna(grp):
        return GLOBAL_GROUP
    return str(grp)


def _analytic_mixture_quantile(
    p: np.ndarray,
    mu: np.ndarray,
    pred_var: np.ndarray,
    q: float,
) -> np.ndarray:
    """Analytic quantile for zero-inflated LogNormal: Bernoulli(p) × LogNormal(mu, pred_var)."""
    p = np.clip(p, 0.0, 1.0)
    mass_at_zero = 1.0 - p
    result = np.zeros_like(p)

    active = q > mass_at_zero
    if not np.any(active):
        return result

    q_adj = (q - mass_at_zero[active]) / np.maximum(p[active], 1e-12)
    q_adj = np.clip(q_adj, 1e-12, 1.0 - 1e-12)

    sigma = np.sqrt(np.maximum(pred_var[active], 0.0))
    has_var = sigma > 0
    result_active = np.where(
        has_var,
        np.exp(mu[active] + sigma * norm.ppf(q_adj)),
        np.exp(mu[active]),
    )
    result[active] = np.maximum(result_active, 0.0)
    return result


def predict_tsb_hb(
    params: TSBHBParams,
    eval_df: pd.DataFrame,
    quantiles: Optional[List[float]] = None,
    n_samples: int = 2000,
    include_hyper_uncertainty: bool = True,
) -> pd.DataFrame:
    """Predict on the evaluation set.

    - Point forecast: per-series constant mean across horizon.
    - Probabilistic forecast: analytic mixture quantiles (zero-inflated LogNormal).
      When bootstrap hyperparameter draws are available *and* include_hyper_uncertainty
      is True, quantiles are averaged over bootstrap draws for robustness.
    """
    out = eval_df[["unique_id", "ds"]].copy()

    if quantiles is None or len(quantiles) == 0:
        sigma = out["unique_id"].map(params.sigma_sq_process).fillna(params.sigma_sq_process.mean())
        size_mean = np.exp(
            out["unique_id"].map(params.shrunk_mean_log).fillna(params.shrunk_mean_log.mean())
            + sigma / 2.0
        )
        p = out["unique_id"].map(params.p_posterior).fillna(params.p_posterior.mean()).clip(lower=0.0, upper=1.0)
        out["yhat"] = (p * size_mean).fillna(0.0)
        return out

    qcols = [f"q_{q}" for q in quantiles]

    uid_arr = out["unique_id"].to_numpy()
    p_arr = pd.Series(uid_arr).map(params.p_posterior).fillna(params.p_posterior.mean()).to_numpy(dtype=float)
    p_arr = np.clip(p_arr, 0.0, 1.0)
    mu_arr = pd.Series(uid_arr).map(params.shrunk_mean_log).fillna(params.shrunk_mean_log.mean()).to_numpy(dtype=float)
    sigma_arr = pd.Series(uid_arr).map(params.sigma_sq_process).fillna(params.sigma_sq_process.mean()).to_numpy(dtype=float)
    var_mu_arr = pd.Series(uid_arr).map(params.posterior_var_mu).fillna(params.posterior_var_mu.mean()).to_numpy(dtype=float)

    pred_var_arr = np.maximum(sigma_arr + var_mu_arr, 1e-9)

    use_bootstrap = (
        include_hyper_uncertainty
        and params.bootstrap_group_hypers is not None
        and "alpha" in params.bootstrap_group_hypers
        and not params.bootstrap_group_hypers["alpha"].empty
    )

    if use_bootstrap:
        n_obs_arr = pd.Series(uid_arr).map(params.n_obs).fillna(0.0).to_numpy(dtype=float)
        s_obs_arr = pd.Series(uid_arr).map(params.s_obs).fillna(0.0).to_numpy(dtype=float)
        n_pos_arr = pd.Series(uid_arr).map(params.n_pos).fillna(0.0).to_numpy(dtype=float)
        sum_log_arr = pd.Series(uid_arr).map(params.sum_log).fillna(0.0).to_numpy(dtype=float)
        mean_mle_arr = np.where(n_pos_arr > 0, sum_log_arr / n_pos_arr, 0.0)
        group_arr = pd.Series(uid_arr).map(params.group_labels).fillna(GLOBAL_GROUP).to_numpy()
        item_var_arr = np.full(len(uid_arr), np.nan)
        if params.item_var_log is not None:
            item_var_arr = pd.Series(uid_arr).map(params.item_var_log).fillna(np.nan).to_numpy(dtype=float)

        var_mode = str(params.item_variance_mode).lower()
        if var_mode == "shrink_item":
            var_mode = "conjugate"
        if var_mode not in {"group", "conjugate"}:
            var_mode = "group"
        var_prior_df = float(max(params.item_variance_shrink_strength, 2.1))

        alpha_df = params.bootstrap_group_hypers["alpha"]
        beta_df = params.bootstrap_group_hypers["beta"]
        mu_df = params.bootstrap_group_hypers["size_mu"]
        sigma_df = params.bootstrap_group_hypers["size_sigma"]
        tau_df = params.bootstrap_group_hypers["size_tau"]
        n_draws = alpha_df.shape[1]

        q_accum = {f"q_{q}": np.zeros(len(uid_arr), dtype=float) for q in quantiles}
        p_accum = np.zeros(len(uid_arr), dtype=float)

        for d in range(n_draws):
            col = alpha_df.columns[d]
            grp_resolved = np.array([
                g if g in alpha_df.index else (GLOBAL_GROUP if GLOBAL_GROUP in alpha_df.index else alpha_df.index[0])
                for g in group_arr
            ])
            alpha_d = pd.Series(grp_resolved).map(alpha_df[col]).to_numpy(dtype=float)
            beta_d = pd.Series(grp_resolved).map(beta_df[col]).to_numpy(dtype=float)
            mu_global_d = pd.Series(grp_resolved).map(mu_df[col]).to_numpy(dtype=float)
            sigma_d = np.maximum(pd.Series(grp_resolved).map(sigma_df[col]).to_numpy(dtype=float), 1e-9)
            tau_d = np.maximum(pd.Series(grp_resolved).map(tau_df[col]).to_numpy(dtype=float), 1e-9)

            alpha_d = np.maximum(alpha_d, 1e-9)
            beta_d = np.maximum(beta_d, 1e-9)
            p_d = (alpha_d + s_obs_arr) / (alpha_d + beta_d + n_obs_arr)
            p_d = np.clip(p_d, 0.0, 1.0)

            if var_mode == "conjugate":
                conj_ok = np.isfinite(item_var_arr) & (n_pos_arr > 1.0)
                nu = var_prior_df
                denom = nu + n_pos_arr - 3.0
                conj_ok = conj_ok & (denom > 1e-9)
                sigma_d = np.where(
                    conj_ok,
                    np.maximum(((nu - 2.0) * sigma_d + (n_pos_arr - 1.0) * item_var_arr) / denom, 1e-9),
                    sigma_d,
                )

            k_d = np.maximum(sigma_d / tau_d, 1e-9)
            cred_d = np.where(n_pos_arr > 0, n_pos_arr / (n_pos_arr + k_d), 0.0)
            mu_d = np.where(n_pos_arr > 0, cred_d * mean_mle_arr + (1.0 - cred_d) * mu_global_d, mu_global_d)
            var_mu_d = np.where(n_pos_arr > 0, sigma_d / (n_pos_arr + k_d), sigma_d / k_d)
            pred_var_d = np.maximum(sigma_d + var_mu_d, 1e-9)

            for q in quantiles:
                q_accum[f"q_{q}"] += _analytic_mixture_quantile(p_d, mu_d, pred_var_d, q)
            p_accum += (1.0 - p_d)

        for q in quantiles:
            out[f"q_{q}"] = q_accum[f"q_{q}"] / n_draws
        out["prob_zero_predicted"] = p_accum / n_draws

    else:
        for q in quantiles:
            out[f"q_{q}"] = _analytic_mixture_quantile(p_arr, mu_arr, pred_var_arr, q)
        out["prob_zero_predicted"] = 1.0 - p_arr

    return out


def _params_from_online_state(state: TSBHBOnlineState) -> TSBHBParams:
    idx = state.n_obs.index
    if state.dynamic_occurrence:
        n_occ = state.n_eff_occ.reindex(idx).fillna(0.0)
        s_occ = state.s_eff_occ.reindex(idx).fillna(0.0)
    else:
        n_occ = state.n_obs.reindex(idx).fillna(0.0)
        s_occ = state.s_obs.reindex(idx).fillna(0.0)

    n_pos = state.n_pos.reindex(idx).fillna(0.0)
    sum_log = state.sum_log.reindex(idx).fillna(0.0)
    sum_sq_log = state.sum_sq_log.reindex(idx).fillna(0.0)
    mean_log = (sum_log / n_pos.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    var_num = sum_sq_log - n_pos * (mean_log.fillna(0.0) ** 2)
    var_log = (var_num / (n_pos - 1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    stats = pd.DataFrame(
        {
            "n_obs": n_occ,
            "s_obs": s_occ,
            "n_pos": n_pos,
            "sum_log": sum_log,
            "sum_sq_log": sum_sq_log,
            "var_log": var_log,
            "group": state.group_labels.reindex(idx).fillna(GLOBAL_GROUP).astype(str),
        },
        index=idx,
    )
    p_post, shrunk_mean_log, posterior_var_mu, sigma_sq_process = _compute_posteriors_from_stats(
        stats=stats,
        alpha_by_group=state.alpha_by_group,
        beta_by_group=state.beta_by_group,
        size_mu_by_group=state.size_global_mean_by_group,
        size_sigma_by_group=state.size_sigma_sq_by_group,
        size_tau_by_group=state.size_tau_sq_by_group,
        adaptive_prior_strength=state.adaptive_prior_strength,
        prior_strength_min=state.prior_strength_min,
        prior_strength_max=state.prior_strength_max,
        prior_strength_power=state.prior_strength_power,
        item_variance_mode=state.item_variance_mode,
        item_variance_shrink_strength=state.item_variance_shrink_strength,
    )

    return TSBHBParams(
        p_posterior=p_post,
        shrunk_mean_log=shrunk_mean_log,
        posterior_var_mu=posterior_var_mu,
        sigma_sq_process=sigma_sq_process,
        group_labels=stats["group"],
        alpha_by_group=state.alpha_by_group,
        beta_by_group=state.beta_by_group,
        size_global_mean_by_group=state.size_global_mean_by_group,
        size_sigma_sq_by_group=state.size_sigma_sq_by_group,
        size_tau_sq_by_group=state.size_tau_sq_by_group,
        n_obs=n_occ,
        s_obs=s_occ,
        n_pos=n_pos,
        sum_log=sum_log,
        sum_sq_log=sum_sq_log,
        bootstrap_group_hypers=state.bootstrap_group_hypers,
        group_shrink_strength=0.0,
        dynamic_occurrence=state.dynamic_occurrence,
        occurrence_discount=_clip_occurrence_discount(state.occurrence_discount),
        adaptive_prior_strength=state.adaptive_prior_strength,
        prior_strength_min=state.prior_strength_min,
        prior_strength_max=state.prior_strength_max,
        prior_strength_power=state.prior_strength_power,
        item_variance_mode=state.item_variance_mode,
        item_variance_shrink_strength=state.item_variance_shrink_strength,
        item_var_log=var_log,
    )


def initialize_online_tsb_hb(
    train_df: pd.DataFrame,
    group_labels: Optional[pd.Series | Dict[str, str]] = None,
    bootstrap_draws: int = 0,
    bootstrap_seed: Optional[int] = None,
    group_shrink_strength: float = 0.0,
    dynamic_occurrence: bool = False,
    occurrence_discount: float = 1.0,
    adaptive_prior_strength: bool = False,
    prior_strength_min: float = 1.0,
    prior_strength_max: float = 1.0,
    prior_strength_power: float = 1.0,
    item_variance_mode: str = "group",
    item_variance_shrink_strength: float = 20.0,
    fit_discount: float = 1.0,
) -> TSBHBOnlineState:
    occ_discount = _clip_occurrence_discount(occurrence_discount)
    params = fit_tsb_hb(
        train_df=train_df,
        group_labels=group_labels,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        group_shrink_strength=group_shrink_strength,
        dynamic_occurrence=dynamic_occurrence,
        occurrence_discount=occ_discount,
        adaptive_prior_strength=adaptive_prior_strength,
        prior_strength_min=prior_strength_min,
        prior_strength_max=prior_strength_max,
        prior_strength_power=prior_strength_power,
        item_variance_mode=item_variance_mode,
        item_variance_shrink_strength=item_variance_shrink_strength,
        fit_discount=fit_discount,
    )
    return TSBHBOnlineState(
        n_obs=params.n_obs.copy(),
        s_obs=params.s_obs.copy(),
        n_eff_occ=params.n_obs.copy(),
        s_eff_occ=params.s_obs.copy(),
        n_pos=params.n_pos.copy(),
        sum_log=params.sum_log.copy(),
        sum_sq_log=params.sum_sq_log.copy(),
        group_labels=params.group_labels.copy(),
        alpha_by_group=params.alpha_by_group.copy(),
        beta_by_group=params.beta_by_group.copy(),
        size_global_mean_by_group=params.size_global_mean_by_group.copy(),
        size_sigma_sq_by_group=params.size_sigma_sq_by_group.copy(),
        size_tau_sq_by_group=params.size_tau_sq_by_group.copy(),
        bootstrap_group_hypers=params.bootstrap_group_hypers,
        dynamic_occurrence=bool(dynamic_occurrence),
        occurrence_discount=occ_discount,
        adaptive_prior_strength=bool(adaptive_prior_strength),
        prior_strength_min=float(prior_strength_min),
        prior_strength_max=float(prior_strength_max),
        prior_strength_power=float(max(prior_strength_power, 0.0)),
        item_variance_mode=str(item_variance_mode).lower(),
        item_variance_shrink_strength=float(max(item_variance_shrink_strength, 1e-6)),
    )


def predict_online_tsb_hb(
    state: TSBHBOnlineState,
    eval_df: pd.DataFrame,
    quantiles: Optional[List[float]] = None,
    n_samples: int = 2000,
    include_hyper_uncertainty: bool = False,
) -> pd.DataFrame:
    params = _params_from_online_state(state)
    return predict_tsb_hb(
        params=params,
        eval_df=eval_df,
        quantiles=quantiles,
        n_samples=n_samples,
        include_hyper_uncertainty=include_hyper_uncertainty,
    )


def update_online_tsb_hb(state: TSBHBOnlineState, observed_df: pd.DataFrame) -> TSBHBOnlineState:
    if observed_df.empty:
        return state

    obs = observed_df[["unique_id", "y"]].copy()
    obs["occ"] = (obs["y"] > 0).astype(float)
    obs["log_y"] = np.nan
    pos_mask = obs["y"] > 0
    obs.loc[pos_mask, "log_y"] = np.log(obs.loc[pos_mask, "y"].astype(float))

    g = obs.groupby("unique_id", sort=False)
    n_add = g["y"].size().astype(float)
    s_add = g["occ"].sum().astype(float)
    n_pos_add = g["log_y"].count().astype(float)
    sum_log_add = g["log_y"].sum(min_count=1).fillna(0.0).astype(float)
    sum_sq_log_add = g["log_y"].apply(lambda x: float(np.square(x.dropna()).sum())).astype(float)

    state.n_obs = state.n_obs.add(n_add, fill_value=0.0)
    state.s_obs = state.s_obs.add(s_add, fill_value=0.0)
    state.n_pos = state.n_pos.add(n_pos_add, fill_value=0.0)
    state.sum_log = state.sum_log.add(sum_log_add, fill_value=0.0)
    state.sum_sq_log = state.sum_sq_log.add(sum_sq_log_add, fill_value=0.0)

    if state.dynamic_occurrence:
        discount = _clip_occurrence_discount(state.occurrence_discount)
        state.n_eff_occ = state.n_eff_occ.mul(discount, fill_value=0.0).add(n_add, fill_value=0.0)
        state.s_eff_occ = state.s_eff_occ.mul(discount, fill_value=0.0).add(s_add, fill_value=0.0)
    else:
        state.n_eff_occ = state.n_obs.copy()
        state.s_eff_occ = state.s_obs.copy()

    for uid in n_add.index:
        if uid not in state.group_labels.index:
            state.group_labels.loc[uid] = GLOBAL_GROUP

    # Keep deterministic ordering for reproducibility
    state.n_obs = state.n_obs.sort_index()
    state.s_obs = state.s_obs.reindex(state.n_obs.index).fillna(0.0)
    state.n_eff_occ = state.n_eff_occ.reindex(state.n_obs.index).fillna(0.0)
    state.s_eff_occ = state.s_eff_occ.reindex(state.n_obs.index).fillna(0.0)
    state.n_pos = state.n_pos.reindex(state.n_obs.index).fillna(0.0)
    state.sum_log = state.sum_log.reindex(state.n_obs.index).fillna(0.0)
    state.sum_sq_log = state.sum_sq_log.reindex(state.n_obs.index).fillna(0.0)
    state.group_labels = state.group_labels.reindex(state.n_obs.index).fillna(GLOBAL_GROUP).astype(str)
    return state


# ---------------------------------------------------------------------------
# Recency-discount selection (no-leakage internal validation)
# ---------------------------------------------------------------------------

DEFAULT_DISCOUNT_GRID: tuple[float, ...] = (1.0, 0.999, 0.997, 0.995, 0.99, 0.98, 0.95, 0.90)


def _split_init_head_tail(init_df: pd.DataFrame, val_ratio: float = 0.2, min_head: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = init_df.sort_values(["unique_id", "ds"], kind="stable").copy()
    tmp["t"] = tmp.groupby("unique_id").cumcount()
    tmp["L"] = tmp.groupby("unique_id")["t"].transform("max") + 1
    head_cut = np.maximum(np.ceil(tmp["L"] * (1.0 - val_ratio)), float(min_head))
    head_mask = tmp["t"] < head_cut
    cols = ["unique_id", "ds", "y"]
    return tmp.loc[head_mask, cols].copy(), tmp.loc[~head_mask, cols].copy()


def select_fit_discount(
    init_df: pd.DataFrame,
    group_labels: Optional[pd.Series | Dict[str, str]] = None,
    grid: Optional[Sequence[float]] = None,
    quantiles: Sequence[float] = (0.5, 0.75, 0.9),
    val_ratio: float = 0.2,
    item_variance_mode: str = "conjugate",
    item_variance_shrink_strength: float = 20.0,
    hyper_split: str = "off",
    hyper_shrink: str = "off",
) -> tuple[float, pd.DataFrame]:
    """Select the recency discount on an internal chronological split of the
    initialization window (first (1-val_ratio) to fit, last val_ratio to score).

    Scoring is mean pinball loss over ``quantiles``, scaled per series by the
    mean absolute demand in the fitting head so that high-volume SKUs do not
    dominate selection. No out-of-sample targets are consumed.
    Returns (best_discount, diagnostics_frame).
    """
    candidates = [float(w) for w in (grid if grid is not None else DEFAULT_DISCOUNT_GRID)]
    head_df, tail_df = _split_init_head_tail(init_df, val_ratio=val_ratio)
    if head_df.empty or tail_df.empty:
        return 1.0, pd.DataFrame({"discount": candidates, "scaled_pinball": np.nan})
    return _score_discount_grid(
        head_df, tail_df, candidates, group_labels, quantiles,
        item_variance_mode, item_variance_shrink_strength, hyper_split, hyper_shrink,
    )


def _score_discount_grid(
    head_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    candidates: Sequence[float],
    group_labels: Optional[pd.Series | Dict[str, str]],
    quantiles: Sequence[float],
    item_variance_mode: str,
    item_variance_shrink_strength: float,
    hyper_split: str = "off",
    hyper_shrink: str = "off",
) -> tuple[float, pd.DataFrame]:

    scale = head_df.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s))))
    scale = scale.replace(0.0, np.nan)
    global_scale = float(np.nanmedian(scale.to_numpy())) if np.isfinite(np.nanmedian(scale.to_numpy())) else 1.0
    scale = scale.fillna(max(global_scale, 1e-9))

    scale = head_df.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s))))
    scale = scale.replace(0.0, np.nan)
    med = np.nanmedian(scale.to_numpy())
    global_scale = float(med) if np.isfinite(med) else 1.0
    scale = scale.fillna(max(global_scale, 1e-9))

    # Score the same specification that will be fitted: when the released model
    # estimates hyperparameters on a held-out half, so must the selector.
    hyper_head_df = (
        None if str(hyper_split).lower() == "off"
        else split_for_hyper_estimation(head_df, mode=hyper_split)[1]
    )

    rows: list[dict[str, float]] = []
    best_w, best_score = 1.0, np.inf
    for w in candidates:
        params = fit_tsb_hb(
            head_df,
            group_labels=group_labels,
            item_variance_mode=item_variance_mode,
            item_variance_shrink_strength=item_variance_shrink_strength,
            fit_discount=w,
            hyper_train_df=hyper_head_df,
            hyper_shrink=hyper_shrink,
        )
        preds = predict_tsb_hb(params, tail_df, quantiles=list(quantiles), include_hyper_uncertainty=False)
        merged = tail_df[["unique_id", "ds", "y"]].merge(preds, on=["unique_id", "ds"], how="left")
        series_scale = merged["unique_id"].map(scale).fillna(max(global_scale, 1e-9)).to_numpy(dtype=float)
        losses = []
        for q in quantiles:
            err = merged["y"].to_numpy(dtype=float) - merged[f"q_{q}"].to_numpy(dtype=float)
            pinball = np.maximum(q * err, (q - 1.0) * err)
            losses.append(pinball / np.maximum(series_scale, 1e-9))
        score = float(np.nanmean(np.concatenate(losses)))
        rows.append({"discount": w, "scaled_pinball": score})
        if np.isfinite(score) and score < best_score - 1e-12:
            best_score, best_w = score, w

    return best_w, pd.DataFrame(rows)


def select_pooling_and_discount(
    init_df: pd.DataFrame,
    candidate_labels: Dict[str, Optional[pd.Series]],
    grid: Optional[Sequence[float]] = None,
    quantiles: Sequence[float] = (0.5, 0.75, 0.9),
    val_ratio: float = 0.2,
    item_variance_mode: str = "conjugate",
    item_variance_shrink_strength: float = 20.0,
    hyper_split: str = "off",
    hyper_shrink: str = "off",
) -> tuple[str, float, pd.DataFrame]:
    """Jointly select the pooling structure and the recency discount on the
    same internal chronological split used by :func:`select_fit_discount`.

    ``candidate_labels`` maps a structure name (e.g. 'global', 'taxonomy',
    'mixture') to its group-label Series (None = single global pool). Returns
    (best_structure_name, best_discount, diagnostics_frame). No out-of-sample
    targets are consumed.
    """
    candidates = [float(w) for w in (grid if grid is not None else DEFAULT_DISCOUNT_GRID)]
    head_df, tail_df = _split_init_head_tail(init_df, val_ratio=val_ratio)
    if head_df.empty or tail_df.empty:
        first = next(iter(candidate_labels))
        return first, 1.0, pd.DataFrame()

    all_rows: list[pd.DataFrame] = []
    best_name, best_w, best_score = next(iter(candidate_labels)), 1.0, np.inf
    for name, labels in candidate_labels.items():
        w, diag = _score_discount_grid(
            head_df, tail_df, candidates, labels, quantiles,
            item_variance_mode, item_variance_shrink_strength, hyper_split, hyper_shrink,
        )
        diag = diag.assign(structure=name)
        all_rows.append(diag)
        score = float(diag["scaled_pinball"].min())
        if np.isfinite(score) and score < best_score - 1e-12:
            best_score, best_name, best_w = score, name, w

    return best_name, best_w, pd.concat(all_rows, ignore_index=True)
