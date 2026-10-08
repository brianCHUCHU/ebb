"""Learned mixture-of-priors pooling for EB-Hurdle.

Instead of assigning items to pooling groups with the fixed ADI/CV^2 taxonomy
thresholds (Syntetos et al., 2005), this module *learns* the pooling structure
from the initialization window: item-level sufficient statistics are modeled as
a K-component mixture where each component carries its own Beta-Binomial
occurrence prior and Normal random-effects size prior. Component membership is
estimated by EM with closed-form E-steps; K is selected by BIC.

The resulting hard assignment is exposed as ordinary group labels so the rest
of the EB-Hurdle pipeline (group-wise EB fitting, posteriors, bootstrap,
calibration) is reused unchanged. The fixed taxonomy becomes an ablation
baseline rather than the model definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.special import gammaln, logsumexp

from models.eb_hurdle import _compute_series_stats


DEFAULT_K_GRID: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
MIXTURE_LABEL_PREFIX = "mix_"


@dataclass
class MixtureComponent:
    alpha: float
    beta: float
    size_mu: float
    size_tau_sq: float
    size_sigma_sq: float
    weight: float


@dataclass
class MixturePoolingResult:
    k: int
    labels: pd.Series               # unique_id -> "mix_j" (argmax responsibility)
    responsibilities: pd.DataFrame  # unique_id x component
    components: list[MixtureComponent]
    log_likelihood: float
    bic: float
    n_iter: int
    k_search: Optional[pd.DataFrame] = None


def _bb_log_marginal_vec(s: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Vectorized Beta-Binomial log marginal, valid for real-valued pseudo-counts."""
    if alpha <= 0 or beta <= 0:
        return np.full_like(s, -np.inf, dtype=float)
    return (
        gammaln(n + 1.0)
        - gammaln(s + 1.0)
        - gammaln(n - s + 1.0)
        + gammaln(s + alpha)
        + gammaln(n - s + beta)
        - gammaln(n + alpha + beta)
        - (gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))
    )


def _fit_weighted_beta_hypers(
    s: np.ndarray,
    n: np.ndarray,
    resp: np.ndarray,
    init: tuple[float, float] = (1.0, 10.0),
) -> tuple[float, float]:
    """Responsibility-weighted Beta-Binomial empirical-Bayes fit."""
    mask = (n > 0) & (s <= n) & (resp > 1e-12)
    if not np.any(mask):
        return float(init[0]), float(init[1])
    s_m, n_m, r_m = s[mask], n[mask], resp[mask]

    def objective(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        if a <= 0 or b <= 0:
            return np.inf
        ll = _bb_log_marginal_vec(s_m, n_m, a, b)
        val = -float(np.sum(r_m * ll))
        return val if np.isfinite(val) else np.inf

    try:
        res = opt.minimize(
            objective,
            x0=[max(init[0], 1e-3), max(init[1], 1e-3)],
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, None)],
        )
    except Exception:
        return float(init[0]), float(init[1])
    if not res.success or not np.all(np.isfinite(res.x)):
        return float(init[0]), float(init[1])
    return float(res.x[0]), float(res.x[1])


def _fit_weighted_size_hypers(
    mean_log: np.ndarray,
    n_pos: np.ndarray,
    var_log: np.ndarray,
    resp: np.ndarray,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Responsibility-weighted random-effects fit for (mu0, sigma^2, tau^2)."""
    fb_mu, fb_sigma, fb_tau = fallback
    pos_mask = (n_pos > 0) & np.isfinite(mean_log) & (resp > 1e-12)
    if not np.any(pos_mask):
        return fb_mu, max(fb_sigma, 1e-6), max(fb_tau, 1e-6)

    y = mean_log[pos_mask]
    m = n_pos[pos_mask]
    r = resp[pos_mask]

    var_mask = pos_mask & (n_pos > 1)
    numer = float(np.sum(resp[var_mask] * (n_pos[var_mask] - 1.0) * var_log[var_mask]))
    denom = float(np.sum(resp[var_mask] * (n_pos[var_mask] - 1.0)))
    sigma_sq = numer / denom if denom > 1e-12 else fb_sigma
    sigma_sq = float(max(sigma_sq, 1e-6))

    r_sum = float(np.sum(r))
    if r_sum <= 1e-12 or len(y) == 1:
        mu0 = float(np.sum(r * y) / max(r_sum, 1e-12)) if r_sum > 1e-12 else fb_mu
        return (mu0 if np.isfinite(mu0) else fb_mu), sigma_sq, max(fb_tau, 1e-6)

    mean_w = float(np.sum(r * y) / r_sum)
    obs_var = float(np.sum(r * (y - mean_w) ** 2) / r_sum)
    avg_sampling = float(np.sum(r * sigma_sq / np.maximum(m, 1.0)) / r_sum)
    tau_init = max(obs_var - avg_sampling, 1e-6)

    def neg_ll(tau_sq: float) -> float:
        if tau_sq <= 0:
            return np.inf
        v = tau_sq + sigma_sq / np.maximum(m, 1.0)
        w = r / v
        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            return np.inf
        mu0_prof = float(np.sum(w * y) / w_sum)
        return float(np.sum(r * (np.log(v) + (y - mu0_prof) ** 2 / v)))

    try:
        res = opt.minimize(
            lambda x: neg_ll(float(x[0])),
            x0=[tau_init],
            method="L-BFGS-B",
            bounds=[(1e-9, None)],
        )
        tau_sq = float(res.x[0]) if res.success else tau_init
    except Exception:
        tau_sq = tau_init
    tau_sq = float(max(tau_sq, 1e-6))

    v = tau_sq + sigma_sq / np.maximum(m, 1.0)
    w = r / v
    mu0 = float(np.sum(w * y) / max(float(np.sum(w)), 1e-12))
    if not np.isfinite(mu0):
        mu0 = fb_mu
    return mu0, sigma_sq, tau_sq


def _component_log_likelihood(
    comp: MixtureComponent,
    s: np.ndarray,
    n: np.ndarray,
    mean_log: np.ndarray,
    n_pos: np.ndarray,
) -> np.ndarray:
    ll = _bb_log_marginal_vec(s, n, comp.alpha, comp.beta)
    pos = (n_pos > 0) & np.isfinite(mean_log)
    if np.any(pos):
        v = comp.size_tau_sq + comp.size_sigma_sq / np.maximum(n_pos[pos], 1.0)
        v = np.maximum(v, 1e-12)
        ll_size = -0.5 * (np.log(2.0 * np.pi * v) + (mean_log[pos] - comp.size_mu) ** 2 / v)
        ll = ll.copy()
        ll[pos] = ll[pos] + ll_size
    return ll


def _global_fallback(stats: pd.DataFrame) -> tuple[float, float, float]:
    pos = stats[stats["n_pos"] > 0]
    mu = float(pos["mean_log"].mean()) if not pos.empty else 0.0
    with_var = pos[pos["n_pos"] > 1]
    if not with_var.empty:
        sigma = float(
            np.sum((with_var["n_pos"] - 1.0) * with_var["var_log"])
            / max(float(np.sum(with_var["n_pos"] - 1.0)), 1e-12)
        )
    else:
        sigma = 1.0
    tau = float(pos["mean_log"].var(ddof=1)) if len(pos) > 1 else 1.0
    return (mu if np.isfinite(mu) else 0.0, max(sigma, 1e-6), max(tau if np.isfinite(tau) else 1.0, 1e-6))


def _init_responsibilities(stats: pd.DataFrame, k: int) -> np.ndarray:
    """Deterministic initialization: quantile bins of the empirical occurrence rate,
    refined by mean log-size within occurrence bins when K allows."""
    n_items = len(stats)
    occ_rate = (stats["s_obs"] / stats["n_obs"].clip(lower=1e-9)).to_numpy(dtype=float)
    mean_log = stats["mean_log"].to_numpy(dtype=float)
    mean_log_f = np.where(np.isfinite(mean_log), mean_log, np.nanmedian(mean_log[np.isfinite(mean_log)]) if np.any(np.isfinite(mean_log)) else 0.0)

    order = np.lexsort((mean_log_f, occ_rate))
    bins = np.zeros(n_items, dtype=int)
    chunks = np.array_split(np.arange(n_items), k)
    for j, chunk in enumerate(chunks):
        bins[order[chunk]] = j

    resp = np.full((n_items, k), 1e-3, dtype=float)
    resp[np.arange(n_items), bins] = 1.0
    resp /= resp.sum(axis=1, keepdims=True)
    return resp


def fit_mixture_pooling(
    stats: pd.DataFrame,
    k: int,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> MixturePoolingResult:
    """EM fit of the K-component mixture of (Beta-Binomial x Normal-RE) priors."""
    idx = stats.index
    s = stats["s_obs"].to_numpy(dtype=float)
    n = stats["n_obs"].to_numpy(dtype=float)
    n_pos = stats["n_pos"].to_numpy(dtype=float)
    mean_log = stats["mean_log"].to_numpy(dtype=float)
    var_log = stats["var_log"].to_numpy(dtype=float)
    n_items = len(idx)
    fallback = _global_fallback(stats)

    resp = _init_responsibilities(stats, k)
    components: list[MixtureComponent] = [
        MixtureComponent(1.0, 10.0, fallback[0], fallback[2], fallback[1], 1.0 / k) for _ in range(k)
    ]

    prev_ll = -np.inf
    n_iter = 0
    log_lik_matrix = np.zeros((n_items, k), dtype=float)

    for n_iter in range(1, max_iter + 1):
        # ---- M-step (from current responsibilities) ----
        new_components: list[MixtureComponent] = []
        for j in range(k):
            r_j = resp[:, j]
            eff = float(np.sum(r_j))
            if eff < 2.0:
                # Degenerate component: freeze to global fallback priors.
                new_components.append(MixtureComponent(1.0, 10.0, fallback[0], fallback[2], fallback[1], max(eff / n_items, 1e-6)))
                continue
            a_j, b_j = _fit_weighted_beta_hypers(s, n, r_j, init=(components[j].alpha, components[j].beta))
            mu_j, sigma_j, tau_j = _fit_weighted_size_hypers(mean_log, n_pos, var_log, r_j, fallback)
            new_components.append(MixtureComponent(a_j, b_j, mu_j, tau_j, sigma_j, eff / n_items))
        components = new_components
        weight_sum = sum(c.weight for c in components)
        for c in components:
            c.weight = max(c.weight / max(weight_sum, 1e-12), 1e-9)

        # ---- E-step ----
        for j, comp in enumerate(components):
            log_lik_matrix[:, j] = np.log(comp.weight) + _component_log_likelihood(comp, s, n, mean_log, n_pos)
        row_norm = logsumexp(log_lik_matrix, axis=1)
        resp = np.exp(log_lik_matrix - row_norm[:, None])
        total_ll = float(np.sum(row_norm))

        if np.isfinite(prev_ll) and abs(total_ll - prev_ll) < tol * max(abs(prev_ll), 1.0):
            prev_ll = total_ll
            break
        prev_ll = total_ll

    n_params = 5 * k + (k - 1)
    bic = -2.0 * prev_ll + n_params * np.log(max(n_items, 1))

    hard = np.argmax(resp, axis=1)
    labels = pd.Series([f"{MIXTURE_LABEL_PREFIX}{j}" for j in hard], index=idx, dtype=object)
    resp_df = pd.DataFrame(resp, index=idx, columns=[f"{MIXTURE_LABEL_PREFIX}{j}" for j in range(k)])

    return MixturePoolingResult(
        k=k,
        labels=labels,
        responsibilities=resp_df,
        components=components,
        log_likelihood=prev_ll,
        bic=float(bic),
        n_iter=n_iter,
    )


def mixture_group_labels(
    train_df: pd.DataFrame,
    k: int = 0,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    fit_discount: float = 1.0,
    max_iter: int = 60,
) -> MixturePoolingResult:
    """Learn pooling labels from the initialization window.

    - ``k`` > 0 fits exactly K components; ``k`` == 0 selects K on ``k_grid`` by BIC.
    - ``fit_discount`` < 1 learns the partition from the same discounted
      sufficient statistics used by the forecaster, keeping the two consistent.
    """
    stats = _compute_series_stats(train_df, group_labels=None, fit_discount=fit_discount)

    if k and int(k) > 0:
        return fit_mixture_pooling(stats, int(k), max_iter=max_iter)

    search_rows: list[dict[str, float]] = []
    best: Optional[MixturePoolingResult] = None
    for kk in k_grid:
        res = fit_mixture_pooling(stats, int(kk), max_iter=max_iter)
        search_rows.append({"k": int(kk), "log_likelihood": res.log_likelihood, "bic": res.bic, "n_iter": res.n_iter})
        if best is None or res.bic < best.bic - 1e-9:
            best = res
    assert best is not None
    best.k_search = pd.DataFrame(search_rows)
    return best
