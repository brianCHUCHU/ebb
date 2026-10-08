"""Pre-fit pooling-room features from a GLOBAL-pool EB fit (docs/DESIGN_room_diagnostic.md).

All quantities come from the fitting window and the single-pool fit; no partition is
fitted. Returns a dict with lev/het/z for the size and occurrence blocks, the closed-form
room score R1, and bookkeeping (m, median n+, median lambda).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.eb_hurdle import _compute_series_stats, fit_eb_hurdle


def room_features(train: pd.DataFrame, w: float, occurrence_fit_discount=None) -> dict:
    params = fit_eb_hurdle(train, group_labels=None, group_shrink_strength=0.0,
                        item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
                        fit_discount=w, occurrence_fit_discount=occurrence_fit_discount)
    stats = _compute_series_stats(train, fit_discount=w, occurrence_fit_discount=occurrence_fit_discount)
    g = params.group_labels.astype(str)
    sigma2 = float(params.size_sigma_sq_by_group.iloc[0])
    tau2 = float(params.size_tau_sq_by_group.iloc[0])
    phi = float(params.alpha_by_group.iloc[0] + params.beta_by_group.iloc[0])
    # ---- size block ----
    n_pos = params.n_pos.astype(float)
    kappa = sigma2 / max(tau2, 1e-12)
    lam_size = n_pos / (n_pos + kappa)
    m = stats["mean_log"]
    ok = np.isfinite(m) & (n_pos.reindex(m.index) > 0)
    m = m[ok]; npos_ok = n_pos.reindex(m.index)
    m_items = int(len(m))
    S2 = float(m.var(ddof=1)) if m_items > 1 else 0.0
    sbar = float(np.mean(sigma2 / npos_ok)) if m_items > 0 else np.nan
    het_size = max(0.0, S2 - sbar)
    z_size = (S2 - sbar) / (np.sqrt(2.0 / max(m_items - 1, 1)) * S2) if S2 > 0 else 0.0
    # ---- occurrence block ----
    n_obs = params.n_obs.astype(float); s_obs = params.s_obs.astype(float)
    lam_occ = n_obs / (n_obs + phi)
    p_hat = (s_obs / n_obs.replace(0, np.nan)).dropna()
    n_o = n_obs.reindex(p_hat.index)
    Sp2 = float(p_hat.var(ddof=1)) if len(p_hat) > 1 else 0.0
    sbar_p = float(np.mean(p_hat * (1 - p_hat) / n_o)) if len(p_hat) > 0 else np.nan
    het_occ = max(0.0, Sp2 - sbar_p)
    z_occ = (Sp2 - sbar_p) / (np.sqrt(2.0 / max(len(p_hat) - 1, 1)) * Sp2) if Sp2 > 0 else 0.0
    lev_size = float(np.median(1.0 - lam_size)); lev_occ = float(np.median(1.0 - lam_occ))
    return {"m_items": m_items, "median_n_pos": float(np.median(n_pos)), "median_n_obs": float(np.median(n_obs)),
            "lambda_size": float(np.median(lam_size)), "lambda_occ": float(np.median(lam_occ)),
            "lev_size": lev_size, "het_size": het_size, "z_size": float(z_size), "sigma2_global": sigma2, "tau2_global": tau2,
            "lev_occ": lev_occ, "het_occ": het_occ, "z_occ": float(z_occ), "phi_global": phi,
            "R1": lev_size ** 2 * het_size + lev_occ ** 2 * het_occ,
            # amendment 2026-09-15: weighted (model) tau^2 as the de-noised heterogeneity
            "R1b": lev_size ** 2 * tau2}
