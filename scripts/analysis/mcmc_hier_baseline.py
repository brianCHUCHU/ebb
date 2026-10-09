"""W2 follow-up: full-posterior Gibbs sampler for model (1) on an OR subsample.

The reviewer objection is that the paper's hierarchical comparisons are
internal (EB at w=1) and never run a sampling-based hierarchical Bayes method.
This script closes the gap at subsample scale: the SAME hierarchical
Beta-Bernoulli / Normal-Normal model as Eq. (1), single global pool, with full
posterior inference by Gibbs-within-Metropolis instead of empirical Bayes.
Pure numpy, deterministic seed, no external dependencies.

Arms:
  mcmc-stationary  -- w = 1 (the stationary hierarchical-Bayes analogue of
                      Chapados/Seeger-style count pooling, on our likelihood)
  mcmc-discounted  -- w = 0.98 via power-prior weighted sufficient statistics
                      (separates inference method from the forgetting axis)
  ebb            -- joint selection + closed-form EB fit on the same series

All arms evaluated on the identical 500-series seed-42 subsample of the
Online Retail split: mean scaled pinball over q in {0.1,...,0.9}, q90 SPL,
MAE, and wall-clock seconds.

Usage: py scripts/analysis/mcmc_hier_baseline.py
Output: outputs/paper_runs/mcmc_hier_baseline.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import betaln

from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from metrics import rmsse, compute_adi_cv2, classify_adi_cv2
from models.eb_hurdle import fit_eb_hurdle, predict_eb_hurdle, select_pooling_and_discount
from models.mixture_pooling import mixture_group_labels

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
N_SUB = 500
SEED = 42
N_ITER, BURN, THIN = 3000, 1000, 5
W_DISC = 0.98


def item_stats(init_set: pd.DataFrame, w: float):
    """Per-item (possibly discounted) sufficient statistics."""
    n, s, m, lsum, lss = [], [], [], [], []
    uids = []
    for uid, g in init_set.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        y = g["y"].to_numpy(dtype=float)
        T = len(y)
        wts = w ** np.arange(T - 1, -1, -1) if w < 1 else np.ones(T)
        pos = y > 0
        uids.append(uid)
        n.append(wts.sum())
        s.append(wts[pos].sum())
        logs = np.log(y[pos])
        wp = wts[pos]
        m.append(wp.sum())
        lsum.append(float((wp * logs).sum()))
        lss.append(float((wp * logs**2).sum()))
    return (np.array(uids), np.array(n), np.array(s), np.array(m),
            np.array(lsum), np.array(lss))


def gibbs(init_set: pd.DataFrame, w: float, rng: np.random.Generator):
    uids, n, s, m, lsum, lss = item_stats(init_set, w)
    N = len(uids)
    has_pos = m > 1e-9
    lbar = np.where(has_pos, lsum / np.maximum(m, 1e-9), 0.0)

    # init
    alpha, beta = 1.0, 4.0
    p = (s + 0.5) / (n + 1.0)
    mu0 = float(lbar[has_pos].mean()) if has_pos.any() else 0.0
    tau2, sigma2 = 1.0, 1.0
    mu = np.where(has_pos, lbar, mu0)

    draws = {"p": [], "mu": [], "sigma2": []}
    for it in range(N_ITER):
        # occurrence probabilities (conjugate, weighted pseudo-counts)
        p = rng.beta(alpha + s, beta + np.maximum(n - s, 1e-9))
        # (alpha, beta) via random-walk MH on logs; likelihood = sum Beta logpdf
        for _ in range(2):
            la, lb = np.log(alpha), np.log(beta)
            la_p, lb_p = la + 0.1 * rng.standard_normal(), lb + 0.1 * rng.standard_normal()
            a_p, b_p = np.exp(la_p), np.exp(lb_p)
            def loglik(a, b):
                return float(np.sum((a - 1) * np.log(np.clip(p, 1e-12, 1)) +
                                    (b - 1) * np.log(np.clip(1 - p, 1e-12, 1))) - N * betaln(a, b))
            # weak Gamma(2, 2) priors on alpha, beta (log-scale Jacobian included)
            def logpost(a, b, la_, lb_):
                return loglik(a, b) + 2 * la_ - a / 2 + 2 * lb_ - b / 2
            if np.log(rng.uniform()) < logpost(a_p, b_p, la_p, lb_p) - logpost(alpha, beta, la, lb):
                alpha, beta = a_p, b_p
        # item means (conjugate)
        prec = m / sigma2 + 1.0 / tau2
        mean = (lsum / sigma2 + mu0 / tau2) / prec
        mu = mean + rng.standard_normal(N) / np.sqrt(prec)
        # mu0, tau2 (conjugate given mu)
        mu0 = float(rng.normal(mu.mean(), np.sqrt(tau2 / N)))
        dev2 = float(((mu - mu0) ** 2).sum())
        tau2 = 1.0 / rng.gamma(1.0 + N / 2.0, 1.0 / (1.0 + dev2 / 2.0))
        # sigma2 (conjugate; weighted within-item SS)
        ss = float(np.sum(lss - 2 * mu * lsum + m * mu**2))
        m_tot = float(m.sum())
        sigma2 = 1.0 / rng.gamma(1.0 + m_tot / 2.0, 1.0 / (1.0 + max(ss, 1e-9) / 2.0))
        if it >= BURN and (it - BURN) % THIN == 0:
            draws["p"].append(p.copy())
            draws["mu"].append(mu.copy())
            draws["sigma2"].append(sigma2)
    P = np.stack(draws["p"])          # (D, N)
    MU = np.stack(draws["mu"])        # (D, N)
    S2 = np.array(draws["sigma2"])    # (D,)
    return uids, P, MU, S2


def predictive_quantiles(uids, P, MU, S2, rng, k_per_draw: int = 5):
    D, N = P.shape
    qs = {}
    samples = np.zeros((D * k_per_draw, N))
    for d in range(D):
        occ = rng.uniform(size=(k_per_draw, N)) < P[d]
        y = np.exp(MU[d] + np.sqrt(S2[d]) * rng.standard_normal((k_per_draw, N)))
        samples[d * k_per_draw:(d + 1) * k_per_draw] = np.where(occ, y, 0.0)
    for q in QUANTILES:
        qs[q] = np.quantile(samples, q, axis=0)
    point = (P * np.exp(MU + S2[:, None] / 2.0)).mean(axis=0)
    return qs, point


def evaluate(uids, qs, point, init_set, eval_set, label, seconds):
    scale = init_set.groupby("unique_id")["y"].apply(lambda v: float(np.mean(np.abs(v))))
    med = float(np.nanmedian(scale.to_numpy()))
    scale = scale.replace(0.0, np.nan).fillna(max(med, 1e-9))
    m = eval_set[["unique_id", "y"]].copy()
    uid_ix = {u: i for i, u in enumerate(uids)}
    idx = m["unique_id"].map(uid_ix).to_numpy()
    denom = m["unique_id"].map(scale).to_numpy(dtype=float)
    yv = m["y"].to_numpy(dtype=float)
    spl = []
    for q in QUANTILES:
        pred_q = np.maximum(qs[q][idx], 0.0)
        err = yv - pred_q
        spl.append(float(np.mean(np.maximum(q * err, (q - 1) * err) / denom)))
    y_pred = point[idx]
    mae = float(np.mean(np.abs(y_pred - yv)))
    mm = eval_set[["unique_id", "ds", "y"]].copy()
    mm["y_pred"] = y_pred
    rm = rmsse(init_set, mm)
    return {"model": label, "SPL_mean": float(np.mean(spl)), "SPL_q90": spl[-1],
            "MAE": mae, "RMSSE": rm, "seconds": seconds}


def ebb_arm(init_set, eval_set):
    t0 = time.perf_counter()
    feats = compute_adi_cv2(init_set)
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)
    tax = feats.set_index("unique_id")["category"].astype(str)
    cands = {"global": None, "taxonomy": tax,
             "mixture": mixture_group_labels(init_set, k=0).labels}
    name, w, _ = select_pooling_and_discount(
        init_set, cands, item_variance_mode="conjugate", item_variance_shrink_strength=20.0)
    params = fit_eb_hurdle(init_set, group_labels=cands[name], group_shrink_strength=0.0,
                        item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
                        fit_discount=w)
    pred_q = predict_eb_hurdle(params, eval_set, quantiles=QUANTILES)
    pred_pt = predict_eb_hurdle(params, eval_set, quantiles=None)
    pred = pred_q.merge(pred_pt, on=["unique_id", "ds"], how="left")
    secs = time.perf_counter() - t0
    print(f"EBB subsample selection: ({name}, {w})")
    merged = eval_set[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
    scale = init_set.groupby("unique_id")["y"].apply(lambda v: float(np.mean(np.abs(v))))
    med = float(np.nanmedian(scale.to_numpy()))
    scale = scale.replace(0.0, np.nan).fillna(max(med, 1e-9))
    denom = merged["unique_id"].map(scale).to_numpy(dtype=float)
    yv = merged["y"].to_numpy(dtype=float)
    spl = []
    for q in QUANTILES:
        col = f"q_{q}"
        pred_q = np.maximum(merged[col].to_numpy(dtype=float), 0.0)
        err = yv - pred_q
        spl.append(float(np.mean(np.maximum(q * err, (q - 1) * err) / denom)))
    mm = merged.rename(columns={"yhat": "y_pred"})[["unique_id", "ds", "y", "y_pred"]]
    mae = float(np.mean(np.abs(mm.y_pred - mm.y)))
    rm = rmsse(init_set, mm)
    return {"model": f"EBB ({name}, {w})", "SPL_mean": float(np.mean(spl)),
            "SPL_q90": spl[-1], "MAE": mae, "RMSSE": rm, "seconds": secs}


def main() -> None:
    df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
    init_all, eval_all = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    rng = np.random.default_rng(SEED)
    uids_all = np.array(sorted(init_all["unique_id"].unique()))
    sub = set(rng.choice(uids_all, size=min(N_SUB, len(uids_all)), replace=False))
    init_set = init_all[init_all.unique_id.isin(sub)].copy()
    eval_set = eval_all[eval_all.unique_id.isin(sub)].copy()
    print(f"subsample: {init_set.unique_id.nunique()} series")

    rows = [ebb_arm(init_set, eval_set)]
    for label, w in (("MCMC hierarchical (w=1)", 1.0),
                     (f"MCMC hierarchical (w={W_DISC})", W_DISC)):
        t0 = time.perf_counter()
        uids, P, MU, S2 = gibbs(init_set, w, np.random.default_rng(SEED + int(w * 100)))
        qs, point = predictive_quantiles(uids, P, MU, S2, np.random.default_rng(SEED + 7))
        secs = time.perf_counter() - t0
        rows.append(evaluate(uids, qs, point, init_set, eval_set, label, secs))
        print(f"{label}: {secs:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "paper_runs" / "mcmc_hier_baseline.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
