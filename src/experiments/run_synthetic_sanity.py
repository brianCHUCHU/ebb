"""Sanity checks for the synthetic resolution experiment.

Written in response to a reviewer objection: the first version of the panel
experiment concluded "information shortage" from aggregate MAE alone, and that
conclusion did not survive scrutiny. Four checks, each targeting a specific way
the earlier reading could have been an artifact.

1. FULL ORACLE LADDER. The earlier "oracle" arm received the true labels but
   still re-estimated the group hyperparameters, so it could not separate "the
   partition carries no information" from "the partition's hyperparameters are
   hard to estimate". The ladder here is
       global (estimated)
       oracle labels + estimated group hyperparameters
       oracle labels + TRUE group hyperparameters
       true conditional predictive distribution (the irreducible floor)

2. FIXED MARGINAL DIFFICULTY. Previously group means were 0, s, 2s, 3s, so
   raising the separation raised the grand mean and the panel's demand scale
   together -- mean MAE ran from 0.7 to 67 across the grid and cells were not
   comparable. Group centers are now symmetric about a fixed grand mean, group
   occurrence levels symmetric in logit about a fixed rate, and within-group
   variance and test noise held fixed, so only the distance between centers
   changes.

3. MULTIPLE RISK MEASURES. MAE rewards zero-leaning forecasts on intermittent
   data and can move against a proper scoring rule. Every arm is reduced to a
   per-observation predictive triple (p, mu, sigma^2) for the hurdle-lognormal
   law, and MAE, mean pinball, occurrence Brier score, log-size MSE and
   predictive NLL are all computed from that one representation. Group
   hyperparameter recovery error is reported alongside.

4. EXTREME IDENTIFIABLE CASE. The first attempt at this check used T=400 and
   found no advantage for the oracle partition. That was the check failing, not
   the model: with 154 positive observations per item the credibility weight is
   ~1 for everyone and no prior, correct or not, can matter. The prior's
   leverage is kappa/(n_i+kappa), so the informative sweep is over series
   length, and the long-T row is kept as the negative control.

DRIFT AND THE ORACLE. The generator drifts the occurrence rate over time. An
earlier version of ``true_predictive`` used each item's *initial* rate, which
made the "true" predictive distribution systematically wrong on exactly the
block that drifts -- its signature was the true predictive scoring worst on
MAE/pinball/Brier while scoring best on log-size MSE, which does not depend on
occurrence. The generator now returns the realized ``p_t`` path and the oracle
reads it, so the true predictive is the true predictive.

Usage:
    py -m experiments.run_synthetic_sanity --case extreme
    py -m experiments.run_synthetic_sanity --case grid --reps 3
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from models.tsb_hb import fit_tsb_hb, predict_tsb_hb

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "outputs" / "synthetic_sanity"
QUANTILES = (0.5, 0.75, 0.9)


# ------------------------------------------------------------------ generator


def simulate(
    rng: np.random.Generator,
    n_groups: int,
    items_per_group: int,
    T: int,
    horizon: int,
    separation: float,
    tau2: float,
    sigma2: float,
    occurrence_mean: float,
    occurrence_spread: float,
    grand_mu: float = 1.0,
    drift: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Hierarchical hurdle panel with fixed marginal difficulty.

    Group size centers are symmetric about ``grand_mu`` and group occurrence
    levels symmetric in logit about ``occurrence_mean``, so increasing the
    separation moves centers apart without moving the panel mean.

    Returns ``(panel, labels, truth_path, group_hypers)`` where ``truth_path``
    carries the realized ``p_t`` and ``mu_i`` for every (item, period) -- the
    oracle needs the path, not the initial value, because the rate drifts.
    """
    k_idx = np.arange(n_groups) - (n_groups - 1) / 2.0
    group_mu = grand_mu + k_idx * separation
    base_logit = np.log(occurrence_mean / (1.0 - occurrence_mean))
    group_p = 1.0 / (1.0 + np.exp(-(base_logit + k_idx * occurrence_spread)))

    rows, labels, path = [], {}, []
    n_periods = T + horizon
    for k in range(n_groups):
        for j in range(items_per_group):
            uid = f"g{k}_i{j}"
            labels[uid] = f"true_{k}"
            mu_i = float(rng.normal(group_mu[k], np.sqrt(tau2)))
            p_i = float(1.0 / (1.0 + np.exp(-rng.normal(
                np.log(group_p[k] / (1 - group_p[k])), 0.25))))
            for t in range(n_periods):
                p_t = p_i
                if drift:
                    p_t = float(np.clip(p_i * (1.0 - drift * t / (n_periods - 1)), 0.002, 0.999))
                y = float(np.exp(rng.normal(mu_i, np.sqrt(sigma2)))) if rng.random() < p_t else 0.0
                rows.append((uid, t, y))
                path.append((uid, t, p_t, mu_i))

    panel = pd.DataFrame(rows, columns=["unique_id", "ds", "y"])
    truth_path = pd.DataFrame(path, columns=["unique_id", "ds", "p_t", "mu_i"])

    hyp = []
    for k in range(n_groups):
        pk = float(group_p[k])
        var_p = (pk * (1 - pk) * 0.25) ** 2  # delta-method variance of the logit jitter
        conc = max(pk * (1 - pk) / max(var_p, 1e-9) - 1.0, 1e-3)
        hyp.append(
            {
                "group": f"true_{k}",
                "alpha": pk * conc,
                "beta": (1 - pk) * conc,
                "size_mu": float(group_mu[k]),
                "size_sigma": float(sigma2),
                "size_tau": float(tau2),
            }
        )
    return panel, pd.Series(labels, name="group"), truth_path, pd.DataFrame(hyp).set_index("group")


def split(panel: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(panel["ds"].max()) - horizon
    return panel[panel["ds"] <= cut].copy(), panel[panel["ds"] > cut].copy()


# ------------------------------------------------------------------ scoring


def score_predictive(test: pd.DataFrame, p: np.ndarray, mu: np.ndarray, s2: np.ndarray) -> dict[str, float]:
    """All risk measures from one hurdle-lognormal predictive triple.

    ``p`` is P(Y>0); positives are lognormal(mu, s2). Computing every metric
    from the same representation means MAE and the proper scoring rules cannot
    disagree because of an inconsistency in how each arm was rendered.
    """
    y = test["y"].to_numpy(dtype=float)
    s = np.sqrt(np.maximum(s2, 1e-12))
    p = np.clip(p, 1e-9, 1 - 1e-9)

    mean_size = np.exp(mu + s2 / 2.0)
    yhat = p * mean_size
    out = {"mae": float(np.mean(np.abs(y - yhat)))}

    # quantiles of the mixture: P(Y<=v) = (1-p) + p*Phi((log v - mu)/s), v>0
    losses = []
    for q in QUANTILES:
        adj = (q - (1.0 - p)) / p
        v = np.where(adj <= 0.0, 0.0, np.exp(mu + s * norm.ppf(np.clip(adj, 1e-9, 1 - 1e-9))))
        err = y - v
        losses.append(np.maximum(q * err, (q - 1.0) * err))
    out["pinball"] = float(np.mean(np.concatenate(losses)))

    occ = (y > 0).astype(float)
    out["brier_occurrence"] = float(np.mean((p - occ) ** 2))

    pos = y > 0
    if pos.any():
        out["log_size_mse"] = float(np.mean((np.log(y[pos]) - mu[pos]) ** 2))
        # NLL of the hurdle-lognormal: -log(1-p) on zeros, -log p - log f(y) on positives
        ll = np.where(pos, 0.0, np.log(1.0 - p))
        lp = (
            np.log(p[pos])
            - np.log(np.maximum(y[pos], 1e-12))
            - np.log(s[pos] * np.sqrt(2 * np.pi))
            - 0.5 * ((np.log(y[pos]) - mu[pos]) / s[pos]) ** 2
        )
        total = float(np.sum(ll[~pos]) + np.sum(lp))
        out["nll"] = -total / len(y)
    return out


def params_to_triple(params, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a fitted TSBHBParams as the per-observation predictive triple."""
    uid = test["unique_id"].to_numpy()
    p = params.p_posterior.reindex(uid).to_numpy(dtype=float)
    mu = params.shrunk_mean_log.reindex(uid).to_numpy(dtype=float)
    # predictive variance of a positive = process variance + posterior variance of the mean
    s2 = (
        params.sigma_sq_process.reindex(uid).to_numpy(dtype=float)
        + params.posterior_var_mu.reindex(uid).to_numpy(dtype=float)
    )
    return np.nan_to_num(p, nan=0.5), np.nan_to_num(mu, nan=0.0), np.nan_to_num(s2, nan=1.0)


# ------------------------------------------------------------------ arms


def run_arms(train, test, true_labels, truth_path, hyp_df, sigma2, w) -> list[dict]:
    fixed = {c: hyp_df[c] for c in ["alpha", "beta", "size_mu", "size_sigma", "size_tau"]}
    arms = [
        ("global-estimated", None, None),
        ("oracle-labels+estimated-hypers", true_labels, None),
        ("oracle-labels+true-hypers", true_labels, fixed),
    ]
    out = []
    for name, labels, fixed_h in arms:
        params = fit_tsb_hb(
            train, group_labels=labels, item_variance_mode="conjugate",
            item_variance_shrink_strength=20.0, fit_discount=w,
            fixed_group_hypers=fixed_h,
        )
        p, mu, s2 = params_to_triple(params, test)
        rec = {"arm": name}
        rec.update(score_predictive(test, p, mu, s2))
        # Size-block credibility under this arm's fitted pool, mirroring
        # scripts/analysis/leverage_dual_lambda.py: lambda_i = n_i^+/(n_i^+ + kappa_g)
        # with discounted effective positive counts. Post-fit computation only;
        # consumes no randomness, so the seeded panels are unchanged.
        g = params.group_labels.astype(str)
        kappa = (params.size_sigma_sq_by_group / params.size_tau_sq_by_group).reindex(g.values)
        kappa.index = g.index
        lam_size = params.n_pos / (params.n_pos + kappa)
        rec["median_lambda_size"] = float(lam_size.median())
        rec["median_n_pos_eff"] = float(params.n_pos.median())
        if labels is not None:
            est_mu = params.size_global_mean_by_group.reindex(hyp_df.index)
            est_tau = params.size_tau_sq_by_group.reindex(hyp_df.index)
            rec["hyper_mu_rmse"] = float(np.sqrt(np.nanmean((est_mu - hyp_df["size_mu"]) ** 2)))
            rec["hyper_tau_rmse"] = float(np.sqrt(np.nanmean((est_tau - hyp_df["size_tau"]) ** 2)))
        out.append(rec)

    # True conditional predictive distribution, reading the realized p_t path.
    tp = test[["unique_id", "ds"]].merge(truth_path, on=["unique_id", "ds"], how="left")
    rec = {"arm": "true-predictive"}
    rec.update(score_predictive(
        test,
        tp["p_t"].to_numpy(dtype=float),
        tp["mu_i"].to_numpy(dtype=float),
        np.full(len(tp), float(sigma2)),
    ))
    out.append(rec)
    return out


# ------------------------------------------------------------------ cases


def case_extreme(rng, out_dir: Path, series_lengths=(400, 60, 25, 12)) -> pd.DataFrame:
    """Check 4: sweep the item's own sample size, holding the structure fixed."""
    frames = []
    for T in series_lengths:
        h = max(T // 5, 6)
        panel, labels, path, hyp = simulate(
            rng, n_groups=2, items_per_group=500, T=T, horizon=h,
            separation=3.0, tau2=0.05, sigma2=0.25,
            occurrence_mean=0.30, occurrence_spread=5.5, grand_mu=1.5, drift=0.0,
        )
        train, test = split(panel, h)
        npos = train.assign(o=(train.y > 0)).groupby("unique_id")["o"].sum()
        rows = run_arms(train, test, labels, path, hyp, sigma2=0.25, w=1.0)
        for r in rows:
            r["T"] = T
            r["median_n_pos"] = float(npos.median())
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "case", "extreme")
    df.to_csv(out_dir / "extreme.csv", index=False)
    return df


def case_grid(rng, out_dir: Path, n_reps: int = 3) -> pd.DataFrame:
    """Checks 1-3 on a difficulty-controlled grid, with drift active."""
    rows = []
    for sep, w, T, rep in itertools.product((0.5, 1.0, 2.0), (1.0, 0.95, 0.90), (120, 30), range(n_reps)):
        h = max(T // 8, 6)
        panel, labels, path, hyp = simulate(
            rng, n_groups=4, items_per_group=60, T=T, horizon=h,
            separation=sep, tau2=0.15, sigma2=1.0,
            occurrence_mean=0.25, occurrence_spread=0.8, grand_mu=1.0, drift=0.5,
        )
        train, test = split(panel, h)
        npos = train.assign(o=(train.y > 0)).groupby("unique_id")["o"].sum()
        for rec in run_arms(train, test, labels, path, hyp, sigma2=1.0, w=w):
            rec.update({"separation": sep, "discount": w, "T": T, "rep": rep,
                        "median_n_pos": float(npos.median())})
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.insert(0, "case", "grid")
    df.to_csv(out_dir / "grid.csv", index=False)
    return df


def _report(df: pd.DataFrame, index, metrics) -> None:
    for metric in metrics:
        if metric not in df:
            continue
        piv = df.pivot_table(index=index, columns="arm", values=metric)
        print(f"\n-- {metric} --")
        print(piv.round(5).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["extreme", "grid", "both"], default="both")
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    metrics = ("mae", "pinball", "brier_occurrence", "log_size_mse", "nll")
    if args.case in ("extreme", "both"):
        df = case_extreme(rng, args.out)
        print("\n=== CHECK 4: sweep over the item's own sample size ===")
        _report(df, ["T", "median_n_pos"], metrics)

    if args.case in ("grid", "both"):
        df = case_grid(rng, args.out, args.reps)
        print("\n=== CHECKS 1-3: difficulty-controlled grid, drift active ===")
        _report(df, ["separation", "T", "discount"], metrics)
        print("\n-- group hyperparameter recovery --")
        print(df.dropna(subset=["hyper_mu_rmse"]).pivot_table(
            index=["separation", "T", "discount"], columns="arm",
            values=["hyper_mu_rmse", "hyper_tau_rmse"]).round(5).to_string())


if __name__ == "__main__":
    main()
