"""Forensics: is RAF's phi(w) elasticity data-driven or an optimizer artifact?

Four probes on the RAF occurrence Beta-Binomial fit:
  A. phi over a fine w-grid  -> smooth monotone curve = data-driven;
     erratic jumps = optimizer.
  B. Multi-start L-BFGS-B at w in {1.0, 0.997} (6 starting points)
     -> same optimum & objective = stable; scattered = local optima.
  C. Optimizer-free method-of-moments phi at each w -> if MoM phi drops
     ~25% too, the cross-sectional dispersion of p-bar really widened.
  D. Flatness: objective at w=0.997 evaluated at the w=1.0 optimum vs its
     own optimum (delta log-lik) -> tiny delta = weakly identified ridge.

Usage: py scripts/analysis/raf_phi_forensics.py
Output: outputs/paper_runs/raf_phi_forensics.csv (+ stdout report)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import scipy.optimize as opt

from data_loading import load_generic_long, train_eval_split_last_h
from models.eb_hurdle import _beta_binom_log_marginal, _compute_series_stats

W_GRID = [1.0, 0.9995, 0.999, 0.9985, 0.998, 0.9975, 0.997, 0.996,
          0.995, 0.99, 0.98, 0.95, 0.90]
STARTS = [(1.0, 10.0), (0.5, 5.0), (2.0, 50.0), (5.0, 100.0),
          (0.1, 1.0), (10.0, 300.0)]


def counts_at(train: pd.DataFrame, w: float) -> tuple[np.ndarray, np.ndarray]:
    c = _compute_series_stats(train, group_labels=None, fit_discount=w)
    s = c["s_obs"].astype(float).to_numpy()
    n = c["n_obs"].astype(float).to_numpy()
    ok = (n > 0) & (s <= n)
    return s[ok], n[ok]


def neg_ll(params, s_arr, n_arr) -> float:
    a, b = float(params[0]), float(params[1])
    if a <= 0 or b <= 0:
        return np.inf
    return -float(sum(_beta_binom_log_marginal(float(si), float(ni), a, b)
                      for si, ni in zip(s_arr, n_arr)))


def fit(s_arr, n_arr, x0=(1.0, 10.0)):
    r = opt.minimize(lambda p: neg_ll(p, s_arr, n_arr), x0=list(x0),
                     method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
    return r


def mom_phi(s_arr, n_arr) -> float:
    pbar = s_arr / n_arr
    m = float(pbar.mean())
    v_obs = float(pbar.var(ddof=1))
    samp = float(np.mean(m * (1 - m) / n_arr))
    denom = m * (1 - m) * (1 - float(np.mean(1.0 / n_arr)))
    rho = (v_obs - samp) / denom if denom > 0 else np.nan
    if not np.isfinite(rho) or rho <= 0:
        return np.inf
    return 1.0 / rho - 1.0


def main() -> None:
    df = load_generic_long(ROOT / "data" / "raf_long.csv")
    train, _ = train_eval_split_last_h(df, h=12)
    rows = []
    print("A/C. phi(w) fine grid — ML vs method-of-moments:")
    for w in W_GRID:
        s, n = counts_at(train, w)
        r = fit(s, n)
        a, b = float(r.x[0]), float(r.x[1])
        phi = a + b
        lam_med = float(np.median(n / (n + phi)))
        pm = mom_phi(s, n)
        pbar = s / n
        rows.append({"w": w, "alpha": a, "beta": b, "phi_ml": phi,
                     "phi_mom": pm, "median_lambda_occ": lam_med,
                     "median_n": float(np.median(n)),
                     "var_pbar": float(pbar.var(ddof=1)),
                     "converged": bool(r.success), "nll": float(r.fun)})
        print(f"  w={w:<7} phi_ML={phi:9.1f}  phi_MoM={pm:9.1f}  "
              f"Var(pbar)={pbar.var(ddof=1):.6f}  med_lam={lam_med:.3f}  "
              f"conv={r.success}", flush=True)

    print("\nB. Multi-start stability:")
    for w in (1.0, 0.997):
        s, n = counts_at(train, w)
        sols = []
        for x0 in STARTS:
            r = fit(s, n, x0)
            sols.append((x0, float(r.x[0] + r.x[1]), float(r.fun), r.success))
        phis = [p for _, p, _, ok in sols if ok]
        print(f"  w={w}: phi across 6 starts = "
              f"[{min(phis):.1f}, {max(phis):.1f}] "
              f"(spread {100*(max(phis)-min(phis))/min(phis):.2f}%)")
        for x0, p, f, ok in sols:
            print(f"    x0={x0}: phi={p:9.1f} nll={f:.3f} ok={ok}")

    print("\nD. Flatness of the w=0.997 objective:")
    s1, n1 = counts_at(train, 1.0)
    s2, n2 = counts_at(train, 0.997)
    opt1, opt2 = fit(s1, n1), fit(s2, n2)
    nll_cross = neg_ll(opt1.x, s2, n2)
    print(f"  nll@own optimum      = {opt2.fun:.3f} (phi={sum(opt2.x):.1f})")
    print(f"  nll@w=1's optimum    = {nll_cross:.3f} (phi={sum(opt1.x):.1f})")
    print(f"  delta log-lik        = {nll_cross - opt2.fun:.3f} over "
          f"{len(s2)} series")

    pd.DataFrame(rows).to_csv(
        ROOT / "outputs" / "paper_runs" / "raf_phi_forensics.csv", index=False)
    print("\nwrote raf_phi_forensics.csv")


if __name__ == "__main__":
    main()
