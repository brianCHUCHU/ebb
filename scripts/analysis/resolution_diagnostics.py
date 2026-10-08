"""Per-component resolution diagnostics (review item A1).

Emits, for every pooling group and every discount, the quantities the
resolution condition is stated in terms of, so that the theorem, the code, the
table and the prose all refer to the same objects:

    m_g       number of items in the group with at least one positive obs
    S2_g      unweighted sample variance of item means, 1/(m_g - 1) normalization
    sbar_g    unweighted mean of the sampling variances sigma_g^2 / n_i
    tau2_mom  the moment estimate max(S2_g - sbar_g, floor) actually used to
              initialize REML in models.eb_hurdle
    tau2_raw  the *untruncated* difference S2_g - sbar_g (may be negative)
    rho_hat   plug-in of rho_g: tau2_raw / S2_g -- signed, in (-inf, 1]
    z_hat     (S2_g - sbar_g) / SE(S2_g), SE(S2_g) ~= sqrt(2/(m_g-1)) * S2_g
              equivalently rho_hat * sqrt((m_g-1)/2); signed
    collapsed indicator that the moment estimate truncates

NOTE ON NAMING. The proposition's rho_g = tau_g^2 / (tau_g^2 + sbar_g) is a
population quantity in (0, 1) and is never negative. z_hat below is its signed
sample analogue and is the only one of the two that can be computed from data.
Earlier drafts of the paper used a single symbol for both; they are distinct.

Usage:
    py scripts/analysis/resolution_diagnostics.py [--dataset online_retail]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_loading import (  # noqa: E402
    load_generic_long,
    load_online_retail,
    preprocess_online_retail,
    train_eval_split_fixed_origin,
    train_eval_split_last_h,
)
from models.mixture_pooling import mixture_group_labels  # noqa: E402
from models.eb_hurdle import _compute_series_stats  # noqa: E402

DEFAULT_DISCOUNTS = (1.0, 0.99, 0.98, 0.95, 0.90)
MOM_FLOOR = 1e-6  # matches models.eb_hurdle


def _load(dataset: str) -> pd.DataFrame:
    if dataset == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
        train, _ = train_eval_split_fixed_origin(df)
        return train
    horizons = {"auto": 6, "carparts": 6, "raf": 12}
    df = load_generic_long(ROOT / "data" / f"{dataset}_long.csv")
    train, _ = train_eval_split_last_h(df, h=horizons[dataset])
    return train


def component_diagnostics(train: pd.DataFrame, labels: pd.Series, discount: float) -> pd.DataFrame:
    stats = _compute_series_stats(train, group_labels=labels, fit_discount=discount)
    rows: list[dict[str, float]] = []
    for grp, sub_all in stats.groupby("group", sort=True):
        sub = sub_all[(sub_all["n_pos"] > 0) & np.isfinite(sub_all["mean_log"])]
        if len(sub) < 2:
            continue
        with_var = sub[sub["n_pos"] > 1]
        denom = float(np.sum(with_var["n_pos"] - 1.0))
        sigma2 = (
            float(np.sum((with_var["n_pos"] - 1.0) * with_var["var_log"]) / denom)
            if denom > 1e-9
            else float(sub["var_log"].mean())
        )
        sigma2 = max(sigma2, 1e-6)
        n_i = np.maximum(sub["n_pos"].to_numpy(dtype=float), 1.0)
        m_g = int(len(sub))
        s2 = float(np.var(sub["mean_log"].to_numpy(dtype=float), ddof=1))
        sbar = float(np.mean(sigma2 / n_i))
        tau2_raw = s2 - sbar
        rho_hat = tau2_raw / s2 if s2 > 0 else np.nan
        rows.append(
            {
                "discount": discount,
                "group": str(grp),
                "m_g": m_g,
                "sigma2_g": sigma2,
                "S2_g": s2,
                "sbar_g": sbar,
                "tau2_raw": tau2_raw,
                "tau2_mom": max(tau2_raw, MOM_FLOOR),
                "rho_hat": rho_hat,
                "z_hat": rho_hat * np.sqrt((m_g - 1) / 2.0),
                "n_pos_median": float(np.median(n_i)),
                "collapsed": bool(tau2_raw <= 0.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="online_retail", choices=["online_retail", "auto", "carparts", "raf"])
    ap.add_argument("--mixture-k", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "paper_runs" / "resolution_diagnostics.csv")
    args = ap.parse_args()

    train = _load(args.dataset)
    labels = mixture_group_labels(train, k=args.mixture_k).labels
    frames = [component_diagnostics(train, labels, w) for w in DEFAULT_DISCOUNTS]
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "dataset", args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"wrote {args.out}  ({len(out)} rows)")
    print(out.pivot(index="group", columns="discount", values="z_hat").round(2).to_string())
    print("\ncollapsed components per discount:")
    print(out.groupby("discount")["collapsed"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
