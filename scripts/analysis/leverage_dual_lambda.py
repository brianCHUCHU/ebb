"""Dual-credibility diagnostic: occurrence vs size lambda for tab:leverage.

The paper overloads the symbol lambda_i: eq. (4) defines the OCCURRENCE
credibility n_i(w)/(n_i(w)+phi_g) (phi_g = alpha+beta), while the resolution
section and tab:leverage define the SIZE credibility n_i/(n_i+kappa_g)
(kappa_g = sigma^2/tau^2, n_i = discounted positive count). This script
computes BOTH under a single global pool at w=1 and at each panel's selected
w*, so the table can report two columns and the ambiguity can be adjudicated
against the published medians.

Usage: py scripts/analysis/leverage_dual_lambda.py [panel ...]
Output: outputs/paper_runs/leverage_dual_lambda.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import (
    load_generic_long, load_m5_long, load_online_retail,
    preprocess_m5, preprocess_online_retail,
    train_eval_split_fixed_origin, train_eval_split_last_h,
)
from models.eb_hurdle import fit_eb_hurdle
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

W_STAR = {"online_retail": 0.98, "m5": 0.90, "auto": 0.995,
          "carparts": 0.90, "raf": 0.997}


def load_init(name: str) -> pd.DataFrame:
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
        init, _ = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
        return init
    if name == "m5":
        set_seed(42)
        sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        df = preprocess_m5(sales, cal, sample_size=5000)
        init, _ = train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
        return init
    h = {"auto": 6, "carparts": 6, "raf": 12}[name]
    df = load_generic_long(ROOT / "data" / f"{name}_long.csv")
    init, _ = train_eval_split_last_h(df, h=h)
    return init


def dual_lambda(init: pd.DataFrame, w: float) -> dict[str, float]:
    params = fit_eb_hurdle(
        init, group_labels=None, group_shrink_strength=0.0,
        item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
        fit_discount=w,
    )
    g = params.group_labels.astype(str)
    phi = (params.alpha_by_group + params.beta_by_group).reindex(g.values)
    phi.index = g.index
    kappa = (params.size_sigma_sq_by_group / params.size_tau_sq_by_group).reindex(g.values)
    kappa.index = g.index

    lam_occ = params.n_obs / (params.n_obs + phi)
    lam_size = params.n_pos / (params.n_pos + kappa)

    return {
        "median_lambda_occ": float(lam_occ.median()),
        "share_occ_lt_0.1": float((lam_occ < 0.1).mean()),
        "median_lambda_size": float(lam_size.median()),
        "share_size_lt_0.1": float((lam_size < 0.1).mean()),
        "median_n_obs": float(params.n_obs.median()),
        "median_n_pos": float(params.n_pos.median()),
        "phi_global": float(phi.iloc[0]),
        "kappa_global": float(kappa.iloc[0]),
    }


def main() -> None:
    panels = sys.argv[1:] or ["online_retail", "auto", "carparts", "raf", "m5"]
    rows = []
    for name in panels:
        init = load_init(name)
        for w in (1.0, W_STAR[name]):
            r = dual_lambda(init, w)
            r.update({"panel": name, "w": w})
            rows.append(r)
            print(f"{name:14s} w={w:<6} "
                  f"occ: med={r['median_lambda_occ']:.3f} <0.1={r['share_occ_lt_0.1']:.1%} | "
                  f"size: med={r['median_lambda_size']:.3f} <0.1={r['share_size_lt_0.1']:.1%}",
                  flush=True)
    out = pd.DataFrame(rows)[[
        "panel", "w", "median_lambda_occ", "share_occ_lt_0.1",
        "median_lambda_size", "share_size_lt_0.1",
        "median_n_obs", "median_n_pos", "phi_global", "kappa_global",
    ]]
    out.to_csv(ROOT / "outputs" / "paper_runs" / "leverage_dual_lambda.csv", index=False)
    print("wrote leverage_dual_lambda.csv")


if __name__ == "__main__":
    main()
