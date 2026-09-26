"""Per-item forgetting rate w_i with empirical-Bayes shrinkage: gate evaluation.

Implements docs/DESIGN_per_item_discount.md on the fixed-origin protocol.

Step A  inner split of the outer fitting head -> per-item loss curves R_i(w)
        (pinball at q in {.5,.75,.9}, scaled by the inner head's mean absolute
        demand, averaged within item; identical formula to _score_discount_grid)
Step B  shrunk objective J_i(w) = n_i^V R_i(w) + kappa * Rbar(w); w_i = argmin
        kappa in KAPPAS; kappa selected on the OUTER validation tail
Step C  final fit on the full initialization window with the per-item Series

For every panel it reports external mean SPL / q90 (B=20, same predictive
settings as the released rows) for the scalar reference w*, every kappa, and
the tail-selected kappa*, next to the TweedieGP reference. Structure is the
panel's audited one; only the forgetting axis changes.

Output: outputs/<date>/peritem_discount_eval.csv, peritem_w_<panel>.csv,
        p1_run_meta.json

Usage: py scripts/integrity/p1_peritem_discount_eval.py [panel ...]
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import (load_generic_long, load_m5_long, load_online_retail,
                          preprocess_m5, preprocess_online_retail,
                          train_eval_split_fixed_origin, train_eval_split_last_h)
from experiments.run_prob import _enforce_monotonic_quantiles, _scaled_pinball_table
from models.mixture_pooling import mixture_group_labels
from models.tsb_hb import (DEFAULT_DISCOUNT_GRID, _split_init_head_tail, fit_tsb_hb,
                           predict_tsb_hb)
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
GRID = [float(w) for w in DEFAULT_DISCOUNT_GRID]
SEL_Q = [0.5, 0.75, 0.9]
EVAL_Q = [0.1, 0.25, 0.5, 0.75, 0.9]
KAPPAS = [np.inf, 300.0, 100.0, 30.0, 10.0, 3.0, 0.0]
CONFIG = {"online_retail": ("global", 0.95), "m5": ("mixture", 0.95),
          "auto": ("global", 0.99), "carparts": ("global", 0.90), "raf": ("mixture", 0.997)}
TG_REF = {"online_retail": 0.8845, "m5": 1.6228, "auto": 0.2985, "carparts": 0.3214, "raf": 0.2763}


def load_panel(name):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data/online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    if name == "m5":
        set_seed(42)
        s, c = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        return train_eval_split_fixed_origin(preprocess_m5(s, c, sample_size=5000),
                                             init_ratio=2 / 3, min_len=1)
    h = {"auto": 6, "carparts": 6, "raf": 12}[name]
    return train_eval_split_last_h(load_generic_long(ROOT / f"data/{name}_long.csv"), h=h)


def per_item_loss_curves(head, tail, labels):
    """R_i(w) for w in GRID (per-item mean scaled pinball over SEL_Q) and n_i^V."""
    scale = head.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s))))
    scale = scale.replace(0.0, np.nan)
    med = np.nanmedian(scale.to_numpy())
    scale = scale.fillna(max(float(med) if np.isfinite(med) else 1.0, 1e-9))
    curves = {}
    for w in GRID:
        params = fit_tsb_hb(head, group_labels=labels, item_variance_mode="conjugate",
                            item_variance_shrink_strength=20.0, fit_discount=w)
        pred = predict_tsb_hb(params, tail, quantiles=SEL_Q, include_hyper_uncertainty=False)
        m = tail[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
        sc = m["unique_id"].map(scale).fillna(float(med)).to_numpy(float)
        losses = []
        for q in SEL_Q:
            err = m["y"].to_numpy(float) - m[f"q_{q}"].to_numpy(float)
            losses.append(np.maximum(q * err, (q - 1.0) * err) / np.maximum(sc, 1e-9))
        m["pin"] = np.nanmean(np.vstack(losses), axis=0)
        curves[w] = m.groupby("unique_id")["pin"].mean()
    R = pd.DataFrame(curves)                      # items x grid
    nV = tail.groupby("unique_id").size().reindex(R.index).fillna(0).astype(float)
    return R, nV


def shrunk_w(R, nV, kappa):
    Rbar = R.mean(axis=0)
    if np.isinf(kappa):
        return pd.Series(float(R.columns[int(np.argmin(Rbar.to_numpy()))]), index=R.index)
    J = R.mul(nV, axis=0) + kappa * Rbar.to_numpy()[None, :]
    idx = np.argmin(J.to_numpy(), axis=1)
    return pd.Series([float(R.columns[i]) for i in idx], index=R.index)


def external_spl(init, ev, labels, fd, draws=20):
    params = fit_tsb_hb(init, group_labels=labels, item_variance_mode="conjugate",
                        item_variance_shrink_strength=20.0, fit_discount=fd,
                        bootstrap_draws=draws, bootstrap_seed=42)
    pred = predict_tsb_hb(params, ev, quantiles=EVAL_Q, include_hyper_uncertainty=draws > 0)
    pred = _enforce_monotonic_quantiles(pred, quantiles=EVAL_Q)
    pred["model"] = "x"
    evv = ev[["unique_id", "ds", "y"]].copy()
    merged = evv.merge(pred, on=["unique_id", "ds"], how="inner")
    spl = _scaled_pinball_table(merged, init_set=init, quantiles=EVAL_Q)
    spl = spl[spl["model"] == "x"]
    return float(spl["scaled_pinball"].mean()), float(spl.loc[np.isclose(spl["quantile"].astype(float), 0.9), "scaled_pinball"].iloc[0])


def main():
    panels = sys.argv[1:] or ["carparts", "auto", "raf", "online_retail", "m5"]
    t_all = time.perf_counter()
    rows = []
    for panel in panels:
        t0 = time.perf_counter()
        init, ev = load_panel(panel)
        init = init.copy(); init["unique_id"] = init["unique_id"].astype(str)
        ev = ev.copy(); ev["unique_id"] = ev["unique_id"].astype(str)
        structure, w_star = CONFIG[panel]
        labels = (mixture_group_labels(init, k=0, fit_discount=w_star).labels
                  if structure == "mixture" else None)
        if labels is not None:
            labels.index = labels.index.astype(str)

        # Step A: inner split inside the outer head
        head_o, tail_o = _split_init_head_tail(init, val_ratio=0.2)
        head_i, tail_i = _split_init_head_tail(head_o, val_ratio=0.2)
        R, nV = per_item_loss_curves(head_i, tail_i, labels)
        t_sel = time.perf_counter() - t0

        # Step B: kappa chosen on the outer tail (fit head_o with w_i, score tail_o)
        scale_o = head_o.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s)))).replace(0.0, np.nan)
        med_o = np.nanmedian(scale_o.to_numpy()); scale_o = scale_o.fillna(max(float(med_o), 1e-9))
        kappa_scores = {}
        w_by_kappa = {}
        for kappa in KAPPAS:
            wi = shrunk_w(R, nV, kappa).reindex(init["unique_id"].unique()).fillna(w_star)
            w_by_kappa[kappa] = wi
            params = fit_tsb_hb(head_o, group_labels=labels, item_variance_mode="conjugate",
                                item_variance_shrink_strength=20.0, fit_discount=wi)
            pred = predict_tsb_hb(params, tail_o, quantiles=SEL_Q, include_hyper_uncertainty=False)
            m = tail_o[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
            sc = m["unique_id"].map(scale_o).fillna(float(med_o)).to_numpy(float)
            losses = []
            for q in SEL_Q:
                err = m["y"].to_numpy(float) - m[f"q_{q}"].to_numpy(float)
                losses.append(np.maximum(q * err, (q - 1.0) * err) / np.maximum(sc, 1e-9))
            kappa_scores[kappa] = float(np.nanmean(np.concatenate(losses)))
        kappa_star = min(kappa_scores, key=kappa_scores.get)

        # Step C: external evaluation
        ref_mean, ref_q90 = external_spl(init, ev, labels, w_star)
        rows.append({"panel": panel, "variant": f"scalar w*={w_star}", "kappa": "",
                     "tail_score": "", "spl_mean": round(ref_mean, 4), "spl_q90": round(ref_q90, 4),
                     "w_median": w_star, "w_share_below_star": "", "tweediegp": TG_REF[panel]})
        for kappa in KAPPAS:
            wi = w_by_kappa[kappa]
            mean_, q90_ = external_spl(init, ev, labels, wi)
            rows.append({"panel": panel, "variant": "per-item", "kappa": ("inf" if np.isinf(kappa) else kappa),
                         "tail_score": round(kappa_scores[kappa], 5),
                         "spl_mean": round(mean_, 4), "spl_q90": round(q90_, 4),
                         "w_median": float(wi.median()),
                         "w_share_below_star": round(float((wi < w_star).mean()), 3),
                         "tweediegp": TG_REF[panel],
                         "selected": kappa == kappa_star})
            print(f"[{panel}] kappa={kappa}: tail {kappa_scores[kappa]:.5f}  external mean "
                  f"{mean_:.4f} q90 {q90_:.4f}  (ref {ref_mean:.4f}, TG {TG_REF[panel]})", flush=True)
        w_by_kappa[kappa_star].rename("w_i").to_csv(OUT / f"peritem_w_{panel}.csv")
        pd.DataFrame(rows).to_csv(OUT / "peritem_discount_eval.csv", index=False)
        print(f"[{panel}] kappa*={kappa_star}; inner-curve time {t_sel:.0f}s; "
              f"panel total {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    meta = {"task": "p1_peritem_discount_eval", "grid": GRID, "kappas": [str(k) for k in KAPPAS],
            "selection_quantiles": SEL_Q, "design": "docs/DESIGN_per_item_discount.md",
            "wall_clock_total_s": round(time.perf_counter() - t_all, 1),
            "platform": platform.platform(), "python": sys.version.split()[0], "seed": 42}
    (OUT / "p1_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote peritem_discount_eval.csv")


if __name__ == "__main__":
    main()
