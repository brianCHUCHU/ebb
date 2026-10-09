"""C3: within-model no-pooling arm for the cold-start experiment (v2, corrected).

Same truncation levels and scoring as c1_coldstart.py. Two arms per level, differing
ONLY in whether the shared prior is on:
  EBB-B0      : audited structure and discount, B=0 (no bootstrap ensemble)
  EBB-nopool  : same labels/discount, adaptive_prior_strength=True with
                prior_strength_min = prior_strength_max = 1e-6, so (alpha_g, beta_g) and
                kappa_g are scaled by 1e-6 and lambda_i -> 1 for every item with data
                (the noninformative-prior limit of Proposition prop:tsb-limit). Items
                with no positive observation in the window fall back to the group
                mean, the only pooling left. Variance prior df stays at EBB's 20.

Why B=0: predict_eb_hurdle's bootstrap-ensemble path (bootstrap_draws>0 together with
include_hyper_uncertainty=True) recomputes posteriors without applying the
adaptive prior-strength scaling, so a B=20 "no-pool" run silently keeps the prior
(verified 2026-09-27 on Carparts L=6: identical quantiles). Both arms therefore use
B=0 so the comparison isolates the prior. The B=20 audited EBB numbers from
outputs/2026-09-09/coldstart_wide.csv are carried along for reference.

Outputs (outputs/<date>/): coldstart_nopool_<panel>.csv, coldstart_nopool_wide.csv.
Usage: py scripts/integrity/c3_coldstart_nopool.py <panel> [panel ...] | all
"""
from __future__ import annotations

import importlib.util
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import fit_eb_hurdle, predict_eb_hurdle

spec = importlib.util.spec_from_file_location("c1", ROOT / "scripts" / "integrity" / "c1_coldstart.py")
c1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c1)

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
PANELS = ["carparts", "auto", "raf", "online_retail", "m5"]


def log(m):
    print(f"[c3] {m}", flush=True)


def fit_score(init, ev, ev3, init_full, labels, w, name, prior_off):
    tb = time.perf_counter()
    kw = dict(group_labels=labels, item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
              fit_discount=w, bootstrap_draws=0, bootstrap_seed=42)
    if prior_off:
        kw.update(adaptive_prior_strength=True, prior_strength_min=1e-6, prior_strength_max=1e-6,
                  prior_strength_power=1.0)
    params = fit_eb_hurdle(init, **kw)
    pred = predict_eb_hurdle(params, ev, quantiles=c1.Q5, include_hyper_uncertainty=True)
    bad = int((~np.isfinite(pred[c1.QCOLS].to_numpy(float))).sum())
    if bad:
        log(f"{name}: {bad} non-finite quantiles -> 0")
        pred[c1.QCOLS] = pred[c1.QCOLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    m_, q_, n_ = c1.score(pred, name, ev3, init_full)
    return {"model": name, "spl_mean": round(m_, 4), "spl_q90": round(q_, 4), "n_series": n_,
            "wall_s": round(time.perf_counter() - tb, 1)}


def run(panel):
    t0 = time.perf_counter()
    structure, w, _ = c1.CONFIG[panel]
    init_full, ev = c1.load_panel(panel)
    init_full = init_full.copy(); init_full["unique_id"] = init_full["unique_id"].astype(str)
    ev = ev.copy(); ev["unique_id"] = ev["unique_id"].astype(str); ev["ds"] = pd.to_datetime(ev["ds"])
    ev3 = ev[["unique_id", "ds", "y"]]
    rows = []
    for L in c1.LEVELS[panel]:
        Ltag = "full" if L is None else str(L)
        init = c1.truncate(init_full, L)
        labels = mixture_group_labels(init, k=0, fit_discount=w).labels if structure == "mixture" else None
        if labels is not None:
            labels.index = labels.index.astype(str)
        base = {"panel": panel, "L": Ltag}
        for name, off in (("EBB-B0", False), ("EBB-nopool", True)):
            r = fit_score(init, ev, ev3, init_full, labels, w, name, off)
            rows.append({**base, **r})
            log(f"{panel} L={Ltag} {name} {r['spl_mean']:.4f} (q90 {r['spl_q90']:.4f}) {r['wall_s']}s")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"coldstart_nopool_{panel}.csv", index=False)
    log(f"{panel} done in {time.perf_counter() - t0:.0f}s")
    return df


def assemble():
    wide = pd.read_csv(ROOT / "outputs" / "2026-09-09" / "coldstart_wide.csv")
    wide["L"] = wide["L"].astype(str)
    frames = [pd.read_csv(OUT / f"coldstart_nopool_{p}.csv") for p in PANELS if (OUT / f"coldstart_nopool_{p}.csv").exists()]
    if not frames:
        return
    n = pd.concat(frames, ignore_index=True); n["L"] = n["L"].astype(str)
    pv = n.pivot_table(index=["panel", "L"], columns="model", values=["spl_mean", "spl_q90"])
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    m = wide.merge(pv.reset_index(), on=["panel", "L"], how="left")
    m["pool_gain_mean_pct"] = (m["spl_mean_EBB-nopool"] / m["spl_mean_EBB-B0"] - 1) * 100
    m["pool_gain_q90_pct"] = (m["spl_q90_EBB-nopool"] / m["spl_q90_EBB-B0"] - 1) * 100
    cols = ["panel", "L", "median_len", "median_lambda_occ", "median_lambda_size", "EBB", "spl_mean_EBB-B0",
            "spl_mean_EBB-nopool", "TweedieGP", "pool_gain_mean_pct", "pool_gain_q90_pct", "adv_vs_TweedieGP"]
    m = m[cols].sort_values(["panel", "median_len"])
    m.round(4).to_csv(OUT / "coldstart_nopool_wide.csv", index=False)
    pd.set_option("display.width", 220)
    print(m.round(3).to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    panels = PANELS if args == ["all"] else args
    for p in panels:
        run(p)
    assemble()
