"""C5: which block carries the pooling gain? Occurrence-only vs size-only pooling.

Same truncation levels, labels, discount, scoring and B=0 path as
c3_coldstart_nopool.py (v2). For each panel and level two fits are made, prior on
and prior off (prior strength scaled by 1e-6), and their per-item posteriors are
recombined into four arms:

  both      : occurrence prior on,  size prior on   (= EBB-B0)
  occ_only  : occurrence prior on,  size prior off
  size_only : occurrence prior off, size prior on
  none      : occurrence prior off, size prior off  (= EBB-nopool)

predict_eb_hurdle with B=0 reads p_posterior (occurrence) and shrunk_mean_log /
posterior_var_mu (size) directly from the params object, so the recombination is
exact; sigma_sq_process is identical across the two fits (variance prior df 20).

Reported effects (percent of mean SPL, positive = the prior helps):
  occ_effect  = SPL(size_only) / SPL(both) - 1    (switch the occurrence prior off)
  size_effect = SPL(occ_only)  / SPL(both) - 1    (switch the size prior off)
  total       = SPL(none)      / SPL(both) - 1

Outputs (outputs/<date>/): coldstart_blocks_<panel>.csv, coldstart_blocks_wide.csv
Usage: py scripts/integrity/c5_coldstart_blocks.py <panel> [panel ...] | all
"""
from __future__ import annotations

import dataclasses
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
Q90 = "q_0.9"


def log(m):
    print(f"[c5] {m}", flush=True)


def score(params, ev, ev3, init_full, name):
    pred = predict_eb_hurdle(params, ev, quantiles=c1.Q5, include_hyper_uncertainty=True)
    pred[c1.QCOLS] = pred[c1.QCOLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    m_, q_, _ = c1.score(pred, name, ev3, init_full)
    return m_, q_


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
        kw = dict(group_labels=labels, item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
                  fit_discount=w, bootstrap_draws=0, bootstrap_seed=42)
        on = fit_eb_hurdle(init, **kw)
        off = fit_eb_hurdle(init, adaptive_prior_strength=True, prior_strength_min=1e-6,
                         prior_strength_max=1e-6, prior_strength_power=1.0, **kw)
        assert np.allclose(on.sigma_sq_process.to_numpy(float), off.sigma_sq_process.to_numpy(float)), \
            "sigma_sq_process differs between fits"
        arms = {
            "both": on,
            "occ_only": dataclasses.replace(on, shrunk_mean_log=off.shrunk_mean_log,
                                            posterior_var_mu=off.posterior_var_mu),
            "size_only": dataclasses.replace(on, p_posterior=off.p_posterior),
            "none": off,
        }
        rec = {"panel": panel, "L": Ltag}
        for name, prm in arms.items():
            m_, q_ = score(prm, ev, ev3, init_full, name)
            rec[f"spl_{name}"] = round(m_, 4); rec[f"q90_{name}"] = round(q_, 4)
        for tag, arm in (("occ_effect", "size_only"), ("size_effect", "occ_only"), ("total", "none")):
            rec[f"{tag}_mean_pct"] = round((rec[f"spl_{arm}"] / rec["spl_both"] - 1) * 100, 2)
            rec[f"{tag}_q90_pct"] = round((rec[f"q90_{arm}"] / rec["q90_both"] - 1) * 100, 2)
        rows.append(rec)
        log(f"{panel} L={Ltag} both {rec['spl_both']:.4f} | occ {rec['occ_effect_mean_pct']:+.2f}% "
            f"size {rec['size_effect_mean_pct']:+.2f}% total {rec['total_mean_pct']:+.2f}%")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"coldstart_blocks_{panel}.csv", index=False)
    log(f"{panel} done in {time.perf_counter() - t0:.0f}s")


def assemble():
    wide = pd.read_csv(ROOT / "outputs" / "2026-09-09" / "coldstart_wide.csv")
    wide["L"] = wide["L"].astype(str)
    frames = [pd.read_csv(OUT / f"coldstart_blocks_{p}.csv") for p in PANELS
              if (OUT / f"coldstart_blocks_{p}.csv").exists()]
    if not frames:
        return
    b = pd.concat(frames, ignore_index=True); b["L"] = b["L"].astype(str)
    m = wide[["panel", "L", "median_len", "median_lambda_occ", "median_lambda_size"]].merge(b, on=["panel", "L"])
    m = m.sort_values(["panel", "median_len"])
    m.to_csv(OUT / "coldstart_blocks_wide.csv", index=False)
    pd.set_option("display.width", 220)
    print(m[["panel", "L", "median_lambda_occ", "median_lambda_size", "spl_both", "occ_effect_mean_pct",
             "size_effect_mean_pct", "total_mean_pct", "occ_effect_q90_pct", "size_effect_q90_pct"]]
          .to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    panels = PANELS if args == ["all"] else args
    for p in panels:
        run(p)
    assemble()
