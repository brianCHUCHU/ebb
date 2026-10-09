"""T3: Online Retail walk-forward under the configuration the 32-candidate selector picks,
(none, 0.95), next to the audited (global, 0.95) as a reproduction check.

Mechanics and scoring are f1_wf_ebb_corrected.py unchanged (initialize -> predict block ->
reveal -> update; seven-day blocks; n_samples=2000; include_hyper_uncertainty=False; B=0).
The only difference in the `none` arm is the prior-strength scaling at initialization.

Outputs (outputs/<date>/): wf_or_none.csv, wf_or_none_quantiles_<arm>.csv
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

from experiments.protocols import iter_walk_forward_frames
from experiments.run_prob import _coverage_summary, _scaled_pinball_table
from models.eb_hurdle import initialize_online_eb_hurdle, predict_online_eb_hurdle, update_online_eb_hurdle

spec = importlib.util.spec_from_file_location("f1", ROOT / "scripts" / "integrity" / "f1_wf_ebb_corrected.py")
f1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f1)

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
NONE_KW = dict(adaptive_prior_strength=True, prior_strength_min=1e-6, prior_strength_max=1e-6,
               prior_strength_power=1.0)


def run(arm, extra):
    t0 = time.perf_counter()
    init_set, eval_set = f1.load_panel("online_retail")
    state = initialize_online_eb_hurdle(
        init_set, group_labels=None, bootstrap_draws=0, bootstrap_seed=f1.SEED,
        group_shrink_strength=0.0, dynamic_occurrence=False, occurrence_discount=1.0,
        item_variance_mode="conjugate", item_variance_shrink_strength=20.0, fit_discount=0.95, **extra)
    frames = []
    for frame in iter_walk_forward_frames(init_set, eval_set, step_size=7):
        q = predict_online_eb_hurdle(state, frame.target, quantiles=f1.QUANTILES, n_samples=2000,
                                  include_hyper_uncertainty=False)
        q["model"] = "EB-Hurdle"
        frames.append(q[["model", "unique_id", "ds"] + f1.QCOLS])
        state = update_online_eb_hurdle(state, frame.target)
    allq = pd.concat(frames, ignore_index=True)
    allq["ds"] = pd.to_datetime(allq["ds"]); allq["unique_id"] = allq["unique_id"].astype(str)
    allq.to_csv(OUT / f"wf_or_none_quantiles_{arm}.csv", index=False)
    ev = eval_set[["unique_id", "ds", "y"]].copy()
    ev["ds"] = pd.to_datetime(ev["ds"]); ev["unique_id"] = ev["unique_id"].astype(str)
    merged = ev.merge(allq, on=["unique_id", "ds"], how="inner")
    spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=f1.QUANTILES)
    spl = spl[spl["model"] == "EB-Hurdle"]
    cov = _coverage_summary(merged); cov = cov[cov["model"] == "EB-Hurdle"].iloc[0]
    rec = {"arm": arm, "spl_mean": float(spl["scaled_pinball"].mean()),
           "coverage80": float(cov["Coverage@80"]), "aiw80": float(cov["AIW@80"]),
           "wall_s": round(time.perf_counter() - t0, 1)}
    for _, r in spl.iterrows():
        rec[f"q{float(r['quantile']):g}"] = float(r["scaled_pinball"])
    # per-series mean SPL for a paired comparison between arms
    per = merged.copy()
    scale = init_set.assign(unique_id=init_set["unique_id"].astype(str)).groupby("unique_id")["y"] \
        .apply(lambda s: float(np.mean(np.abs(s))))
    print(f"[t3] {arm}: mean SPL {rec['spl_mean']:.4f}  q90 {rec.get('q0.9', float('nan')):.4f}  "
          f"cov80 {rec['coverage80']:.3f}  {rec['wall_s']}s", flush=True)
    return rec


if __name__ == "__main__":
    rows = [run("global_0.95", {}), run("none_0.95", NONE_KW)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wf_or_none.csv", index=False)
    print(df.round(4).to_string(index=False))
