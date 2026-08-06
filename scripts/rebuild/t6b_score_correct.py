"""Correct-protocol (init 2/3) M5 seed-variance table + corrected M5 column.

Sources: m5_prob_correct_split (seed 42), m5_prob_seed43_correct,
m5_prob_seed44_correct. Zero is computed from each seed's own panel (its SPL
needs no model run). Scale = each seed's own init window, current scoring
functions throughout.

Outputs:
  m5_seed_variance_correct.csv / m5_seed_variance_correct_per_seed.csv
  m5_correct_column.csv  (seed-42 corrected tab:prob M5 column incl. Zero)

Usage: py scripts/rebuild/t6b_score_correct.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_m5_long, preprocess_m5, train_eval_split_fixed_origin
from experiments.run_prob import _scaled_pinball_table, _series_naive_scale
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / "aistats2027_rebuild"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = [f"q_{q}" for q in QUANTILES]
RUNS = {42: "m5_prob_correct_split", 43: "m5_prob_seed43_correct",
        44: "m5_prob_seed44_correct"}
MODELS = ["TSB-HB", "AutoARIMA", "CP-IMAPA", "Zero"]

_sales = _cal = None


def panel(seed: int):
    global _sales, _cal
    if _sales is None:
        _sales, _cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    set_seed(seed)
    df = preprocess_m5(_sales, _cal, sample_size=5000)
    init, ev = train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
    for d in (init, ev):
        d["unique_id"] = d["unique_id"].astype(str)
        d["ds"] = pd.to_datetime(d["ds"])
    return init, ev


def main() -> None:
    rows = []
    col42 = None
    for seed, run in RUNS.items():
        p = OUT / run / "prob_quantiles.csv"
        if not p.exists():
            print(f"[t6b-c] seed {seed}: {run} missing, skipped", flush=True)
            continue
        init, ev = panel(seed)
        q = pd.read_csv(p).drop(columns=["protocol"], errors="ignore")
        q["unique_id"] = q["unique_id"].astype(str)
        q["ds"] = pd.to_datetime(q["ds"])
        zq = ev[["unique_id", "ds"]].copy()
        for c in QCOLS:
            zq[c] = 0.0
        zq["model"] = "Zero"
        q = pd.concat([q[["model", "unique_id", "ds"] + QCOLS],
                       zq[["model", "unique_id", "ds"] + QCOLS]],
                      ignore_index=True)
        merged = ev[["unique_id", "ds", "y"]].merge(q, on=["unique_id", "ds"],
                                                    how="inner")
        spl = _scaled_pinball_table(merged, init_set=init, quantiles=QUANTILES)
        if seed == 42:
            col42 = spl.copy()
        for m in sorted(spl.model.unique()):
            sub = spl[spl.model == m]
            rows.append({
                "seed": seed, "model": m,
                "spl_mean": float(sub.scaled_pinball.mean()),
                "spl_q90": float(sub[sub["quantile"] == 0.9].scaled_pinball.iloc[0]),
                "n_series": int(sub.n_series_scaled.max()),
            })
        print(f"[t6b-c] seed {seed}: done ({spl.model.nunique()} models)",
              flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "m5_seed_variance_correct_per_seed.csv", index=False)
    focus = per[per.model.isin(MODELS)]
    agg = focus.groupby("model").agg(
        spl_mean_avg=("spl_mean", "mean"), spl_mean_sd=("spl_mean", "std"),
        spl_q90_avg=("spl_q90", "mean"), spl_q90_sd=("spl_q90", "std"),
        n_seeds=("seed", "nunique")).reset_index()
    agg.to_csv(OUT / "m5_seed_variance_correct.csv", index=False)
    print(agg.round(4).to_string(index=False))

    if col42 is not None:
        col42.insert(0, "protocol", "fixed_init_2/3")
        col42.to_csv(OUT / "m5_correct_column.csv", index=False)
        print("[t6b-c] wrote m5_correct_column.csv (seed-42 corrected M5 "
              "column incl. Zero)")
    print("[t6b-c] wrote m5_seed_variance_correct(.csv/_per_seed.csv)",
          flush=True)


if __name__ == "__main__":
    main()
