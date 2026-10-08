"""Spec Task 6b scorer: M5 seed variance from three seed runs.

Consumes prob_quantiles.csv from:
  seed 42: outputs/paper_runs/m5_prob_select (EBB) + m5_prob_fixed (baselines)
  seed 43/44: outputs/paper_rebuild/m5_prob_seed{43,44}/ (fresh runs)
Computes mean SPL (q in {0.1..0.9}) for EBB, AutoARIMA, CP-IMAPA, Zero per
seed, then mean +/- sd across seeds. Scale denominators are recomputed per
seed from that seed's own initialization window (each seed is its own panel).

Output: outputs/paper_rebuild/m5_seed_variance.csv
Usage: py scripts/rebuild/t6b_score.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_m5_long, preprocess_m5, train_eval_split_fixed_origin
from experiments.run_prob import _qcols, _scaled_pinball_table
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / "paper_rebuild"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = _qcols(QUANTILES)
MODELS = ["EB-Hurdle", "AutoARIMA", "CP-IMAPA", "Zero"]

SOURCES = {
    42: [ROOT / "outputs" / "paper_runs" / "m5_prob_select",
         ROOT / "outputs" / "paper_runs" / "m5_prob_fixed"],
    43: [OUT / "m5_prob_seed43"],
    44: [OUT / "m5_prob_seed44"],
}


def load_panel(seed: int):
    set_seed(seed)
    sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    df = preprocess_m5(sales, cal, sample_size=5000)
    return train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)


def main() -> None:
    recs = []
    for seed, dirs in SOURCES.items():
        frames = []
        for d in dirs:
            p = d / "prob_quantiles.csv"
            if p.exists():
                q = pd.read_csv(p).drop(columns=["protocol"], errors="ignore")
                q["ds"] = pd.to_datetime(q["ds"])
                q["unique_id"] = q["unique_id"].astype(str)
                frames.append(q)
        if not frames:
            print(f"[t6b] seed {seed}: no runs found, skipped", flush=True)
            continue
        all_q = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["model", "unique_id", "ds"], keep="first")
        init_set, eval_set = load_panel(seed)
        ev = eval_set[["unique_id", "ds", "y"]].copy()
        ev["ds"] = pd.to_datetime(ev["ds"])
        ev["unique_id"] = ev["unique_id"].astype(str)
        zq = ev[["unique_id", "ds"]].copy()
        for c in QCOLS:
            zq[c] = 0.0
        zq["model"] = "Zero"
        all_q = pd.concat([all_q, zq[["model", "unique_id", "ds"] + QCOLS]],
                          ignore_index=True)
        merged = ev.merge(all_q, on=["unique_id", "ds"], how="inner")
        spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=QUANTILES)
        for m in MODELS:
            sub = spl[spl["model"] == m]
            if sub.empty:
                print(f"[t6b] seed {seed}: {m} missing", flush=True)
                continue
            recs.append({"seed": seed, "model": m,
                         "spl_mean": float(sub["scaled_pinball"].mean()),
                         "spl_q90": float(sub[sub["quantile"] == 0.9]["scaled_pinball"].iloc[0]),
                         "n_series": int(merged[merged.model == m].unique_id.nunique())})
        print(f"[t6b] seed {seed}: done", flush=True)

    per_seed = pd.DataFrame(recs)
    per_seed.to_csv(OUT / "m5_seed_variance_per_seed.csv", index=False)
    agg = per_seed.groupby("model").agg(
        spl_mean_avg=("spl_mean", "mean"), spl_mean_sd=("spl_mean", "std"),
        spl_q90_avg=("spl_q90", "mean"), spl_q90_sd=("spl_q90", "std"),
        n_seeds=("seed", "nunique")).reset_index()
    agg.to_csv(OUT / "m5_seed_variance.csv", index=False)
    print(per_seed.round(4).to_string(index=False))
    print(agg.round(4).to_string(index=False))
    print("[t6b] wrote m5_seed_variance.csv", flush=True)


if __name__ == "__main__":
    main()
