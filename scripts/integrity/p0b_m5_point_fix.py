"""Integrity P0b: M5 point EBB row at the corrected configuration.

`run_point --dataset m5` routes to the M5-specific branch, which honors
`--hb-grouping` only for `select` and otherwise ignores both `--hb-grouping`
and `--hb-fit-discount` (the P0 driver's M5 point run therefore silently
produced global @ w=1). This script reproduces the generic forced-config
semantics on the M5 point protocol: mixture labels learned on the full
initialization window at the corrected w*=0.95 (matching run_point's generic
mixture branch and run_prob's mixture branch), one EB fit, one prediction,
scored with the existing `evaluate_point_models`.

Output: overwrites outputs/integrity_<date>/m5_point_ebb_corrected/
        point_metrics_m5.csv (the invalid global@w=1 artifact) and writes
        p0b_run_meta.json.

Usage: py scripts/integrity/p0b_m5_point_fix.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_m5_long, preprocess_m5, train_eval_split_fixed_origin
from experiments.protocols import evaluate_point_models
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import fit_eb_hurdle, predict_eb_hurdle
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
DEST = OUT / "m5_point_ebb_corrected"
W_STAR = 0.95


def main() -> None:
    t0 = time.time()
    set_seed(42)
    sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    df = preprocess_m5(sales, cal, sample_size=5000)
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=2.0 / 3.0, min_len=1)

    mix_res = mixture_group_labels(init_set, k=0, fit_discount=W_STAR)
    params = fit_eb_hurdle(
        init_set,
        group_labels=mix_res.labels,
        group_shrink_strength=0.0,
        item_variance_mode="conjugate",
        item_variance_shrink_strength=20.0,
        fit_discount=W_STAR,
    )
    pred = predict_eb_hurdle(params, eval_set, quantiles=None)
    merged = eval_set[["unique_id", "ds", "y"]].merge(
        pred.rename(columns={"yhat": "EB-Hurdle"}), on=["unique_id", "ds"], how="left")
    metrics = evaluate_point_models(init_set, merged, model_cols=["EB-Hurdle"])
    metrics.insert(0, "protocol", "fixed")
    DEST.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DEST / "point_metrics_m5.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"mixture K={mix_res.k}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "p0b_m5_point_fix",
        "commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": 42,
        "config": {"structure": "mixture", "discount": W_STAR, "mixture_k": int(mix_res.k)},
        "reason": "run_point m5 branch ignores forced --hb-grouping/--hb-fit-discount",
        "wall_clock_total_s": round(time.time() - t0, 1),
    }
    (OUT / "p0b_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote m5_point_ebb_corrected/point_metrics_m5.csv, p0b_run_meta.json")


if __name__ == "__main__":
    main()
