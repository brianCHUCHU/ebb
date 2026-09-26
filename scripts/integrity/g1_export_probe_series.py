"""Gate 1 (export step): deterministic 12-series probe samples for the
TweedieGP scaling test on Online Retail and M5.

Sampling: rng = np.random.default_rng(42) over the sorted unique_id list of
each panel's paper protocol; indices recorded in the output. Exports long
frames (panel, unique_id, ds, y, is_train) for the TweedieGP venv runner.

Output: outputs/<date>/tweediegp_probe_input.csv

Usage: py scripts/integrity/g1_export_probe_series.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import (
    load_m5_long, load_online_retail, preprocess_m5, preprocess_online_retail,
    train_eval_split_fixed_origin,
)
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
N_PROBE = 12


def export(panel: str, init_set: pd.DataFrame, eval_set: pd.DataFrame) -> pd.DataFrame:
    uids = np.array(sorted(init_set["unique_id"].astype(str).unique()))
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(uids), size=N_PROBE, replace=False))
    chosen = uids[idx]
    print(f"[{panel}] probe indices {idx.tolist()}")
    frames = []
    for part, df in (("train", init_set), ("eval", eval_set)):
        sub = df[df["unique_id"].astype(str).isin(chosen)][["unique_id", "ds", "y"]].copy()
        sub["is_train"] = int(part == "train")
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    out["panel"] = panel
    out["unique_id"] = out["unique_id"].astype(str)
    return out.sort_values(["unique_id", "is_train", "ds"], ascending=[True, False, True])


def main() -> None:
    df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
    init_or, eval_or = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    set_seed(42)
    sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    m5 = preprocess_m5(sales, cal, sample_size=5000)
    init_m5, eval_m5 = train_eval_split_fixed_origin(m5, init_ratio=2 / 3, min_len=1)

    out = pd.concat([export("online_retail", init_or, eval_or),
                     export("m5", init_m5, eval_m5)], ignore_index=True)
    out.to_csv(OUT / "tweediegp_probe_input.csv", index=False)
    n_or = init_or["unique_id"].nunique()
    n_m5 = init_m5["unique_id"].nunique()
    (OUT / "tweediegp_probe_panelsizes.csv").write_text(
        f"panel,n_series\nonline_retail,{n_or}\nm5,{n_m5}\n", encoding="utf-8")
    print(f"wrote {OUT/'tweediegp_probe_input.csv'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
