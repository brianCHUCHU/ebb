"""Export split panels as plain CSVs for the isolated-env runners (Task 4/8).

or1000: seed-42 random 1,000-series subsample of Online Retail (spec allows
1,000 for Task 4); carparts: full panel.

Usage: py scripts/rebuild/export_panels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import (
    load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin,
    load_generic_long, train_eval_split_last_h,
)

OUT = ROOT / "outputs" / "aistats2027_rebuild"
OUT.mkdir(parents=True, exist_ok=True)

df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
init, ev = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
rng = np.random.default_rng(42)
uids = np.sort(init.unique_id.unique())
keep = set(rng.choice(uids, size=1000, replace=False))
init[init.unique_id.isin(keep)].to_csv(OUT / "panel_or1000_init.csv", index=False)
ev[ev.unique_id.isin(keep)].to_csv(OUT / "panel_or1000_eval.csv", index=False)
print("or1000 exported:", len(keep), "series")

dfc = load_generic_long(ROOT / "data" / "carparts_long.csv")
ci, ce = train_eval_split_last_h(dfc, h=6)
ci.to_csv(OUT / "panel_carparts_init.csv", index=False)
ce.to_csv(OUT / "panel_carparts_eval.csv", index=False)
print("carparts exported:", ci.unique_id.nunique(), "series")
