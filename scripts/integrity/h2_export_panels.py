"""Task 2 (export step): full Online Retail and M5 panels for the TweedieGP
venv runner, under our paper protocols.

Output: outputs/<date>/tweediegp_panel_{online_retail,m5}.csv
        (unique_id, ds, y, is_train)

Usage: py scripts/integrity/h2_export_panels.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from data_loading import (
    load_m5_long, load_online_retail, preprocess_m5, preprocess_online_retail,
    train_eval_split_fixed_origin,
)
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)


def export(panel: str, init_set: pd.DataFrame, eval_set: pd.DataFrame) -> None:
    frames = []
    for flag, df in ((1, init_set), (0, eval_set)):
        sub = df[["unique_id", "ds", "y"]].copy()
        sub["is_train"] = flag
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    out["unique_id"] = out["unique_id"].astype(str)
    out.to_csv(OUT / f"tweediegp_panel_{panel}.csv", index=False)
    print(f"{panel}: {out['unique_id'].nunique()} series, {len(out)} rows")


def main() -> None:
    if sys.argv[1:] == ["carparts"]:
        from data_loading import load_generic_long, train_eval_split_last_h
        df = load_generic_long(ROOT / "data" / "carparts_long.csv")
        export("carparts", *train_eval_split_last_h(df, h=6))
        return
    df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
    init_or, eval_or = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    export("online_retail", init_or, eval_or)
    set_seed(42)
    sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    m5 = preprocess_m5(sales, cal, sample_size=5000)
    init_m5, eval_m5 = train_eval_split_fixed_origin(m5, init_ratio=2 / 3, min_len=1)
    export("m5", init_m5, eval_m5)


if __name__ == "__main__":
    main()
