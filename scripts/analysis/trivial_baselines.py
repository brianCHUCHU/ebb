"""B4: trivial baselines (constant-zero and naive last-value) on all panels.

Computed directly from the data under the identical splits — no model runs
needed. Reports MAE, RMSSE, MASE, and SPL (zero forecast: all quantiles 0;
naive: all quantiles = last observed value) for exact comparability with
Tables 1-2.

Usage: py scripts/analysis/trivial_baselines.py
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
    load_generic_long, train_eval_split_last_h, load_m5_long, preprocess_m5,
)
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed
from metrics import rmsse

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def load_dataset(name: str):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    if name == "m5":
        set_seed(42)
        sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        df = preprocess_m5(sales, cal, sample_size=5000)
        return train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
    h = {"auto": 6, "carparts": 6, "raf": 12}[name]
    df = load_generic_long(ROOT / "data" / f"{name}_long.csv")
    return train_eval_split_last_h(df, h=h)


def eval_constant(init_set: pd.DataFrame, eval_set: pd.DataFrame, pred: pd.Series, label: str):
    m = eval_set[["unique_id", "ds", "y"]].copy()
    m["y_pred"] = m["unique_id"].map(pred).fillna(0.0)
    mae = float((m.y_pred - m.y).abs().mean())
    rm = rmsse(init_set, m)
    # MASE: per-series scaling by in-sample naive MAE
    init_sorted = init_set.sort_values(["unique_id", "ds"]).copy()
    init_sorted["ad"] = (init_sorted.y - init_sorted.groupby("unique_id").y.shift(1)).abs()
    scale = init_sorted.groupby("unique_id")["ad"].mean()
    scale = scale[scale > 0]
    mm = m[m.unique_id.isin(scale.index)]
    mase = float(((mm.y_pred - mm.y).abs() / mm.unique_id.map(scale)).mean())
    # SPL with all quantiles equal to the constant prediction
    denom = mm.unique_id.map(scale).to_numpy()
    spl_qs = []
    for q in QUANTILES:
        err = mm.y.to_numpy() - mm.y_pred.to_numpy()
        pin = np.maximum(q * err, (q - 1.0) * err) / denom
        spl_qs.append(float(np.mean(pin)))
    return {"model": label, "MAE": mae, "RMSSE": rm, "MASE": mase,
            "SPL_mean": float(np.mean(spl_qs)), "SPL_q90": spl_qs[-1]}


def main() -> None:
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["online_retail", "m5", "auto", "carparts", "raf"]
    out_path = ROOT / "outputs" / "paper_runs" / "trivial_baselines.csv"
    rows = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
        rows = [r for r in rows if r["dataset"] not in datasets]
    for ds in datasets:
        init_set, eval_set = load_dataset(ds)
        zero = pd.Series(0.0, index=init_set.unique_id.unique())
        last = init_set.sort_values(["unique_id", "ds"]).groupby("unique_id").y.last()
        for pred, label in [(zero, "Zero"), (last, "Naive-last")]:
            r = eval_constant(init_set, eval_set, pred, label)
            r["dataset"] = ds
            rows.append(r)
        print(f"{ds} done")
    out = pd.DataFrame(rows)[["dataset", "model", "MAE", "RMSSE", "MASE", "SPL_mean", "SPL_q90"]]
    out.to_csv(out_path, index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
