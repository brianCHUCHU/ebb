"""D1: tuned TSB on all five panels under the identical protocol.

Grid-searches TSB's (alpha_d, alpha_p) on the same internal chronological
80/20 split REMIX selects on (25 candidates, scaled-MAE criterion by default;
an MAE-criterion column is also reported), refits on the full initialization
window, and scores on the untouched evaluation span. Splits are byte-identical
to the paper's (same loaders and split helpers as run_point / trivial_baselines).

Usage: py scripts/analysis/tsb_tuned_panels.py [dataset1,dataset2,...]
Output: outputs/aistats2027/tsb_tuned_panels.csv
"""

from __future__ import annotations

import sys
import time
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
from models.tsb_tuned import fit_predict_tsb_tuned, TSB_TUNED_COL

FREQ = {"online_retail": "D", "m5": "D", "auto": "MS", "carparts": "MS", "raf": "MS"}


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


def main() -> None:
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        "online_retail", "m5", "auto", "carparts", "raf",
    ]
    out_path = ROOT / "outputs" / "aistats2027" / "tsb_tuned_panels.csv"
    rows = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
        rows = [r for r in rows if r["dataset"] not in datasets]

    for ds in datasets:
        init_set, eval_set = load_dataset(ds)
        freq = FREQ[ds]
        criteria = ("scaled_mae", "mae") if "--both-criteria" in sys.argv else ("scaled_mae",)
        for criterion in criteria:
            t0 = time.perf_counter()
            pred, chosen, diag = fit_predict_tsb_tuned(
                init_set[["unique_id", "ds", "y"]].copy(),
                eval_set[["unique_id", "ds", "y"]].copy(),
                freq=freq,
                criterion=criterion,
            )
            if pred.empty:
                print(f"{ds} [{criterion}]: EMPTY prediction, skipped")
                continue
            m = eval_set[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
            m = m.rename(columns={TSB_TUNED_COL: "y_pred"})
            mae = float((m.y_pred - m.y).abs().mean())
            rmse = float(np.sqrt(((m.y_pred - m.y) ** 2).mean()))
            rm = rmsse(init_set, m)
            row = {
                "dataset": ds, "criterion": criterion,
                "alpha_d": chosen["alpha_d"], "alpha_p": chosen["alpha_p"],
                "MAE": mae, "RMSE": rmse, "RMSSE": rm,
                "seconds": time.perf_counter() - t0,
            }
            rows.append(row)
            if diag is not None and not diag.empty:
                diag.to_csv(
                    ROOT / "outputs" / "aistats2027" / f"tsb_tuned_diag_{ds}_{criterion}.csv",
                    index=False,
                )
            print(f"{ds:13s} [{criterion:10s}] alpha_d={chosen['alpha_d']:.2f} "
                  f"alpha_p={chosen['alpha_p']:.2f} MAE={mae:.4f} RMSSE={rm:.4f} "
                  f"({row['seconds']:.0f}s)", flush=True)
        # Incremental save so a killed run keeps completed panels.
        pd.DataFrame(rows)[[
            "dataset", "criterion", "alpha_d", "alpha_p", "MAE", "RMSE", "RMSSE", "seconds",
        ]].to_csv(out_path, index=False)

    out = pd.DataFrame(rows)[[
        "dataset", "criterion", "alpha_d", "alpha_p", "MAE", "RMSE", "RMSSE", "seconds",
    ]]
    out.to_csv(out_path, index=False)
    print(out.round(4).to_string(index=False))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
