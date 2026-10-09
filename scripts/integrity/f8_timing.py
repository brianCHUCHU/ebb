"""F8 (stage 1h): measured wall-clock under the audited configurations.

Two families of numbers for Appendix D / E.5:
  (1) joint selection time per panel EXCLUDING data loading -- re-measured
      here (t1_run_meta.json wall-clocks include loading);
  (2) final model-only prediction time: one full-window EB fit with B=20
      bootstrap refits, point prediction, and five-quantile prediction over
      the external span, for Carparts, Online Retail, and M5.

Configurations: OR (global, 0.95), M5 (mixture, 0.95), Auto (global, 0.99),
Carparts (global, 0.90), RAF (mixture, 0.997).

Output: outputs/integrity_<date>/f8_timing.csv + f8_run_meta.json

Usage: py scripts/integrity/f8_timing.py
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

from data_loading import (
    load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin,
    load_generic_long, train_eval_split_last_h, load_m5_long, preprocess_m5,
)
from experiments.run_prob import _build_regime_group_labels
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import (
    _split_init_head_tail, fit_eb_hurdle, predict_eb_hurdle, select_pooling_and_discount,
)
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
OUT.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
SEED = 42
CONFIG = {
    "online_retail": ("global", 0.95),
    "m5": ("mixture", 0.95),
    "auto": ("global", 0.99),
    "carparts": ("global", 0.90),
    "raf": ("mixture", 0.997),
}
PREDICT_PANELS = ["carparts", "online_retail", "m5"]


def load_panel(name: str):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    if name == "m5":
        set_seed(SEED)
        sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        df = preprocess_m5(sales, cal, sample_size=5000)
        return train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
    h = {"auto": 6, "carparts": 6, "raf": 12}[name]
    df = load_generic_long(ROOT / "data" / f"{name}_long.csv")
    return train_eval_split_last_h(df, h=h)


def main() -> None:
    t0 = time.time()
    rows = []
    for panel, (structure, w) in CONFIG.items():
        init_set, eval_set = load_panel(panel)  # loading excluded from timings
        n_series = int(init_set["unique_id"].nunique())

        # --- selection time, excluding load ---
        t_sel = time.perf_counter()
        head, _ = _split_init_head_tail(init_set, val_ratio=0.2)
        mix = mixture_group_labels(head, k=0)
        candidates = {"global": None,
                      "taxonomy": _build_regime_group_labels(head),
                      "mixture": mix.labels}
        select_pooling_and_discount(
            init_set, candidates,
            item_variance_mode="conjugate", item_variance_shrink_strength=20.0)
        sel_s = time.perf_counter() - t_sel

        row = {"panel": panel, "structure": structure, "discount": w,
               "n_series": n_series,
               "selection_s": round(sel_s, 1),
               "selection_ms_per_series": round(1000 * sel_s / n_series, 1)}

        # --- final fit + prediction time (B=20), selected panels ---
        if panel in PREDICT_PANELS:
            labels = (None if structure == "global"
                      else mixture_group_labels(init_set, k=0, fit_discount=w).labels)
            t_fit = time.perf_counter()
            params = fit_eb_hurdle(
                init_set, group_labels=labels,
                item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
                fit_discount=w, bootstrap_draws=20, bootstrap_seed=SEED)
            fit_s = time.perf_counter() - t_fit
            t_pt = time.perf_counter()
            predict_eb_hurdle(params, eval_set, quantiles=None)
            pt_s = time.perf_counter() - t_pt
            t_q = time.perf_counter()
            predict_eb_hurdle(params, eval_set, quantiles=QUANTILES,
                           include_hyper_uncertainty=True)
            q_s = time.perf_counter() - t_q
            row.update({"fit_b20_s": round(fit_s, 1),
                        "point_predict_s": round(pt_s, 1),
                        "quantile_predict_s": round(q_s, 1),
                        "final_total_s": round(fit_s + pt_s + q_s, 1),
                        "external_obs": int(len(eval_set))})
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "f8_timing.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "f8_timing", "commit": commit, "python": sys.version,
        "platform": platform.platform(), "processor": platform.processor(),
        "numpy": np.__version__, "pandas": pd.__version__, "seed": SEED,
        "note": "selection_s excludes data loading; final_total_s = full-window "
                "EB fit with 20 bootstrap refits + point + five-quantile "
                "prediction over the external span",
        "wall_clock_total_s": round(time.time() - t0, 1),
    }
    (OUT / "f8_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote f8_timing.csv")


if __name__ == "__main__":
    main()
