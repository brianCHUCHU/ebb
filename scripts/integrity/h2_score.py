"""Task 2 scorer: TweedieGP full OR / M5 quantiles -> results CSV.

Scores with the existing pipeline functions (identical scaling and
quantile post-processing): SPL per quantile + mean over {0.1..0.9},
far tail {0.95, 0.975, 0.99}, MAE (predictive mean), RMSSE.

Output: outputs/<date>/tweediegp_or_m5_results.csv

Usage: py scripts/integrity/h2_score.py
"""

from __future__ import annotations

import json
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
from experiments.run_prob import _enforce_monotonic_quantiles, _scaled_pinball_table
from metrics import rmsse
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
CENTRAL = [0.1, 0.25, 0.5, 0.75, 0.9]
FARTAIL = [0.95, 0.975, 0.99]
ALL_Q = CENTRAL + FARTAIL
SEED = 42


def load_panel(name: str):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    set_seed(SEED)
    sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    df = preprocess_m5(sales, cal, sample_size=5000)
    return train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)


def main() -> None:
    rows = []
    for panel in ("online_retail", "m5"):
        qpath = OUT / f"tweediegp_quantiles_{panel}.csv"
        if not qpath.exists():
            print(f"[h2-score] {panel}: missing, skipped", flush=True)
            continue
        meta = json.loads((OUT / f"h2_run_meta_{panel}.json").read_text())
        q = pd.read_csv(qpath)
        q["unique_id"] = q["unique_id"].astype(str)
        q["ds"] = pd.to_datetime(q["ds"])
        q["model"] = "TweedieGP"
        for c in [f"q_{x}" for x in ALL_Q]:
            q[c] = np.maximum(q[c].astype(float), 0.0)
        q = _enforce_monotonic_quantiles(q, quantiles=ALL_Q)

        init_set, eval_set = load_panel(panel)
        ev = eval_set[["unique_id", "ds", "y"]].copy()
        ev["unique_id"] = ev["unique_id"].astype(str)
        ev["ds"] = pd.to_datetime(ev["ds"])
        merged = ev.merge(q, on=["unique_id", "ds"], how="inner")
        n_series = int(merged.unique_id.nunique())
        print(f"[h2-score] {panel}: {n_series} series joined, "
              f"failed-after-retries {meta['n_failed_after_3_attempts']}", flush=True)

        spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=ALL_Q)
        spl = spl[spl["model"] == "TweedieGP"]

        def add(metric, quantile, value):
            rows.append({"dataset": panel, "protocol": "fixed_origin",
                         "model": "TweedieGP", "metric": metric,
                         "quantile": quantile, "value": value,
                         "n_series": n_series,
                         "wall_clock_sec": meta["wall_clock_total_s"],
                         "seed": SEED})

        for _, r in spl.iterrows():
            add("SPL", r["quantile"], r["scaled_pinball"])
        central = spl[spl["quantile"].isin(CENTRAL)]["scaled_pinball"]
        add("SPL_mean", "NA", float(central.mean()))
        m = merged.dropna(subset=["mean"]).copy()
        m["y_pred"] = m["mean"].astype(float)
        add("MAE", "NA", float((m.y_pred - m.y).abs().mean()))
        add("RMSSE", "NA", rmsse(init_set, m))
        add("n_failed_series", "NA", meta["n_failed_after_3_attempts"])
        add("n_zero_fallback", "NA", meta["n_zero_fallback"])

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tweediegp_or_m5_results.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
