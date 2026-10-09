"""Spec Task 3 scorer: TweedieGP quantiles -> spec-schema results CSV.

Consumes tweediegp_quantiles_{dataset}.csv produced by the venv runner and
scores with the existing pipeline functions (identical scaling / quantile
post-processing): SPL per quantile + mean over {0.1..0.9}, far tail
{0.95,0.975,0.99}, MAE (predictive mean), RMSSE.

Output: outputs/paper_rebuild/tweedie_gp_fixed_origin.csv

Usage: py scripts/rebuild/t3_score.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_generic_long, train_eval_split_last_h
from experiments.run_prob import _enforce_monotonic_quantiles, _scaled_pinball_table
from metrics import rmsse

OUT = ROOT / "outputs" / "paper_rebuild"
H = {"auto": 6, "carparts": 6, "raf": 12}
CENTRAL = [0.1, 0.25, 0.5, 0.75, 0.9]
FARTAIL = [0.95, 0.975, 0.99]
ALL_Q = CENTRAL + FARTAIL
SEED = 42


def main() -> None:
    rows = []
    for ds, h in H.items():
        qpath = OUT / f"tweediegp_quantiles_{ds}.csv"
        if not qpath.exists():
            print(f"[t3-score] {ds}: quantiles missing, skipped", flush=True)
            continue
        meta = json.loads((OUT / f"t3_run_meta_{ds}.json").read_text())
        q = pd.read_csv(qpath)
        q["unique_id"] = q["unique_id"].astype(str)
        q["ds"] = pd.to_datetime(q["ds"])
        q["model"] = "TweedieGP"
        # zero-truncate + monotonize exactly like the pipeline
        for c in [f"q_{x}" for x in ALL_Q]:
            q[c] = np.maximum(q[c].astype(float), 0.0)
        q = _enforce_monotonic_quantiles(q, quantiles=ALL_Q)

        df = load_generic_long(ROOT / "data" / f"{ds}_long.csv")
        init_set, eval_set = train_eval_split_last_h(df, h=h)
        ev = eval_set[["unique_id", "ds", "y"]].copy()
        ev["unique_id"] = ev["unique_id"].astype(str)
        ev["ds"] = pd.to_datetime(ev["ds"])
        merged = ev.merge(q, on=["unique_id", "ds"], how="inner")
        n_series = int(merged.unique_id.nunique())
        n_failed = int(merged[merged["q_0.5"].isna()].unique_id.nunique())
        print(f"[t3-score] {ds}: {n_series} series joined, "
              f"{n_failed} with failed fits", flush=True)

        spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=ALL_Q)
        spl = spl[spl["model"] == "TweedieGP"]

        def add(metric, quantile, value):
            rows.append({
                "dataset": ds, "protocol": "fixed_origin", "model": "TweedieGP",
                "metric": metric, "quantile": quantile, "value": value,
                "n_series": n_series,
                "wall_clock_sec": meta["wall_clock_sec"], "seed": SEED,
            })

        for _, r in spl.iterrows():
            add("SPL", r["quantile"], r["scaled_pinball"])
        central = spl[spl["quantile"].isin(CENTRAL)]["scaled_pinball"]
        add("SPL_mean", "NA", float(central.mean()))

        m = merged.dropna(subset=["mean"]).copy()
        m["y_pred"] = m["mean"].astype(float)
        add("MAE", "NA", float((m.y_pred - m.y).abs().mean()))
        add("RMSSE", "NA", rmsse(init_set, m))
        add("n_failed_series", "NA", n_failed)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tweedie_gp_fixed_origin.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("[t3-score] wrote tweedie_gp_fixed_origin.csv", flush=True)


if __name__ == "__main__":
    main()
