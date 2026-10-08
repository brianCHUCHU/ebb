"""Spec Task 4 scorer: DeepState quantiles -> hierarchical_prior_work.csv.

Scores deepstate_quantiles_{or1000,carparts}.csv against the exported panels
with the existing pipeline functions. Wall-clock from t4_run_meta_*.json
(the spec explicitly asks for compute cost).

Usage: py scripts/rebuild/t4_score.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from experiments.run_prob import _enforce_monotonic_quantiles, _scaled_pinball_table
from metrics import rmsse

OUT = ROOT / "outputs" / "paper_rebuild"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
SEED = 42


def main() -> None:
    rows = []
    for name in ("or1000", "carparts"):
        qp = OUT / f"deepstate_quantiles_{name}.csv"
        if not qp.exists():
            print(f"[t4-score] {name}: missing, skipped", flush=True)
            continue
        meta = json.loads((OUT / f"t4_run_meta_{name}.json").read_text())
        q = pd.read_csv(qp)
        q["unique_id"] = q["unique_id"].astype(str)
        q["ds"] = pd.to_datetime(q["ds"])
        q["model"] = "DeepState"
        for c in [f"q_{x}" for x in QUANTILES]:
            q[c] = np.maximum(q[c].astype(float), 0.0)
        q = _enforce_monotonic_quantiles(q, quantiles=QUANTILES)

        init = pd.read_csv(OUT / f"panel_{name}_init.csv")
        ev = pd.read_csv(OUT / f"panel_{name}_eval.csv")
        for df in (init, ev):
            df["unique_id"] = df["unique_id"].astype(str)
            df["ds"] = pd.to_datetime(df["ds"])
        merged = ev[["unique_id", "ds", "y"]].merge(
            q, on=["unique_id", "ds"], how="inner")
        n_series = int(merged.unique_id.nunique())
        print(f"[t4-score] {name}: {n_series} series, {len(merged)} rows",
              flush=True)

        spl = _scaled_pinball_table(merged, init_set=init, quantiles=QUANTILES)
        spl = spl[spl["model"] == "DeepState"]

        def add(metric, quantile, value):
            rows.append({
                "dataset": name, "protocol": "fixed_origin_chained",
                "model": "DeepState (Amazon ISSM line; NOT Seeger-2016 itself)",
                "metric": metric, "quantile": quantile, "value": value,
                "n_series": n_series,
                "wall_clock_sec": meta["wall_clock_sec"], "seed": SEED,
            })

        for _, r in spl.iterrows():
            add("SPL", r["quantile"], r["scaled_pinball"])
        add("SPL_mean", "NA", float(spl["scaled_pinball"].mean()))
        m = merged.dropna(subset=["mean"]).copy()
        m["y_pred"] = m["mean"].astype(float)
        add("MAE", "NA", float((m.y_pred - m.y).abs().mean()))
        add("RMSSE", "NA", rmsse(init, m))
        add("train_seconds", "NA", meta["train_seconds"])

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "hierarchical_prior_work.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("[t4-score] wrote hierarchical_prior_work.csv", flush=True)


if __name__ == "__main__":
    main()
