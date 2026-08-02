"""Far-tail SPL of the constant-zero forecast on carparts/raf (E2 support).

Same M5-U scaling (in-sample naive MAE denominator) as trivial_baselines.py,
extended to the far-tail quantile set. Zero forecast: every quantile is 0.

Usage: py scripts/analysis/zero_fartail.py
Output: outputs/aistats2027/zero_fartail.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_generic_long, train_eval_split_last_h

QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.975, 0.99]


def main() -> None:
    rows = []
    for name, h in (("carparts", 6), ("raf", 12)):
        df = load_generic_long(ROOT / "data" / f"{name}_long.csv")
        init_set, eval_set = train_eval_split_last_h(df, h=h)
        init_sorted = init_set.sort_values(["unique_id", "ds"]).copy()
        init_sorted["ad"] = (init_sorted.y - init_sorted.groupby("unique_id").y.shift(1)).abs()
        scale = init_sorted.groupby("unique_id")["ad"].mean()
        scale = scale[scale > 0]
        m = eval_set[eval_set.unique_id.isin(scale.index)]
        denom = m.unique_id.map(scale).to_numpy()
        y = m.y.to_numpy(dtype=float)
        row = {"dataset": name}
        for q in QUANTILES:
            pin = np.maximum(q * y, (q - 1.0) * y) / denom  # forecast = 0
            row[f"q{q}"] = float(np.mean(pin))
        row["mean_all"] = float(np.mean([row[f"q{q}"] for q in QUANTILES]))
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "aistats2027" / "zero_fartail.csv", index=False)
    print("written")


if __name__ == "__main__":
    main()
