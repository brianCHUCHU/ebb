"""Integrity: M5 seed 43/44 leakage-safe joint selection (head-only labels).

Same protocol as t1_leakage_safe_selection.py, M5 panel only, for the two
robustness seeds. Seed 42's corrected selection moved w* from 0.90 to 0.95;
this run establishes whether the corrected structure/discount is stable
across the three published 5,000-series samples.

Outputs (appended under outputs/integrity_<date>/):
  selection_surfaces_m5_seeds.csv   panel, seed, structure, discount, validation_spl
  selected_pairs_m5_seeds.csv
  t1b_run_meta.json

Usage: py scripts/integrity/t1b_m5_seed_selection.py [seed ...]   (default 43 44)
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

from data_loading import load_m5_long, preprocess_m5, train_eval_split_fixed_origin
from experiments.run_prob import _build_regime_group_labels
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import _split_init_head_tail, select_pooling_and_discount
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    seeds = [int(s) for s in (sys.argv[1:] or ["43", "44"])]
    t0_all = time.time()
    sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    surface_rows, pair_rows = [], []
    for seed in seeds:
        t0 = time.time()
        set_seed(seed)
        df = preprocess_m5(sales, cal, sample_size=5000)
        init, _ = train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
        head, _tail = _split_init_head_tail(init, val_ratio=0.2)
        mix_res = mixture_group_labels(head, k=0)
        candidates = {
            "global": None,
            "taxonomy": _build_regime_group_labels(head),
            "mixture": mix_res.labels,
        }
        sel_name, sel_w, diag = select_pooling_and_discount(
            init, candidates,
            item_variance_mode="conjugate",
            item_variance_shrink_strength=20.0,
        )
        wall = time.time() - t0
        diag = diag.rename(columns={"scaled_pinball": "validation_spl"})
        diag.insert(0, "seed", seed)
        diag.insert(0, "panel", "m5")
        surface_rows.append(diag[["panel", "seed", "structure", "discount", "validation_spl"]])
        pair_rows.append({"panel": "m5", "seed": seed, "structure": sel_name,
                          "discount": sel_w, "mixture_k": int(mix_res.k),
                          "wall_clock_s": round(wall, 1)})
        print(f"[m5 seed {seed}] selected ({sel_name}, {sel_w})  K={mix_res.k}  "
              f"wall={wall:.1f}s", flush=True)

    pd.concat(surface_rows, ignore_index=True).to_csv(
        OUT / "selection_surfaces_m5_seeds.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "selected_pairs_m5_seeds.csv", index=False)
    print(pairs.to_string(index=False))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "t1b_m5_seed_selection",
        "commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seeds": seeds,
        "protocol": "init 2/3, val_ratio 0.2, head-only labels, existing scoring",
        "wall_clock_total_s": round(time.time() - t0_all, 1),
        "per_seed": pair_rows,
    }
    (OUT / "t1b_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote selection_surfaces_m5_seeds.csv, selected_pairs_m5_seeds.csv, t1b_run_meta.json")


if __name__ == "__main__":
    main()
