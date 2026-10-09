"""Integrity task 1: leakage-safe joint selection audit, five panels.

The legacy `*_prob_select` runs (2026-07-19) built the taxonomy and mixture
candidate labels on the FULL initialization window, so the partition had seen
the 20% validation tail that subsequently scored it -- the v4 draft's
adjudication is that "mixture wins everywhere" was the fingerprint of exactly
this leakage. This audit rebuilds the joint selection with the partition
builder restricted to the fitting head:

  head, tail = _split_init_head_tail(init, val_ratio=0.2)   # existing function
  taxonomy   = _build_regime_group_labels(head)             # existing function
  mixture    = mixture_group_labels(head, k=0).labels       # existing function
  select_pooling_and_discount(init, candidates, ...)        # existing function

`select_pooling_and_discount` re-derives the identical head/tail split
internally, so the labels never see the scoring tail. No model, hyper, or
selection logic is modified; scoring is the existing implementation.

Outputs (to outputs/integrity_<date>/):
  selection_surfaces.csv   panel, structure, discount, validation_spl
  selected_pairs.csv       panel, structure, discount, mixture_k, wall_clock_s
  t1_run_meta.json

Usage: py scripts/integrity/t1_leakage_safe_selection.py [panel ...]
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
    load_generic_long, load_m5_long, load_online_retail,
    preprocess_m5, preprocess_online_retail,
    train_eval_split_fixed_origin, train_eval_split_last_h,
)
from experiments.run_prob import _build_regime_group_labels
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import _split_init_head_tail, select_pooling_and_discount
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
OUT.mkdir(parents=True, exist_ok=True)

PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]
EXPECTED = {"online_retail": "taxonomy", "m5": "mixture", "auto": "global",
            "carparts": "taxonomy", "raf": "taxonomy"}


def load_init(name: str) -> pd.DataFrame:
    """Initialization windows exactly as in the paper protocols (mirrors
    scripts/analysis/leverage_dual_lambda.py, which reproduced the published
    medians digit-level)."""
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
        init, _ = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
        return init
    if name == "m5":
        set_seed(42)
        sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        df = preprocess_m5(sales, cal, sample_size=5000)
        init, _ = train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
        return init
    h = {"auto": 6, "carparts": 6, "raf": 12}[name]
    df = load_generic_long(ROOT / "data" / f"{name}_long.csv")
    init, _ = train_eval_split_last_h(df, h=h)
    return init


def main() -> None:
    panels = sys.argv[1:] or PANELS
    t0_all = time.time()
    surface_rows, pair_rows = [], []
    for name in panels:
        t0 = time.time()
        init = load_init(name)
        head, tail = _split_init_head_tail(init, val_ratio=0.2)
        print(f"[{name}] init rows={len(init)} uids={init['unique_id'].nunique()} "
              f"head rows={len(head)} tail rows={len(tail)}", flush=True)

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
        diag.insert(0, "panel", name)
        surface_rows.append(diag[["panel", "structure", "discount", "validation_spl"]])
        pair_rows.append({
            "panel": name, "structure": sel_name, "discount": sel_w,
            "mixture_k": int(mix_res.k), "wall_clock_s": round(wall, 1),
            "expected_structure": EXPECTED[name],
            "matches_expected": sel_name == EXPECTED[name],
        })
        print(f"[{name}] selected ({sel_name}, {sel_w})  mixture K={mix_res.k}  "
              f"wall={wall:.1f}s  expected={EXPECTED[name]} "
              f"{'MATCH' if sel_name == EXPECTED[name] else 'MISMATCH'}", flush=True)

    pd.concat(surface_rows, ignore_index=True).to_csv(OUT / "selection_surfaces.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "selected_pairs.csv", index=False)
    print(pairs.to_string(index=False))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "t1_leakage_safe_selection",
        "commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": {"m5_sample": 42},
        "val_ratio": 0.2,
        "quantiles": [0.5, 0.75, 0.9],
        "item_variance_mode": "conjugate",
        "item_variance_shrink_strength": 20.0,
        "label_builder_scope": "selection fitting head only (leakage-safe)",
        "wall_clock_total_s": round(time.time() - t0_all, 1),
        "per_panel": pair_rows,
    }
    (OUT / "t1_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {OUT}/selection_surfaces.csv, selected_pairs.csv, t1_run_meta.json")


if __name__ == "__main__":
    main()
