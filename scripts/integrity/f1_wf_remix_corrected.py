"""F1: walk-forward REMIX rows under the head-only-selection configurations.

The published walk-forward REMIX rows were produced under the superseded
selection outcome. This reruns ONLY the REMIX row with the audited
configurations -- Online Retail (global, w=0.95), seven-day blocks;
Carparts (global, w=0.90), monthly blocks -- using the identical online
mechanics and scoring functions as scripts/rebuild/t1_wf_or.py /
t2_wf_carparts.py (initialize -> predict block -> reveal block -> update;
discount applied once at initialization, never compounded). Baselines are
untouched.

Outputs (outputs/integrity_<date>/):
  wf_remix_corrected.csv      dataset, quantile-level SPL + mean, Cov@80,
                              Cov+@80, AIW@80, wall-clock
  wf_remix_quantiles_<panel>.csv  raw REMIX quantile frames (audit trail)
  f1_run_meta.json

Usage: py scripts/integrity/f1_wf_remix_corrected.py
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
    load_generic_long, load_online_retail, preprocess_online_retail,
    train_eval_split_fixed_origin, train_eval_split_last_h,
)
from experiments.protocols import iter_walk_forward_frames
from experiments.run_prob import _coverage_summary, _qcols, _scaled_pinball_table
from models.tsb_hb import (
    initialize_online_tsb_hb, predict_online_tsb_hb, update_online_tsb_hb,
)

OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
OUT.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = _qcols(QUANTILES)
SEED = 42

PANELS = {
    # panel: (loader kwargs, walk step, structure(None=global), w*)
    "online_retail": {"step": 7, "labels": None, "w": 0.95},
    "carparts": {"step": 1, "labels": None, "w": 0.90},
}


def load_panel(name: str):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    df = load_generic_long(ROOT / "data" / "carparts_long.csv")
    return train_eval_split_last_h(df, h=6)


def run_panel(name: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    init_set, eval_set = load_panel(name)
    state = initialize_online_tsb_hb(
        init_set,
        group_labels=cfg["labels"],
        bootstrap_draws=0, bootstrap_seed=SEED,
        group_shrink_strength=0.0,
        dynamic_occurrence=False, occurrence_discount=1.0,
        item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
        fit_discount=cfg["w"],
    )
    frames = []
    n_blocks = 0
    for frame in iter_walk_forward_frames(init_set, eval_set, step_size=cfg["step"]):
        hbq = predict_online_tsb_hb(
            state, frame.target, quantiles=QUANTILES,
            n_samples=2000, include_hyper_uncertainty=False,
        )
        for c in QCOLS:
            if c not in hbq.columns:
                hbq[c] = np.nan
        hbq["model"] = "TSB-HB"
        frames.append(hbq[["model", "unique_id", "ds"] + QCOLS])
        state = update_online_tsb_hb(state, frame.target)
        n_blocks += 1
    all_q = pd.concat(frames, ignore_index=True)
    all_q["ds"] = pd.to_datetime(all_q["ds"])
    all_q["unique_id"] = all_q["unique_id"].astype(str)
    all_q.to_csv(OUT / f"wf_remix_quantiles_{name}.csv", index=False)

    ev = eval_set[["unique_id", "ds", "y"]].copy()
    ev["ds"] = pd.to_datetime(ev["ds"])
    ev["unique_id"] = ev["unique_id"].astype(str)
    merged = ev.merge(all_q, on=["unique_id", "ds"], how="inner")

    spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=QUANTILES)
    cov = _coverage_summary(merged)
    cov_pos = _coverage_summary(merged[merged["y"] > 0].copy())
    wall = time.perf_counter() - t0

    spl_hb = spl[spl["model"] == "TSB-HB"]
    rows = []
    for _, r in spl_hb.iterrows():
        rows.append({"dataset": name, "metric": "scaled_pinball",
                     "quantile": r["quantile"], "value": float(r["scaled_pinball"])})
    rows.append({"dataset": name, "metric": "scaled_pinball", "quantile": "mean",
                 "value": float(spl_hb["scaled_pinball"].mean())})
    c = cov[cov["model"] == "TSB-HB"].iloc[0]
    cp = cov_pos[cov_pos["model"] == "TSB-HB"].iloc[0]
    rows += [
        {"dataset": name, "metric": "coverage80", "quantile": "", "value": float(c["Coverage@80"])},
        {"dataset": name, "metric": "coverage80_positive", "quantile": "", "value": float(cp["Coverage@80"])},
        {"dataset": name, "metric": "aiw80", "quantile": "", "value": float(c["AIW@80"])},
        {"dataset": name, "metric": "wall_clock_s", "quantile": "", "value": round(wall, 1)},
    ]
    meta = {"panel": name, "config": ("global", cfg["w"]), "blocks": n_blocks,
            "wall_clock_s": round(wall, 1)}
    print(f"[{name}] (global, {cfg['w']})  blocks={n_blocks}  "
          f"SPL mean={float(spl_hb['scaled_pinball'].mean()):.4f}  wall={wall:.0f}s",
          flush=True)
    return pd.DataFrame(rows), meta


def main() -> None:
    t0 = time.time()
    tables, metas = [], []
    for name, cfg in PANELS.items():
        tab, meta = run_panel(name, cfg)
        tables.append(tab)
        metas.append(meta)
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(OUT / "wf_remix_corrected.csv", index=False)
    print(out.to_string(index=False))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "f1_wf_remix_corrected", "commit": commit,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "seed": SEED,
        "mechanics": "initialize_online_tsb_hb -> predict block -> update block; "
                     "identical to scripts/rebuild/t2_wf_carparts.py REMIX path",
        "wall_clock_total_s": round(time.time() - t0, 1),
        "panels": metas,
    }
    (OUT / "f1_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote wf_remix_corrected.csv, f1_run_meta.json")


if __name__ == "__main__":
    main()
