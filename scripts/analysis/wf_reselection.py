"""E4: walk-forward re-selection cadence on Online Retail (W4 follow-up).

Three arms under an identical block-wise walk-forward protocol (7-day blocks,
full closed-form refit on all revealed data at every block):

  never   -- (g, w) selected once at the origin, held fixed
  every4  -- joint (g, w) re-selection every 4 blocks (~monthly)
  every1  -- joint (g, w) re-selection at every block

Re-selection = recompute candidate partitions (taxonomy + mixture labels) on
all revealed data, then re-run the joint chronological-validation selection.
Between re-selections the structure/labels/w stay fixed, but hyperparameters
and posteriors are refit on the revealed window each block (cheap, closed
form). Reported: overall MAE / RMSSE, plus the selection trajectory.

Usage: py scripts/analysis/wf_reselection.py
Output: outputs/aistats2027/wf_reselection.csv (+ _trajectory.csv)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from experiments.protocols import iter_walk_forward_frames
from metrics import rmsse, compute_adi_cv2, classify_adi_cv2
from models.tsb_hb import fit_tsb_hb, predict_tsb_hb, select_pooling_and_discount
from models.mixture_pooling import mixture_group_labels

ITEM_VARIANCE_MODE = "conjugate"
VARIANCE_PRIOR_DF = 20.0
STEP = 7


def taxonomy_labels(df: pd.DataFrame):
    feats = compute_adi_cv2(df)
    if feats.empty:
        return None
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)
    return feats.set_index("unique_id")["category"].astype(str)


def select_on(train_df: pd.DataFrame):
    cands = {
        "global": None,
        "taxonomy": taxonomy_labels(train_df),
        "mixture": mixture_group_labels(train_df, k=0).labels,
    }
    name, w, _ = select_pooling_and_discount(
        train_df, cands,
        item_variance_mode=ITEM_VARIANCE_MODE,
        item_variance_shrink_strength=VARIANCE_PRIOR_DF,
    )
    return name, w, cands[name]


def run_arm(init_set, eval_set, reselect_every: int):
    """reselect_every=0 -> origin selection only."""
    t0 = time.perf_counter()
    name, w, labels = select_on(init_set)
    traj = [{"block": 0, "structure": name, "w": w}]
    preds = []
    for b, frame in enumerate(iter_walk_forward_frames(init_set, eval_set, step_size=STEP)):
        if reselect_every and b > 0 and b % reselect_every == 0:
            name, w, labels = select_on(frame.history)
            traj.append({"block": b, "structure": name, "w": w})
        params = fit_tsb_hb(
            frame.history,
            group_labels=labels,
            group_shrink_strength=0.0,
            item_variance_mode=ITEM_VARIANCE_MODE,
            item_variance_shrink_strength=VARIANCE_PRIOR_DF,
            fit_discount=w,
        )
        p = predict_tsb_hb(params, frame.target, quantiles=None)
        preds.append(frame.target[["unique_id", "ds", "y"]].merge(p, on=["unique_id", "ds"], how="left"))
    merged = pd.concat(preds, ignore_index=True).rename(columns={"yhat": "y_pred"})
    mae = float((merged.y_pred - merged.y).abs().mean())
    rmse = float(np.sqrt(((merged.y_pred - merged.y) ** 2).mean()))
    rm = rmsse(init_set, merged)
    return {
        "arm": {0: "never", 1: "every1", 4: "every4"}.get(reselect_every, str(reselect_every)),
        "reselect_every": reselect_every,
        "n_selections": len(traj),
        "MAE": mae, "RMSE": rmse, "RMSSE": rm,
        "seconds": time.perf_counter() - t0,
    }, traj


def main() -> None:
    df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    rows, trajs = [], []
    for cadence in (0, 4, 1):
        r, traj = run_arm(init_set, eval_set, cadence)
        rows.append(r)
        for t in traj:
            t["arm"] = r["arm"]
        trajs.extend(traj)
        print(f"{r['arm']:7s} selections={r['n_selections']:3d} MAE={r['MAE']:.4f} "
              f"RMSE={r['RMSE']:.4f} RMSSE={r['RMSSE']:.4f} ({r['seconds']:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "aistats2027" / "wf_reselection.csv", index=False)
    pd.DataFrame(trajs).to_csv(ROOT / "outputs" / "aistats2027" / "wf_reselection_trajectory.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
