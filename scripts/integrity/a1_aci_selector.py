"""A1: leakage-safe rule for WHEN to apply the adaptive conformal layer to EBB.

Rule (fixed before running): split the initialization window 80/20
chronologically per series (same split as the discount selection). Initialize
EBB on the head at the audited (structure, w) with mixture labels learned on
the head only, then run a small walk-forward over the tail (same block step as
the external protocol) with the plain predictive quantiles and with the ACI
layer (gamma 0.05, same update as W8). Score both on the tail with scaled
pinball at q in {.5,.75,.9} (head naive scale). Apply ACI in deployment iff
its tail score is lower. External walk-forward numbers for both variants
already exist (W3 assembly); this script only records the decision and the
resulting "selected" row.

Outputs (outputs/<date>/): aci_selector.csv, a1_run_meta.json
Usage: py scripts/integrity/a1_aci_selector.py [panel ...]
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import (load_generic_long, load_m5_long, load_online_retail, preprocess_m5,
                          preprocess_online_retail, train_eval_split_fixed_origin,
                          train_eval_split_last_h)
from experiments.protocols import iter_walk_forward_frames
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import (_split_init_head_tail, initialize_online_eb_hurdle, predict_online_eb_hurdle,
                           update_online_eb_hurdle)
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
Q5 = [0.1, 0.25, 0.5, 0.75, 0.9]
SEL_Q = [0.5, 0.75, 0.9]
GAMMA, LEVEL_MIN, LEVEL_MAX = 0.05, 1e-3, 0.999
GRID = np.round(np.concatenate([[0.001, 0.0025, 0.005], np.arange(0.01, 0.995, 0.005), [0.995, 0.9975, 0.999]]), 4)
CONFIG = {"online_retail": ("global", 0.95, 7), "m5": ("mixture", 0.95, 7),
          "auto": ("global", 0.99, 1), "carparts": ("global", 0.90, 1), "raf": ("mixture", 0.997, 1)}
PANELS = ["carparts", "auto", "raf", "online_retail", "m5"]


def log(m):
    print(f"[a1] {m}", flush=True)


def load_panel(name):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data/online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    if name == "m5":
        set_seed(42)
        s, c = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        return train_eval_split_fixed_origin(preprocess_m5(s, c, sample_size=5000), init_ratio=2 / 3, min_len=1)
    h = {"auto": 6, "carparts": 6, "raf": 12}[name]
    return train_eval_split_last_h(load_generic_long(ROOT / f"data/{name}_long.csv"), h=h)


def interp_levels(gq, levels):
    return np.array([np.interp(levels[r], GRID, gq[r]) for r in range(len(levels))])


def tail_walk_forward(head, tail, structure, w, step):
    """-> (plain_score, aci_score) on the tail, scaled pinball over SEL_Q."""
    labels = mixture_group_labels(head, k=0, fit_discount=w).labels if structure == "mixture" else None
    if labels is not None:
        labels.index = labels.index.astype(str)
    state = initialize_online_eb_hurdle(
        head, group_labels=labels, bootstrap_draws=0, bootstrap_seed=42, group_shrink_strength=0.0,
        dynamic_occurrence=False, occurrence_discount=1.0, item_variance_mode="conjugate",
        item_variance_shrink_strength=20.0, fit_discount=w)
    uids = sorted(head["unique_id"].unique())
    alpha = pd.DataFrame({f"a_{q}": q for q in Q5}, index=uids)
    scale = head.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s)))).replace(0.0, np.nan)
    med = np.nanmedian(scale.to_numpy()); scale = scale.fillna(max(float(med), 1e-9))
    grid_cols = [f"q_{q}" for q in GRID]
    loss_plain, loss_aci = [], []
    for frame in iter_walk_forward_frames(head, tail, step_size=step):
        tgt = frame.target[["unique_id", "ds", "y"]].copy(); tgt["unique_id"] = tgt["unique_id"].astype(str)
        pred = predict_online_eb_hurdle(state, frame.target, quantiles=list(GRID), n_samples=2000,
                                     include_hyper_uncertainty=False)
        pred["unique_id"] = pred["unique_id"].astype(str)
        pred = tgt[["unique_id", "ds"]].merge(pred, on=["unique_id", "ds"], how="left")
        gq = np.maximum.accumulate(np.nan_to_num(pred[grid_cols].to_numpy(float), nan=0.0), axis=1)
        y = tgt["y"].to_numpy(float)
        sc = tgt["unique_id"].map(scale).fillna(float(med)).to_numpy(float)
        a_now = alpha.reindex(pred["unique_id"].values)
        q_aci = {}
        for q in Q5:
            lev = np.clip(a_now[f"a_{q}"].to_numpy(float), LEVEL_MIN, LEVEL_MAX)
            q_aci[q] = np.maximum(interp_levels(gq, lev), 0.0)
        for q in SEL_Q:
            qp = interp_levels(gq, np.full(len(pred), q))
            e = y - qp; loss_plain.append(np.maximum(q * e, (q - 1) * e) / np.maximum(sc, 1e-9))
            e = y - q_aci[q]; loss_aci.append(np.maximum(q * e, (q - 1) * e) / np.maximum(sc, 1e-9))
        upd = pd.DataFrame({"uid": pred["unique_id"].to_numpy()})
        for q in Q5:
            covered = (y <= q_aci[q]).astype(float)
            g = pd.DataFrame({"uid": upd["uid"], "delta": GAMMA * (q - covered)}).groupby("uid")["delta"].sum()
            alpha.loc[g.index, f"a_{q}"] = np.clip(alpha.loc[g.index, f"a_{q}"].to_numpy(float) + g.to_numpy(float),
                                                   LEVEL_MIN, LEVEL_MAX)
        state = update_online_eb_hurdle(state, frame.target)
    return float(np.nanmean(np.concatenate(loss_plain))), float(np.nanmean(np.concatenate(loss_aci)))


def external_rows(panel):
    """external walk-forward mean/q90 for EBB and ACI-EBB from the latest W3 assembly."""
    for d in sorted((ROOT / "outputs").glob("2026-09-*"), reverse=True):
        f = d / "wf_all_panels_wide.csv"
        if f.exists():
            w = pd.read_csv(f).set_index(["panel", "model"])
            if (panel, "ACI-EBB") in w.index:
                return {m: (float(w.loc[(panel, m), "mean"]), float(w.loc[(panel, m), "0.9"])) for m in ("EBB", "ACI-EBB")}
    return None


def main():
    panels = sys.argv[1:] or PANELS
    t_all = time.perf_counter(); rows = []
    prev = OUT / "aci_selector.csv"
    if prev.exists():
        rows = pd.read_csv(prev).to_dict("records")
        rows = [r for r in rows if r["panel"] not in panels]
    for panel in panels:
        t0 = time.perf_counter()
        init, _ev = load_panel(panel)
        init = init.copy(); init["unique_id"] = init["unique_id"].astype(str)
        structure, w, step = CONFIG[panel]
        head, tail = _split_init_head_tail(init, val_ratio=0.2)
        plain, aci = tail_walk_forward(head, tail, structure, w, step)
        choose = aci < plain
        ext = external_rows(panel) or {"EBB": (np.nan, np.nan), "ACI-EBB": (np.nan, np.nan)}
        sel = ext["ACI-EBB"] if choose else ext["EBB"]
        rows.append({"panel": panel, "tail_blocks": int(np.ceil(tail.groupby("unique_id").size().max() / step)),
                     "tail_plain": round(plain, 5), "tail_aci": round(aci, 5),
                     "tail_gain_pct": round(100 * (plain - aci) / plain, 2), "apply_aci": bool(choose),
                     "ext_EBB_mean": ext["EBB"][0], "ext_ACI_mean": ext["ACI-EBB"][0],
                     "ext_selected_mean": sel[0], "ext_selected_q90": sel[1],
                     "ext_best_of_two": min(ext["EBB"][0], ext["ACI-EBB"][0]),
                     "correct": bool(np.isclose(sel[0], min(ext["EBB"][0], ext["ACI-EBB"][0])))})
        log(f"{panel}: tail plain {plain:.5f} aci {aci:.5f} -> apply ACI = {choose}; external EBB {ext['EBB'][0]:.4f} "
            f"ACI-EBB {ext['ACI-EBB'][0]:.4f} -> selected {sel[0]:.4f} ({(time.perf_counter()-t0)/60:.1f} min)")
        pd.DataFrame(rows).to_csv(OUT / "aci_selector.csv", index=False)
    meta = {"task": "a1_aci_selector", "rule": "apply ACI iff tail scaled pinball (q .5/.75/.9) of ACI-EBB < plain EBB on the 80/20 chronological split of the initialization window; head-only labels; same gamma/update as W8",
            "wall_clock_total_s": round(time.perf_counter() - t_all, 1), "platform": platform.platform(),
            "python": sys.version.split()[0], "seed": 42}
    (OUT / "a1_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
