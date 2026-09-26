"""W8: adaptive conformal inference (ACI) applied to EBB under the walk-forward
protocol, on all five panels.

Treatment mirrors ACI-ADIDA / ACI-TSB exactly (gamma = 0.05, per-series x
per-quantile adaptive levels, levels clipped to [1e-3, 0.999], one update per
revealed block summing gamma*(q - covered) over the block's observations). The
only difference is the base quantile function: instead of a residual pool at
the adaptive level, ACI-EBB reads EBB's analytic predictive quantile at that
level (evaluated on a fine level grid and interpolated per series).

Also re-emits plain EBB from the same state as a sanity row; it must reproduce
the released EBB walk-forward numbers.

Outputs (outputs/<date>/): wf_aci_ebb_<panel>.csv (long, w2 format),
  wf_aci_ebb_quantiles_<panel>.csv, w8_run_meta_<panel>.json
Usage: py scripts/integrity/w8_wf_aci_ebb.py <panel> [panel ...]
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
from experiments.run_prob import _coverage_summary, _qcols, _scaled_pinball_table
from models.mixture_pooling import mixture_group_labels
from models.tsb_hb import initialize_online_tsb_hb, predict_online_tsb_hb, update_online_tsb_hb
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
Q5 = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = _qcols(Q5)
GAMMA = 0.05
LEVEL_MIN, LEVEL_MAX = 1e-3, 0.999
GRID = np.round(np.concatenate([[0.001, 0.0025, 0.005], np.arange(0.01, 0.995, 0.005), [0.995, 0.9975, 0.999]]), 4)
CONFIG = {"online_retail": ("global", 0.95, 7), "m5": ("mixture", 0.95, 7),
          "auto": ("global", 0.99, 1), "carparts": ("global", 0.90, 1), "raf": ("mixture", 0.997, 1)}


def log(m):
    print(f"[w8] {m}", flush=True)


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


def interp_levels(grid_q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """grid_q: (n_rows, len(GRID)) quantiles at GRID levels; levels: (n_rows,) -> (n_rows,)."""
    out = np.empty(len(levels))
    for r in range(len(levels)):
        out[r] = np.interp(levels[r], GRID, grid_q[r])
    return out


def run(panel):
    t0 = time.perf_counter()
    structure, w, step = CONFIG[panel]
    init_set, eval_set = load_panel(panel)
    init_set = init_set.copy(); init_set["unique_id"] = init_set["unique_id"].astype(str)
    eval_set = eval_set.copy(); eval_set["unique_id"] = eval_set["unique_id"].astype(str)
    labels = mixture_group_labels(init_set, k=0, fit_discount=w).labels if structure == "mixture" else None
    if labels is not None:
        labels.index = labels.index.astype(str)
    state = initialize_online_tsb_hb(
        init_set, group_labels=labels, bootstrap_draws=0, bootstrap_seed=42,
        group_shrink_strength=0.0, dynamic_occurrence=False, occurrence_discount=1.0,
        item_variance_mode="conjugate", item_variance_shrink_strength=20.0, fit_discount=w)
    uids = sorted(init_set["unique_id"].unique())
    alpha = pd.DataFrame({f"a_{q}": q for q in Q5}, index=uids)
    frames_out, nb = [], 0
    t_pred = t_aci = t_upd = 0.0
    grid_cols = [f"q_{q}" for q in GRID]
    for frame in iter_walk_forward_frames(init_set, eval_set, step_size=step):
        nb += 1
        tgt = frame.target[["unique_id", "ds", "y"]].copy()
        tgt["unique_id"] = tgt["unique_id"].astype(str)
        tb = time.perf_counter()
        pred = predict_online_tsb_hb(state, frame.target, quantiles=list(GRID), n_samples=2000,
                                     include_hyper_uncertainty=False)
        pred["unique_id"] = pred["unique_id"].astype(str)
        pred = tgt[["unique_id", "ds"]].merge(pred, on=["unique_id", "ds"], how="left")
        gq = pred[grid_cols].to_numpy(float)
        gq = np.maximum.accumulate(np.nan_to_num(gq, nan=0.0), axis=1)   # monotone in level
        t_pred += time.perf_counter() - tb
        # plain EBB at nominal levels (sanity)
        plain = pred[["unique_id", "ds"]].copy()
        for q in Q5:
            plain[f"q_{q}"] = interp_levels(gq, np.full(len(pred), q))
        plain["model"] = "EBB"
        frames_out.append(plain[["model", "unique_id", "ds"] + QCOLS])
        # ACI-EBB at adaptive levels
        tb = time.perf_counter()
        a_now = alpha.reindex(pred["unique_id"].values)
        out = pred[["unique_id", "ds"]].copy()
        for q in Q5:
            lev = np.clip(a_now[f"a_{q}"].to_numpy(float), LEVEL_MIN, LEVEL_MAX)
            out[f"q_{q}"] = np.maximum(interp_levels(gq, lev), 0.0)
        arr = out[QCOLS].to_numpy(float)
        for j in range(1, arr.shape[1]):
            arr[:, j] = np.maximum(arr[:, j], arr[:, j - 1])
        out[QCOLS] = arr
        out["model"] = "ACI-EBB"
        frames_out.append(out[["model", "unique_id", "ds"] + QCOLS].copy())
        upd = out.merge(tgt, on=["unique_id", "ds"]).sort_values(["unique_id", "ds"])
        for q in Q5:
            covered = (upd["y"].to_numpy(float) <= upd[f"q_{q}"].to_numpy(float)).astype(float)
            g = pd.DataFrame({"uid": upd["unique_id"].to_numpy(), "delta": GAMMA * (q - covered)}).groupby("uid")["delta"].sum()
            alpha.loc[g.index, f"a_{q}"] = np.clip(alpha.loc[g.index, f"a_{q}"].to_numpy(float) + g.to_numpy(float),
                                                   LEVEL_MIN, LEVEL_MAX)
        t_aci += time.perf_counter() - tb
        tb = time.perf_counter()
        state = update_online_tsb_hb(state, frame.target)
        t_upd += time.perf_counter() - tb
        if nb % 10 == 0 or step == 1:
            log(f"{panel} block {nb} done ({(time.perf_counter()-t0)/60:.1f} min)")
    aq = pd.concat(frames_out, ignore_index=True)
    aq["ds"] = pd.to_datetime(aq["ds"])
    aq.to_csv(OUT / f"wf_aci_ebb_quantiles_{panel}.csv", index=False)
    ev = eval_set[["unique_id", "ds", "y"]].copy(); ev["ds"] = pd.to_datetime(ev["ds"])
    merged = ev.merge(aq, on=["unique_id", "ds"], how="inner")
    spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=Q5)
    cov = _coverage_summary(merged).set_index("model")
    cov_pos = _coverage_summary(merged[merged["y"] > 0].copy()).set_index("model")
    rows = []
    for m, g in spl.groupby("model"):
        for _, r in g.iterrows():
            rows.append({"dataset": panel, "protocol": "walk_forward", "model": m, "metric": "scaled_pinball",
                         "quantile": r["quantile"], "value": float(r["scaled_pinball"])})
        rows.append({"dataset": panel, "protocol": "walk_forward", "model": m, "metric": "scaled_pinball",
                     "quantile": "mean", "value": float(g["scaled_pinball"].mean())})
        cp = float(cov_pos.loc[m, "Coverage@80"]) if m in cov_pos.index else float("nan")
        wall = {"EBB": t_pred + t_upd, "ACI-EBB": t_pred + t_aci + t_upd}[m]
        for name, val in (("coverage80", cov.loc[m, "Coverage@80"]), ("coverage80_positive", cp),
                          ("aiw80", cov.loc[m, "AIW@80"]), ("wall_clock_s", wall)):
            rows.append({"dataset": panel, "protocol": "walk_forward", "model": m, "metric": name,
                         "quantile": "", "value": float(val)})
    pd.DataFrame(rows).to_csv(OUT / f"wf_aci_ebb_{panel}.csv", index=False)
    final_alpha = alpha.describe().loc[["mean", "50%", "min", "max"]].round(4).to_dict()
    meta = {"task": f"w8_wf_aci_ebb_{panel}", "panel": panel, "n_series": len(uids), "blocks": nb, "step": step,
            "ebb_config": [structure, w], "gamma": GAMMA, "level_clip": [LEVEL_MIN, LEVEL_MAX],
            "level_grid_size": int(len(GRID)), "final_alpha_summary": final_alpha,
            "walls_s": {"predict": round(t_pred, 1), "aci": round(t_aci, 1), "update": round(t_upd, 1)},
            "wall_clock_total_s": round(time.perf_counter() - t0, 1),
            "platform": platform.platform(), "python": sys.version.split()[0], "seed": 42}
    (OUT / f"w8_run_meta_{panel}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for m, g in spl.groupby("model"):
        log(f"{panel} {m}: mean SPL {float(g['scaled_pinball'].mean()):.4f}  q90 "
            f"{float(g.loc[np.isclose(g['quantile'].astype(float), 0.9), 'scaled_pinball'].iloc[0]):.4f}")


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["carparts", "auto", "raf", "online_retail", "m5"]):
        run(p)
