"""O1: two-rate forgetting (separate occurrence discount) gate evaluation, per
docs/DESIGN_occurrence_discount.md.

For each panel the audited (structure, w) is fixed. The occurrence discount
w_o is chosen on the initialization window's 80/20 chronological head/tail
split with the same criterion as the released discount selection (scaled
pinball at q in {.5,.75,.9}); 'tied' (w_o = w) is in the grid. External mean
SPL / q90 (B=20) is then reported for every grid value, and the share of
evaluation rows whose q90 forecast is exactly zero is recorded as the M5
diagnostic.

Outputs (outputs/<date>/): occdisc_selection.csv, occdisc_eval.csv, o1_run_meta.json
Usage: py scripts/integrity/o1_occdisc_eval.py [panel ...]
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
from experiments.run_prob import _enforce_monotonic_quantiles, _scaled_pinball_table
from models.mixture_pooling import mixture_group_labels
from models.tsb_hb import _split_init_head_tail, fit_tsb_hb, predict_tsb_hb
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
SEL_Q = [0.5, 0.75, 0.9]
EVAL_Q = [0.1, 0.25, 0.5, 0.75, 0.9]
GRID = ["tied", 0.999, 0.997, 0.995, 0.99, 0.98, 0.95, 0.90, 0.80, 0.70]
CONFIG = {"online_retail": ("global", 0.95), "m5": ("mixture", 0.95),
          "auto": ("global", 0.99), "carparts": ("global", 0.90), "raf": ("mixture", 0.997)}
TG_REF = {"online_retail": 0.8845, "m5": 1.6228, "auto": 0.2985, "carparts": 0.3214, "raf": 0.2763}


def log(m):
    print(f"[o1] {m}", flush=True)


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


def tail_score(head, tail, labels, w, wo):
    params = fit_tsb_hb(head, group_labels=labels, item_variance_mode="conjugate",
                        item_variance_shrink_strength=20.0, fit_discount=w, occurrence_fit_discount=wo)
    pred = predict_tsb_hb(params, tail, quantiles=SEL_Q, include_hyper_uncertainty=False)
    m = tail[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
    scale = head.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s)))).replace(0.0, np.nan)
    med = np.nanmedian(scale.to_numpy()); scale = scale.fillna(max(float(med), 1e-9))
    sc = m["unique_id"].map(scale).fillna(float(med)).to_numpy(float)
    losses = []
    for q in SEL_Q:
        err = m["y"].to_numpy(float) - m[f"q_{q}"].to_numpy(float)
        losses.append(np.maximum(q * err, (q - 1.0) * err) / np.maximum(sc, 1e-9))
    return float(np.nanmean(np.concatenate(losses)))


def external(init, ev, labels, w, wo):
    params = fit_tsb_hb(init, group_labels=labels, item_variance_mode="conjugate",
                        item_variance_shrink_strength=20.0, fit_discount=w, occurrence_fit_discount=wo,
                        bootstrap_draws=20, bootstrap_seed=42)
    pred = predict_tsb_hb(params, ev, quantiles=EVAL_Q, include_hyper_uncertainty=True)
    pred = _enforce_monotonic_quantiles(pred, quantiles=EVAL_Q)
    pred["model"] = "x"
    merged = ev[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="inner")
    spl = _scaled_pinball_table(merged, init_set=init, quantiles=EVAL_Q)
    spl = spl[spl["model"] == "x"]
    q90zero = float((merged["q_0.9"] <= 0.0).mean())
    return (float(spl["scaled_pinball"].mean()),
            float(spl.loc[np.isclose(spl["quantile"].astype(float), 0.9), "scaled_pinball"].iloc[0]), q90zero)


def main():
    panels = sys.argv[1:] or ["carparts", "auto", "raf", "online_retail", "m5"]
    t_all = time.perf_counter()
    sel_rows, eval_rows = [], []
    for panel in panels:
        t0 = time.perf_counter()
        init, ev = load_panel(panel)
        init = init.copy(); init["unique_id"] = init["unique_id"].astype(str)
        ev = ev.copy(); ev["unique_id"] = ev["unique_id"].astype(str)
        structure, w = CONFIG[panel]
        labels = mixture_group_labels(init, k=0, fit_discount=w).labels if structure == "mixture" else None
        if labels is not None:
            labels.index = labels.index.astype(str)
        head, tail = _split_init_head_tail(init, val_ratio=0.2)
        head_labels = None
        if structure == "mixture":
            head_labels = mixture_group_labels(head, k=0, fit_discount=w).labels
            head_labels.index = head_labels.index.astype(str)
        scores = {}
        for g in GRID:
            wo = None if g == "tied" else float(g)
            scores[g] = tail_score(head, tail, head_labels, w, wo)
            sel_rows.append({"panel": panel, "w": w, "w_occ": g, "tail_score": round(scores[g], 6)})
        best = min(scores, key=scores.get)
        log(f"{panel}: tail scores {{{', '.join(f'{k}:{v:.5f}' for k, v in scores.items())}}} -> w_occ*={best}")
        for g in GRID:
            wo = None if g == "tied" else float(g)
            m_, q_, z_ = external(init, ev, labels, w, wo)
            eval_rows.append({"panel": panel, "w": w, "w_occ": g, "selected": g == best,
                              "spl_mean": round(m_, 4), "spl_q90": round(q_, 4), "q90_zero_share": round(z_, 4),
                              "tweediegp": TG_REF[panel]})
            log(f"{panel} w_occ={g}: external mean {m_:.4f} q90 {q_:.4f} q90-zero {z_:.3f}")
        for name, rows in (("occdisc_selection.csv", sel_rows), ("occdisc_eval.csv", eval_rows)):
            new_df = pd.DataFrame(rows)
            f = OUT / name
            if f.exists():   # merge with panels written by earlier invocations
                old_df = pd.read_csv(f)
                old_df = old_df[~old_df["panel"].isin(new_df["panel"].unique())]
                new_df = pd.concat([old_df, new_df], ignore_index=True)
            new_df.to_csv(f, index=False)
        log(f"{panel} done in {(time.perf_counter()-t0)/60:.1f} min")
    meta = {"task": "o1_occdisc_eval", "design": "docs/DESIGN_occurrence_discount.md", "grid": [str(g) for g in GRID],
            "selection": "80/20 head/tail of the initialization window; scaled pinball q in {.5,.75,.9}; tied in grid",
            "scalar_path_hash_before_after": "1605a3ae... identical (outputs/2026-09-07/scalar_hash_{before,after}.txt)",
            "wall_clock_total_s": round(time.perf_counter() - t_all, 1), "platform": platform.platform(),
            "python": sys.version.split()[0], "seed": 42}
    (OUT / "o1_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
