"""T2: leakage-safe joint selection with a no-pooling candidate (docs/DESIGN_none_candidate.md).

Adds `none` x 8 discounts to the audited 24-candidate surface. Scoring replicates
models.eb_hurdle._score_discount_grid line by line (fit on the fitting head, analytic quantiles,
include_hyper_uncertainty=False, pooled scaled pinball over Q={.5,.75,.9} on the validation tail);
the only difference for `none` is the prior-strength scaling passed to fit_eb_hurdle. The global arm
is recomputed with the same replica and checked against the audited surface.

Outputs (outputs/<date>/): selection_surfaces_with_none.csv, selected_pairs_with_none.csv
Usage: py scripts/integrity/t2_selection_with_none.py [panel ...]
"""
from __future__ import annotations

import importlib.util
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from models.eb_hurdle import DEFAULT_DISCOUNT_GRID, _split_init_head_tail, fit_eb_hurdle, predict_eb_hurdle

spec = importlib.util.spec_from_file_location("t1", ROOT / "scripts" / "integrity" / "t1_leakage_safe_selection.py")
t1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t1)

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
AUDITED = ROOT / "outputs" / "integrity_2026-08-06" / "selection_surfaces.csv"
PANELS = ["carparts", "auto", "raf", "online_retail", "m5"]
Q = (0.5, 0.75, 0.9)
NONE_KW = dict(adaptive_prior_strength=True, prior_strength_min=1e-6, prior_strength_max=1e-6,
               prior_strength_power=1.0)


def log(m):
    print(f"[t2] {m}", flush=True)


def score_grid(head, tail, extra_kw):
    scale = head.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s))))
    scale = scale.replace(0.0, np.nan)
    med = np.nanmedian(scale.to_numpy())
    global_scale = float(med) if np.isfinite(med) else 1.0
    scale = scale.fillna(max(global_scale, 1e-9))
    rows = []
    for w in DEFAULT_DISCOUNT_GRID:
        params = fit_eb_hurdle(head, group_labels=None, item_variance_mode="conjugate",
                            item_variance_shrink_strength=20.0, fit_discount=float(w),
                            hyper_train_df=None, hyper_shrink="off", **extra_kw)
        preds = predict_eb_hurdle(params, tail, quantiles=list(Q), include_hyper_uncertainty=False)
        merged = tail[["unique_id", "ds", "y"]].merge(preds, on=["unique_id", "ds"], how="left")
        s = merged["unique_id"].map(scale).fillna(max(global_scale, 1e-9)).to_numpy(dtype=float)
        losses = []
        for q in Q:
            err = merged["y"].to_numpy(dtype=float) - merged[f"q_{q}"].to_numpy(dtype=float)
            losses.append(np.maximum(q * err, (q - 1.0) * err) / np.maximum(s, 1e-9))
        rows.append({"discount": float(w), "validation_spl": float(np.nanmean(np.concatenate(losses)))})
    return pd.DataFrame(rows)


def main():
    panels = sys.argv[1:] or PANELS
    audited = pd.read_csv(AUDITED)
    surf_rows, pair_rows = [], []
    for name in panels:
        t0 = time.time()
        init = t1.load_init(name)
        head, tail = _split_init_head_tail(init, val_ratio=0.2)
        n_items = head["unique_id"].nunique()
        pos = head[head["y"] > 0].groupby("unique_id").size()
        share_nopos = 1.0 - len(pos) / n_items
        log(f"{name}: head rows={len(head)} tail rows={len(tail)} items={n_items} "
            f"share with no positive in head={share_nopos:.3f}")
        g = score_grid(head, tail, {})
        a = audited[(audited["panel"] == name) & (audited["structure"] == "global")].sort_values("discount")
        chk = float(np.max(np.abs(g.sort_values("discount")["validation_spl"].to_numpy()
                                  - a["validation_spl"].to_numpy())))
        log(f"{name}: global arm vs audited surface max |diff| = {chk:.2e}")
        n = score_grid(head, tail, NONE_KW)
        n.insert(0, "structure", "none"); n.insert(0, "panel", name)
        full = pd.concat([audited[audited["panel"] == name], n], ignore_index=True)
        full = full.sort_values("validation_spl").reset_index(drop=True)
        full["rank"] = np.arange(1, len(full) + 1)
        surf_rows.append(full)
        win = full.iloc[0]
        bn = full[full["structure"] == "none"].iloc[0]
        old = audited[audited["panel"] == name].sort_values("validation_spl").iloc[0]
        pair_rows.append({
            "panel": name, "selected_structure": win["structure"], "selected_discount": win["discount"],
            "selected_spl": round(float(win["validation_spl"]), 6),
            "audited_structure": old["structure"], "audited_discount": old["discount"],
            "audited_spl": round(float(old["validation_spl"]), 6),
            "best_none_discount": bn["discount"], "best_none_spl": round(float(bn["validation_spl"]), 6),
            "best_none_rank_of_32": int(bn["rank"]),
            "none_vs_audited_pct": round((float(bn["validation_spl"]) / float(old["validation_spl"]) - 1) * 100, 3),
            "global_check_maxdiff": chk, "share_items_no_positive_in_head": round(share_nopos, 4),
            "wall_s": round(time.time() - t0, 1),
        })
        log(f"{name}: selected ({win['structure']}, {win['discount']}) | audited ({old['structure']}, "
            f"{old['discount']}) | best none w={bn['discount']} rank {int(bn['rank'])}/32, "
            f"{pair_rows[-1]['none_vs_audited_pct']:+.3f}% vs audited  [{time.time() - t0:.0f}s]")
        pd.concat(surf_rows, ignore_index=True).to_csv(OUT / "selection_surfaces_with_none.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(OUT / "selected_pairs_with_none.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(pair_rows).to_string(index=False))


if __name__ == "__main__":
    main()
