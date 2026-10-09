"""T5: does the selection outcome depend on how validation losses are aggregated?

The implementation (models.eb_hurdle._score_discount_grid) averages the scaled pinball loss over
all validation rows and quantiles (every validation observation has equal weight). The paper's
earlier Eq. (selection-risk) described an item-equal average (mean within item, then mean over
items). The two coincide when all items have validation tails of equal length (monthly panels)
and differ otherwise (Online Retail, M5). This script scores the 24 audited candidates (and the
8 `none` candidates) under BOTH aggregations on the same fits and reports the arg min of each.

Outputs (outputs/<date>/): selection_weighting_check.csv, selection_weighting_surfaces.csv
Usage: py scripts/integrity/t5_selection_weighting_check.py [panel ...]
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

from experiments.run_prob import _build_regime_group_labels
from models.mixture_pooling import mixture_group_labels
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
    print(f"[t5] {m}", flush=True)


def score(head, tail, labels, extra):
    scale = head.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s))))
    scale = scale.replace(0.0, np.nan)
    med = np.nanmedian(scale.to_numpy())
    gs = float(med) if np.isfinite(med) else 1.0
    scale = scale.fillna(max(gs, 1e-9))
    rows = []
    for w in DEFAULT_DISCOUNT_GRID:
        params = fit_eb_hurdle(head, group_labels=labels, item_variance_mode="conjugate",
                            item_variance_shrink_strength=20.0, fit_discount=float(w),
                            hyper_train_df=None, hyper_shrink="off", **extra)
        preds = predict_eb_hurdle(params, tail, quantiles=list(Q), include_hyper_uncertainty=False)
        m = tail[["unique_id", "ds", "y"]].merge(preds, on=["unique_id", "ds"], how="left")
        s = m["unique_id"].map(scale).fillna(max(gs, 1e-9)).to_numpy(dtype=float)
        loss = np.zeros(len(m))
        for q in Q:
            err = m["y"].to_numpy(dtype=float) - m[f"q_{q}"].to_numpy(dtype=float)
            loss += np.maximum(q * err, (q - 1.0) * err) / np.maximum(s, 1e-9)
        loss /= len(Q)
        pooled = float(np.nanmean(loss))
        item_equal = float(pd.Series(loss).groupby(m["unique_id"].to_numpy()).mean().mean())
        rows.append({"discount": float(w), "pooled": pooled, "item_equal": item_equal})
    return pd.DataFrame(rows)


def main():
    panels = sys.argv[1:] or PANELS
    audited = pd.read_csv(AUDITED)
    surf, summ = [], []
    for name in panels:
        t0 = time.time()
        init = t1.load_init(name)
        head, tail = _split_init_head_tail(init, val_ratio=0.2)
        lens = tail.groupby("unique_id").size()
        cands = {"global": (None, {}), "taxonomy": (_build_regime_group_labels(head), {}),
                 "mixture": (mixture_group_labels(head, k=0).labels, {}), "none": (None, NONE_KW)}
        frames = []
        for sname, (labels, extra) in cands.items():
            d = score(head, tail, labels, extra)
            d.insert(0, "structure", sname); d.insert(0, "panel", name)
            frames.append(d)
        full = pd.concat(frames, ignore_index=True)
        a = audited[audited["panel"] == name].merge(full, on=["panel", "structure", "discount"])
        chk = float((a["validation_spl"] - a["pooled"]).abs().max())
        surf.append(full)
        rec = {"panel": name, "tail_len_min": int(lens.min()), "tail_len_max": int(lens.max()),
               "pooled_check_maxdiff": chk}
        for fam, sub in (("24", full[full["structure"] != "none"]), ("32", full)):
            for agg in ("pooled", "item_equal"):
                r = sub.sort_values(agg).iloc[0]
                rec[f"{agg}_{fam}"] = f"({r['structure']}, {r['discount']:g})"
        summ.append(rec)
        log(f"{name}: tails {rec['tail_len_min']}-{rec['tail_len_max']} | check {chk:.1e} | "
            f"24: pooled {rec['pooled_24']} item-equal {rec['item_equal_24']} | "
            f"32: pooled {rec['pooled_32']} item-equal {rec['item_equal_32']}  [{time.time() - t0:.0f}s]")
        pd.concat(surf, ignore_index=True).to_csv(OUT / "selection_weighting_surfaces.csv", index=False)
        pd.DataFrame(summ).to_csv(OUT / "selection_weighting_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ).to_string(index=False))


if __name__ == "__main__":
    main()
