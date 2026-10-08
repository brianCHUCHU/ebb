"""W9: per-series paired significance tests under the walk-forward protocol.

Same construction as the fixed-origin family (f57): per-series mean scaled
pinball over q in {.1,.25,.5,.75,.9}, scale = naive in-sample scale from the
initialization window; paired t and Wilcoxon signed-rank on the per-series
difference (reference - comparator); Holm within each (panel, reference)
family; "significant" requires both Holm-adjusted p < 0.05.

References tested: EBB (released online rows) and ACI-EBB (W8).
Comparators: every walk-forward method with stored per-series quantiles.

Outputs (outputs/<date>/): spl_significance_wf.csv, wf_rescored_check.csv,
  w9_run_meta.json
Usage: py scripts/integrity/w9_wf_significance.py
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
from scipy import stats

from data_loading import (load_generic_long, load_m5_long, load_online_retail, preprocess_m5,
                          preprocess_online_retail, train_eval_split_fixed_origin,
                          train_eval_split_last_h)
from experiments.run_prob import _series_naive_scale
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
RB = ROOT / "outputs" / "paper_rebuild"
I0807 = ROOT / "outputs" / "integrity_2026-08-07"
D0905 = ROOT / "outputs" / "2026-09-05"
D0906 = ROOT / "outputs" / "2026-09-06"
Q5 = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = [f"q_{q}" for q in Q5]
RENAME = {"EB-Hurdle": "EBB", "CP-CrostonClassic": "CP-Croston", "CP-CrostonSBA": "CP-SBA"}
PANELS = ["online_retail", "carparts", "auto", "raf", "m5"]


def latest(name: str) -> Path | None:
    for d in sorted((ROOT / "outputs").glob("2026-09-*"), reverse=True):
        if (d / name).exists():
            return d / name
    return None


# (path, models to keep or None for all, models to drop)
SOURCES = {
    "online_retail": [(RB / "t1_quantiles_new.csv", None, ["EB-Hurdle", "Zero"]),
                      (RB / "t5_quantiles.csv", None, []),
                      (I0807 / "wf_ebb_quantiles_online_retail.csv", ["EB-Hurdle"], []),
                      (D0906 / "wf_tweediegp_quantiles_online_retail.csv", None, [])],
    "carparts": [(RB / "t2_quantiles_carparts.csv", None, ["EB-Hurdle", "Zero"]),
                 (D0905 / "wf_roster_quantiles_carparts_aci.csv", ["ACI-ADIDA"], []),
                 (I0807 / "wf_ebb_quantiles_carparts.csv", ["EB-Hurdle"], []),
                 (D0906 / "wf_tweediegp_quantiles_carparts.csv", None, [])],
    "auto": [(D0905 / "wf_roster_quantiles_auto.csv", None, ["Zero"]),
             (D0906 / "wf_tweediegp_quantiles_auto.csv", None, [])],
    "raf": [(D0905 / "wf_roster_quantiles_raf.csv", None, ["Zero"]),
            (D0906 / "wf_tweediegp_quantiles_raf.csv", None, [])],
    "m5": [(D0905 / "m5_wf_quantiles.csv", None, ["Zero"])],
}


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


def holm(p):
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(run, 1.0)
    return adj


def read_q(path, keep, drop):
    q = pd.read_csv(path, low_memory=False)
    if "model" not in q.columns:
        q["model"] = "TweedieGP"
    q = q[["model", "unique_id", "ds"] + QCOLS].copy()
    q["ds"] = pd.to_datetime(q["ds"]); q["unique_id"] = q["unique_id"].astype(str)
    if keep is not None:
        q = q[q["model"].isin(keep)]
    if drop:
        q = q[~q["model"].isin(drop)]
    q["model"] = q["model"].map(lambda m: RENAME.get(m, m))
    return q


def main():
    t0 = time.time()
    sig_rows, check_rows, notes = [], [], []
    for panel in PANELS:
        init_set, eval_set = load_panel(panel)
        scales = _series_naive_scale(init_set)
        ev = eval_set[["unique_id", "ds", "y"]].copy()
        ev["ds"] = pd.to_datetime(ev["ds"]); ev["unique_id"] = ev["unique_id"].astype(str)
        frames = [read_q(p, k, d) for p, k, d in SOURCES[panel] if p.exists()]
        w8 = latest(f"wf_aci_ebb_quantiles_{panel}.csv")
        if w8 is not None:
            frames.append(read_q(w8, ["ACI-EBB"], []))
        else:
            notes.append(f"{panel}: no ACI-EBB quantiles yet")
        all_q = pd.concat(frames, ignore_index=True).drop_duplicates(["model", "unique_id", "ds"], keep="first")
        merged = ev.merge(all_q, on=["unique_id", "ds"], how="inner")
        models = sorted(merged["model"].unique())
        per_series = {}
        for m in models:
            sub = merged[merged["model"] == m]
            pins = []
            for q_ in Q5:
                err = sub["y"] - sub[f"q_{q_}"]
                pin = np.maximum(q_ * err, (q_ - 1) * err)
                byu = pin.groupby(sub["unique_id"]).mean()
                sc = byu.index.map(scales)
                ok = (sc > 0) & np.isfinite(sc)
                pins.append((byu[ok] / sc[ok]).astype(float))
            ps = pd.concat(pins, axis=1).mean(axis=1)
            per_series[m] = ps
            check_rows.append({"panel": panel, "model": m, "spl_mean_rescored": round(float(np.mean([p.mean() for p in pins])), 4),
                               "n_series": int(len(ps))})
        print(f"[{panel}] models: {models}", flush=True)
        for ref in ("EBB", "ACI-EBB"):
            if ref not in per_series:
                continue
            comps = [m for m in models if m not in (ref, "Zero")]
            recs, tp, wp = [], [], []
            for m in comps:
                joint = pd.concat([per_series[ref].rename("ref"), per_series[m].rename("cmp")], axis=1).dropna()
                d = (joint["ref"] - joint["cmp"]).to_numpy()
                t_stat, t_p = stats.ttest_1samp(d, 0.0)
                try:
                    _, w_p = stats.wilcoxon(d)
                except ValueError:
                    w_p = 1.0
                recs.append({"panel": panel, "reference": ref, "comparator": m, "mean_diff": float(d.mean()),
                             "n_series": int(len(d)), "t_stat": float(t_stat), "t_pval_raw": float(t_p),
                             "wilcoxon_pval_raw": float(w_p)})
                tp.append(t_p); wp.append(w_p)
            for r, a, b in zip(recs, holm(np.array(tp)), holm(np.array(wp))):
                r["t_pval_holm"] = float(a); r["wilcoxon_pval_holm"] = float(b)
                r["ref_better"] = r["mean_diff"] < 0
                r["significant_5pct"] = bool(a < 0.05 and b < 0.05)
                sig_rows.append(r)
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(OUT / "spl_significance_wf.csv", index=False)
    pd.DataFrame(check_rows).to_csv(OUT / "wf_rescored_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(sig[["panel", "reference", "comparator", "mean_diff", "t_pval_holm", "wilcoxon_pval_holm", "ref_better", "significant_5pct"]].round(4).to_string())
    meta = {"task": "w9_wf_significance", "families": "per (panel, reference); Holm; significant = both adjusted p<0.05",
            "notes": notes, "wall_clock_total_s": round(time.time() - t0, 1),
            "platform": platform.platform(), "python": sys.version.split()[0], "seed": 42}
    (OUT / "w9_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
