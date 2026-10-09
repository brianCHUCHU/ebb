"""F5 + F7: unified rescoring of tab:prob and per-series SPL significance.

F7  Every model's stored quantile predictions for all five panels are fed
    through the current scoring implementation (identical per-series scale
    and aggregation), producing one internally consistent tab:prob.
F5  Per-series mean SPL paired tests, EBB vs every comparator including
    TweedieGP, paired t + Wilcoxon, Holm-corrected within panel.

Sources (predictions only; no model is rerun):
  EBB        outputs/integrity_2026-08-06/<panel>_prob_ebb_corrected
  AA/AT/CP-*   or_prob_fixed_paper / m5_prob_correct_split / <m>_prob_fixed
  iETS         <panel>_prob_iets / m5_prob_iets_tweedie_correct
  Tweedie-GLM  or_prob_tweedie / <m>_prob_fixed / m5_prob_iets_tweedie_correct
  TweedieGP    outputs/paper_rebuild/tweediegp_quantiles_<m>.csv
  Zero         synthesized (diagnostic; excluded from the test family)

Outputs (outputs/integrity_<date>/):
  rescored_tab_prob.csv, spl_significance_corrected.csv, f57_run_meta.json

Usage: py scripts/integrity/f57_significance_rescore.py
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
from scipy import stats

from experiments.run_prob import _series_naive_scale
from data_loading import (
    load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin,
    load_generic_long, train_eval_split_last_h, load_m5_long, preprocess_m5,
)
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

A = ROOT / "outputs" / "paper_runs"
B = ROOT / "outputs" / "paper_rebuild"
CORR = ROOT / "outputs" / "integrity_2026-08-06"
OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
OUT.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = [f"q_{q}" for q in QUANTILES]
SEED = 42
RENAME = {"CP-CrostonClassic": "CP-Croston", "CP-CrostonSBA": "CP-SBA",
          "Tweedie": "Tweedie-GLM"}

SOURCES = {
    "online_retail": [A / "or_prob_fixed_paper", A / "or_prob_iets", A / "or_prob_tweedie"],
    "m5": [B / "m5_prob_correct_split", B / "m5_prob_iets_tweedie_correct"],
    "auto": [A / "auto_prob_fixed", A / "auto_prob_iets"],
    "carparts": [A / "carparts_prob_fixed", A / "carparts_prob_iets"],
    "raf": [A / "raf_prob_fixed", A / "raf_prob_iets"],
}
EBB_DIR = {p: CORR / f"{p}_prob_ebb_corrected" for p in SOURCES}
TWEEDIEGP = {m: B / f"tweediegp_quantiles_{m}.csv" for m in ("auto", "carparts", "raf")}
TWEEDIEGP.update({m: ROOT / "outputs" / "2026-09-05" / f"tweediegp_quantiles_{m}.csv"
                  for m in ("online_retail", "m5")})


def load_panel(name: str):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    if name == "m5":
        set_seed(SEED)
        sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        df = preprocess_m5(sales, cal, sample_size=5000)
        return train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
    h = {"auto": 6, "carparts": 6, "raf": 12}[name]
    df = load_generic_long(ROOT / "data" / f"{name}_long.csv")
    return train_eval_split_last_h(df, h=h)


def holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx])
        adj[idx] = min(run, 1.0)
    return adj


def norm_frame(q: pd.DataFrame) -> pd.DataFrame:
    q = q.drop(columns=["protocol"], errors="ignore").copy()
    q["ds"] = pd.to_datetime(q["ds"])
    q["unique_id"] = q["unique_id"].astype(str)
    return q


def main() -> None:
    t0 = time.time()
    rescored_rows, sig_rows = [], []
    for panel, dirs in SOURCES.items():
        init_set, eval_set = load_panel(panel)
        scales = _series_naive_scale(init_set)
        ev = eval_set[["unique_id", "ds", "y"]].copy()
        ev["ds"] = pd.to_datetime(ev["ds"])
        ev["unique_id"] = ev["unique_id"].astype(str)

        frames = []
        # comparators (drop each source's own EB-Hurdle row)
        for d in dirs:
            q = norm_frame(pd.read_csv(d / "prob_quantiles.csv"))
            frames.append(q[q["model"] != "EB-Hurdle"])
        # EBB from the audited configuration
        q = norm_frame(pd.read_csv(EBB_DIR[panel] / "prob_quantiles.csv"))
        frames.append(q[q["model"] == "EB-Hurdle"])
        # TweedieGP (official implementation, monthly panels)
        if panel in TWEEDIEGP:
            q = norm_frame(pd.read_csv(TWEEDIEGP[panel]))
            q["model"] = "TweedieGP"
            frames.append(q[["model", "unique_id", "ds"] + QCOLS])
        # Zero diagnostic
        z = ev[["unique_id", "ds"]].copy()
        for c in QCOLS:
            z[c] = 0.0
        z["model"] = "Zero"
        frames.append(z)

        all_q = pd.concat(frames, ignore_index=True)
        all_q = all_q.drop_duplicates(subset=["model", "unique_id", "ds"], keep="first")
        merged = ev.merge(all_q[["model", "unique_id", "ds"] + QCOLS],
                          on=["unique_id", "ds"], how="inner")

        models = sorted(merged["model"].unique())
        per_series = {}
        for m in models:
            sub = merged[merged["model"] == m]
            per_q = {}
            pins = []
            for q_ in QUANTILES:
                err = sub["y"] - sub[f"q_{q_}"]
                pin = np.maximum(q_ * err, (q_ - 1) * err)
                byu = pin.groupby(sub["unique_id"]).mean()
                sc = byu.index.map(scales)
                ok = (sc > 0) & np.isfinite(sc)
                scaled = (byu[ok] / sc[ok]).astype(float)
                per_q[q_] = float(scaled.mean())
                pins.append(scaled)
            ps = pd.concat(pins, axis=1).mean(axis=1)
            per_series[m] = ps
            name = RENAME.get(m, m)
            rescored_rows.append({
                "panel": panel, "model": name,
                "spl_mean": round(float(np.mean(list(per_q.values()))), 4),
                "spl_q90": round(per_q[0.9], 4),
                "n_series": int(len(ps)),
            })
        print(f"[{panel}] rescored {len(models)} models", flush=True)

        ebb = per_series["EB-Hurdle"]
        comparators = [m for m in models if m not in ("EB-Hurdle", "Zero")]
        recs, t_ps, w_ps = [], [], []
        for m in comparators:
            joint = pd.concat([ebb.rename("ebb"), per_series[m].rename("cmp")],
                              axis=1).dropna()
            d = (joint["ebb"] - joint["cmp"]).to_numpy()
            t_stat, t_p = stats.ttest_1samp(d, 0.0)
            try:
                _, w_p = stats.wilcoxon(d)
            except ValueError:
                w_p = 1.0
            recs.append({"panel": panel, "comparator": RENAME.get(m, m),
                         "mean_diff": float(d.mean()), "n_series": int(len(d)),
                         "t_stat": float(t_stat), "t_pval_raw": float(t_p),
                         "wilcoxon_pval_raw": float(w_p)})
            t_ps.append(t_p)
            w_ps.append(w_p)
        t_adj, w_adj = holm(np.array(t_ps)), holm(np.array(w_ps))
        for r, tp, wp in zip(recs, t_adj, w_adj):
            r["t_pval_holm"] = float(tp)
            r["wilcoxon_pval_holm"] = float(wp)
            r["ebb_better"] = r["mean_diff"] < 0
            r["significant_5pct"] = bool(tp < 0.05 and wp < 0.05)
            sig_rows.append(r)

    resc = pd.DataFrame(rescored_rows)
    resc.to_csv(OUT / "rescored_tab_prob.csv", index=False)
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(OUT / "spl_significance_corrected.csv", index=False)
    print(resc.pivot_table(index="model", columns="panel", values="spl_mean").round(4).to_string())
    n_sig_better = int((sig["ebb_better"] & sig["significant_5pct"]).sum())
    n_sig_worse = int((~sig["ebb_better"] & sig["significant_5pct"]).sum())
    print(f"significance: {n_sig_better}/{len(sig)} EBB significantly better, "
          f"{n_sig_worse} significantly worse")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "f57_significance_rescore", "commit": commit,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "seed": SEED,
        "scoring": "current implementation; per-quantile per-series mean pinball "
                   "/ _series_naive_scale, mean over series, mean over quantiles",
        "zero_excluded_from_tests": True,
        "wall_clock_total_s": round(time.time() - t0, 1),
    }
    (OUT / "f57_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote rescored_tab_prob.csv, spl_significance_corrected.csv")


if __name__ == "__main__":
    main()
