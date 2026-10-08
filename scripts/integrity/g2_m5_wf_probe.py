"""Gate 2: cost probe for an M5 strict walk-forward evaluation. NO full run.

Protocol being costed (mirror of the Online Retail study): seven-day blocks
over the 647-day external span (~93 blocks), 5,000 series; CP wrappers refit
each block, AutoARIMA/AutoTheta/iETS refit every block (full) or every 4
(option a), EBB updates sufficient statistics each block.

Measurements:
  - EBB online: initialization + predict/update on the first 3 blocks
  - CP wrappers: full-panel refit on the first 3 blocks
  - AutoARIMA/AutoTheta: one refit on a deterministic 200-series subset,
    scaled linearly to 5,000
  - iETS: one refit on a deterministic 100-series subset (R, 20 jobs,
    10 s/series cap), scaled linearly

Output: outputs/<date>/m5_wf_cost_probe.csv + printed extrapolations for
full, option (a) refit-every-4, option (b) 1,000-series subsample.

Usage: py scripts/integrity/g2_m5_wf_probe.py [--skip-iets]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_m5_long, preprocess_m5, train_eval_split_fixed_origin
from experiments.protocols import iter_walk_forward_frames
from experiments.run_prob import _predict_non_hb_prob_models_once
from models.conformal import fit_predict_conformal_baselines
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import (
    initialize_online_eb_hurdle, predict_online_eb_hurdle, update_online_eb_hurdle,
)
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
IETS_RSCRIPT = r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe"
N_BLOCK_PROBE = 3
SUB_AA = 200
SUB_IETS = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-iets", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    df = preprocess_m5(sales, cal, sample_size=5000)
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
    n_series = init_set["unique_id"].nunique()
    span = int(eval_set.groupby("unique_id").size().max())
    n_blocks = int(np.ceil(span / 7))
    print(f"n_series={n_series} eval span={span}d -> {n_blocks} seven-day blocks", flush=True)

    frames = list(iter_walk_forward_frames(init_set, eval_set, step_size=7))[:N_BLOCK_PROBE]
    rows = []

    # --- EBB online ---
    t0 = time.perf_counter()
    labels = mixture_group_labels(init_set, k=0, fit_discount=0.95).labels
    state = initialize_online_eb_hurdle(
        init_set, group_labels=labels, bootstrap_draws=0, bootstrap_seed=42,
        group_shrink_strength=0.0, dynamic_occurrence=False, occurrence_discount=1.0,
        item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
        fit_discount=0.95)
    init_s = time.perf_counter() - t0
    per_block = []
    for fr in frames:
        tb = time.perf_counter()
        predict_online_eb_hurdle(state, fr.target, quantiles=QUANTILES,
                              n_samples=2000, include_hyper_uncertainty=False)
        state = update_online_eb_hurdle(state, fr.target)
        per_block.append(time.perf_counter() - tb)
    rows.append({"component": "EBB_init", "measured_s": round(init_s, 1),
                 "unit": "once", "n_measured": 1})
    rows.append({"component": "EBB_per_block", "measured_s": round(float(np.mean(per_block)), 1),
                 "unit": "per block (full panel)", "n_measured": len(per_block)})
    print(f"EBB: init {init_s:.0f}s, per-block {np.mean(per_block):.1f}s", flush=True)

    # --- CP wrappers (all five in one call) ---
    cp = []
    for fr in frames:
        tb = time.perf_counter()
        fit_predict_conformal_baselines(fr.history, fr.target,
                                        quantiles=QUANTILES, cal_ratio=0.2, freq="D")
        cp.append(time.perf_counter() - tb)
    rows.append({"component": "CP5_per_block", "measured_s": round(float(np.mean(cp)), 1),
                 "unit": "per block (full panel, all five wrappers)", "n_measured": len(cp)})
    print(f"CP wrappers: per-block {np.mean(cp):.1f}s", flush=True)

    # --- AutoARIMA/AutoTheta: one refit on a subset, linear scaling ---
    uids = np.array(sorted(init_set["unique_id"].astype(str).unique()))
    sub = uids[np.sort(np.random.default_rng(42).choice(len(uids), SUB_AA, replace=False))]
    h1 = frames[0]
    hist = h1.history[h1.history["unique_id"].astype(str).isin(sub)]
    targ = h1.target[h1.target["unique_id"].astype(str).isin(sub)]
    tb = time.perf_counter()
    _predict_non_hb_prob_models_once(hist, targ, quantiles=QUANTILES,
                                     baseline_mode="paper", freq="D", season_length=7)
    aa_s = time.perf_counter() - tb
    aa_full = aa_s * n_series / SUB_AA
    rows.append({"component": "AA+AT_per_refit_sub200", "measured_s": round(aa_s, 1),
                 "unit": f"per refit ({SUB_AA} series)", "n_measured": 1})
    rows.append({"component": "AA+AT_per_refit_full_est", "measured_s": round(aa_full, 1),
                 "unit": "per refit (5000 series, linear extrapolation)", "n_measured": 0})
    print(f"AA+AT: {aa_s:.0f}s / {SUB_AA} series -> ~{aa_full:.0f}s per full refit", flush=True)

    # --- iETS: one refit on a smaller subset ---
    iets_full = np.nan
    if not args.skip_iets:
        from models.iets_baseline import fit_predict_iets_prob_panel
        sub2 = uids[np.sort(np.random.default_rng(43).choice(len(uids), SUB_IETS, replace=False))]
        hist2 = h1.history[h1.history["unique_id"].astype(str).isin(sub2)]
        targ2 = h1.target[h1.target["unique_id"].astype(str).isin(sub2)]
        tb = time.perf_counter()
        try:
            fit_predict_iets_prob_panel(hist2, targ2, quantiles=QUANTILES,
                                        rscript=IETS_RSCRIPT, occurrence="auto",
                                        timeout_seconds=None,
                                        per_series_timeout_seconds=10,
                                        n_jobs=20, cache_path=None)
            iets_s = time.perf_counter() - tb
            iets_full = iets_s * n_series / SUB_IETS
            rows.append({"component": "iETS_per_refit_sub100", "measured_s": round(iets_s, 1),
                         "unit": f"per refit ({SUB_IETS} series, 20 jobs)", "n_measured": 1})
            rows.append({"component": "iETS_per_refit_full_est", "measured_s": round(iets_full, 1),
                         "unit": "per refit (5000 series, linear extrapolation)",
                         "n_measured": 0})
            print(f"iETS: {iets_s:.0f}s / {SUB_IETS} series -> ~{iets_full:.0f}s per full refit",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({"component": "iETS_FAILED", "measured_s": np.nan,
                         "unit": repr(e)[:120], "n_measured": 0})
            print(f"iETS probe failed: {e!r}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "m5_wf_cost_probe.csv", index=False)

    # --- extrapolations ---
    ebb_total = init_s + np.mean(per_block) * n_blocks
    cp_total = np.mean(cp) * n_blocks
    aa_every1 = aa_full * n_blocks
    aa_every4 = aa_full * int(np.ceil(n_blocks / 4))
    iets_every1 = iets_full * n_blocks if np.isfinite(iets_full) else np.nan
    iets_every4 = iets_full * int(np.ceil(n_blocks / 4)) if np.isfinite(iets_full) else np.nan

    def hrs(x):
        return "n/a" if not np.isfinite(x) else f"{x/3600:.1f}h"

    print("\n=== extrapolation (5000 series, %d blocks) ===" % n_blocks)
    print(f"EBB total: {hrs(ebb_total)}   CP wrappers total: {hrs(cp_total)}")
    print(f"AA+AT: every block {hrs(aa_every1)} | every 4 {hrs(aa_every4)}")
    print(f"iETS:  every block {hrs(iets_every1)} | every 4 {hrs(iets_every4)}")
    full = ebb_total + cp_total + aa_every1 + (iets_every1 if np.isfinite(iets_every1) else 0)
    opta = ebb_total + cp_total + aa_every4 + (iets_every4 if np.isfinite(iets_every4) else 0)
    print(f"FULL (all refit each block): {hrs(full)}")
    print(f"OPTION (a) AA/AT/iETS every 4: {hrs(opta)}")
    print(f"OPTION (b) 1000-series subsample, refit-every-4: {hrs(opta/5)} (linear)")
    (OUT / "m5_wf_cost_probe_summary.txt").write_text(
        f"blocks={n_blocks}\nfull={full:.0f}s\noption_a={opta:.0f}s\n"
        f"option_b={opta/5:.0f}s\n", encoding="utf-8")


if __name__ == "__main__":
    main()
