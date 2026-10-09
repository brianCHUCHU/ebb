"""Spec Task 2: strict walk-forward on Carparts (monthly, 2,509 series).

Same mechanics as the Online Retail walk-forward: EBB selects (g, w) once
on the initialization window, then updates online sufficient statistics per
block; every other method refits per block (6 monthly blocks, so no refit
compromise is needed: refit_interval=1). Methods: EBB + 5 CP wrappers +
Zero + AutoARIMA + AutoTheta + iETS. Scoring via the existing run_prob
functions (identical scaling + quantile post-processing).

Output: outputs/paper_rebuild/wf_carparts.csv (+ t2_run_meta.json,
        t2_quantiles_carparts.csv audit trail)

Usage: py scripts/rebuild/t2_wf_carparts.py [--skip-iets]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_generic_long, train_eval_split_last_h
from experiments.protocols import iter_walk_forward_frames
from experiments.run_point import _build_regime_group_labels
from experiments.run_prob import (
    _coverage_summary,
    _predict_non_hb_prob_models_once,
    _qcols,
    _scaled_pinball_table,
)
from models.conformal import fit_predict_conformal_baselines
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import (
    initialize_online_eb_hurdle,
    predict_online_eb_hurdle,
    select_pooling_and_discount,
    update_online_eb_hurdle,
)

OUT = ROOT / "outputs" / "paper_rebuild"
OUT.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = _qcols(QUANTILES)
SEED = 42
H = 6
IETS_RSCRIPT = r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe"


def log(msg: str) -> None:
    print(f"[t2] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-iets", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()

    df = load_generic_long(ROOT / "data" / "carparts_long.csv")
    init_set, eval_set = train_eval_split_last_h(df, h=H)
    n_series = int(init_set.unique_id.nunique())
    n_blocks = int(eval_set.groupby("unique_id").size().max())
    log(f"carparts: {n_series} series, eval span {n_blocks} months")

    # --- EBB: joint selection once on the initialization window ----------
    walls: dict[str, float] = {}
    t0 = time.perf_counter()
    taxonomy_labels = _build_regime_group_labels(init_set)
    mix = mixture_group_labels(init_set, k=0)
    candidates = {"global": None, "taxonomy": taxonomy_labels, "mixture": mix.labels}
    sel_name, sel_w, _ = select_pooling_and_discount(
        init_set, candidates,
        item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
    )
    log(f"EBB selection: structure={sel_name} (K={mix.k}), w={sel_w}")
    hb_state = initialize_online_eb_hurdle(
        init_set,
        group_labels=candidates[sel_name],
        bootstrap_draws=0, bootstrap_seed=SEED,
        group_shrink_strength=0.0,
        dynamic_occurrence=False, occurrence_discount=1.0,
        item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
        fit_discount=sel_w,
    )

    frames: list[pd.DataFrame] = []
    aa_wall = cp_wall = 0.0
    for frame in iter_walk_forward_frames(init_set, eval_set, step_size=1):
        tb = time.perf_counter()
        hbq = predict_online_eb_hurdle(
            hb_state, frame.target, quantiles=QUANTILES,
            n_samples=2000, include_hyper_uncertainty=False,
        )
        for c in QCOLS:
            if c not in hbq.columns:
                hbq[c] = np.nan
        hbq["model"] = "EB-Hurdle"
        frames.append(hbq[["model", "unique_id", "ds"] + QCOLS])
        walls["EB-Hurdle"] = walls.get("EB-Hurdle", 0.0) + time.perf_counter() - tb

        tb = time.perf_counter()
        nq = _predict_non_hb_prob_models_once(
            frame.history, frame.target, quantiles=QUANTILES,
            baseline_mode="paper", freq="MS", season_length=12)
        frames.append(nq)
        aa_wall += time.perf_counter() - tb

        tb = time.perf_counter()
        cq = fit_predict_conformal_baselines(
            frame.history, frame.target, quantiles=QUANTILES,
            cal_ratio=0.2, freq="MS")
        if not cq.empty:
            frames.append(cq)
        cp_wall += time.perf_counter() - tb

        hb_state = update_online_eb_hurdle(hb_state, frame.target)
        log(f"block step {frame.step}: done")

    walls["AutoARIMA"] = walls["AutoTheta"] = aa_wall / 2
    for m in ("CP-CrostonClassic", "CP-CrostonSBA", "CP-TSB", "CP-ADIDA", "CP-IMAPA"):
        walls[m] = cp_wall / 5

    zq = eval_set[["unique_id", "ds"]].copy()
    for c in QCOLS:
        zq[c] = 0.0
    zq["model"] = "Zero"
    frames.append(zq[["model", "unique_id", "ds"] + QCOLS])
    walls["Zero"] = 0.0

    failures: list[str] = []
    if not args.skip_iets:
        from models.iets_baseline import fit_predict_iets_prob_panel
        t0i = time.perf_counter()
        try:
            iets_frames = []
            for frame in iter_walk_forward_frames(init_set, eval_set, step_size=1):
                iq = fit_predict_iets_prob_panel(
                    frame.history, frame.target, quantiles=QUANTILES,
                    rscript=IETS_RSCRIPT, occurrence="auto",
                    timeout_seconds=None, per_series_timeout_seconds=10,
                    n_jobs=20, cache_path=None)
                iets_frames.append(iq)
                log(f"iETS block step {frame.step}: done "
                    f"({time.perf_counter()-t0i:.0f}s cumulative)")
            iq_all = pd.concat(iets_frames, ignore_index=True)
            frames.append(iq_all)
            walls[str(iq_all['model'].iloc[0])] = time.perf_counter() - t0i
        except Exception as e:  # noqa: BLE001 - recorded, not silent
            failures.append(f"iETS carparts walk-forward: {e!r}")
    else:
        failures.append("iETS: skipped by flag")

    all_q = pd.concat(frames, ignore_index=True)
    all_q["ds"] = pd.to_datetime(all_q["ds"])
    all_q["unique_id"] = all_q["unique_id"].astype(str)
    all_q.to_csv(OUT / "t2_quantiles_carparts.csv", index=False)

    ev = eval_set[["unique_id", "ds", "y"]].copy()
    ev["ds"] = pd.to_datetime(ev["ds"])
    ev["unique_id"] = ev["unique_id"].astype(str)
    merged = ev.merge(all_q, on=["unique_id", "ds"], how="inner")
    log(f"merged rows: {len(merged)} over {merged.model.nunique()} models")

    spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=QUANTILES)
    cov = _coverage_summary(merged)
    cov_pos = _coverage_summary(merged[merged["y"] > 0].copy())

    rows = []

    def add(model, metric, quantile, value):
        rows.append({
            "dataset": "carparts", "protocol": "walk_forward", "model": model,
            "metric": metric, "quantile": quantile, "value": value,
            "n_series": n_series, "wall_clock_sec": walls.get(model, np.nan),
            "seed": SEED,
            "refit_interval": ("online(1)" if model == "EB-Hurdle"
                               else "NA" if model == "Zero" else "1"),
        })

    for model, dfm in spl.groupby("model"):
        for _, r in dfm.iterrows():
            add(model, "SPL", r["quantile"], r["scaled_pinball"])
        add(model, "SPL_mean", "NA", float(dfm["scaled_pinball"].mean()))
    for _, r in cov.iterrows():
        add(r["model"], "coverage80_marginal", "NA", r["Coverage@80"])
        add(r["model"], "AIW80", "NA", r["AIW@80"])
    for _, r in cov_pos.iterrows():
        add(r["model"], "coverage80_positive", "NA", r["Coverage@80"])
    pd.DataFrame(rows).to_csv(OUT / "wf_carparts.csv", index=False)
    log("wrote wf_carparts.csv")
    print(spl.pivot_table(index="model", columns="quantile",
                          values="scaled_pinball").round(4).to_string())

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "t2_wf_carparts", "git_commit": git, "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__,
        "wall_clock_sec": time.perf_counter() - t_start,
        "per_model_wall_clock_sec": walls,
        "hardware": platform.processor(), "platform": platform.platform(),
        "seed": SEED, "gpu": False, "block": "1 month", "n_blocks": n_blocks,
        "ebb_selection": {"structure": sel_name, "w": sel_w},
        "failures": failures,
    }
    (OUT / "t2_run_meta.json").write_text(json.dumps(meta, indent=2))
    if failures:
        with open(OUT / "FAILURES.md", "a", encoding="utf-8") as f:
            for line in failures:
                f.write(f"- [task2] {line}\n")
    log(f"done in {(time.perf_counter()-t_start)/60:.1f} min; "
        f"failures: {failures or 'none'}")


if __name__ == "__main__":
    main()
