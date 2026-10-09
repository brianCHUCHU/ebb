"""Spec Task 1: complete the Online Retail walk-forward baseline set.

Adds Zero / AutoARIMA / AutoTheta (refit every 4 blocks) / iETS (attempted,
time-boxed) to the existing 7-day-block walk-forward protocol, scoring
EVERYTHING through the existing run_prob functions (identical scaling and
quantile post-processing; spec hard rule 2/3).

Acceptance gate: the five existing conformal wrappers re-scored through this
path must reproduce the stored Table-3 numbers to <1% (they are the same
frames, so the expected difference is 0).

Outputs (outputs/paper_rebuild/):
  wf_online_retail.csv  tidy per spec schema + refit_interval
  t1_quantiles_new.csv  raw quantile frames for the new models (audit trail)
  t1_run_meta.json
Appends to NOTES.md / FAILURES.md at the same level.

Usage: py scripts/rebuild/t1_wf_or.py [--skip-iets] [--iets-budget-hours 4]
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

from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from experiments.protocols import iter_walk_forward_frames
from experiments.run_prob import (
    _coverage_summary,
    _predict_non_hb_prob_models_once,
    _qcols,
    _scaled_pinball_table,
)

OUT_OLD = ROOT / "outputs" / "paper_runs" / "or_prob_wf_ebb"
OUT = ROOT / "outputs" / "paper_rebuild"
OUT.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = _qcols(QUANTILES)
SEED = 42
WALK_STEP = 7
REFIT_EVERY = 4  # blocks, for AutoARIMA / AutoTheta / iETS (spec-allowed)
IETS_RSCRIPT = r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe"


def log(msg: str) -> None:
    print(f"[t1] {msg}", flush=True)


def load_or():
    df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
    return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)


def chunked_frames(init_set, eval_set):
    """Group walk-forward frames into refit chunks of REFIT_EVERY blocks.

    Yields (chunk_index, history_at_chunk_start, concatenated_target).
    History at chunk start contains init + all earlier revealed blocks, so no
    evaluation-span observation is used before it is revealed (hard rule 1);
    within a chunk the model predicts multi-step ahead without refitting.
    """
    frames = list(iter_walk_forward_frames(init_set, eval_set, step_size=WALK_STEP))
    for ci in range(0, len(frames), REFIT_EVERY):
        chunk = frames[ci:ci + REFIT_EVERY]
        history = chunk[0].history
        target = pd.concat([f.target for f in chunk], ignore_index=True)
        yield ci // REFIT_EVERY, history, target


def acceptance_check(init_set, eval_set) -> pd.DataFrame:
    old_q = pd.read_csv(OUT_OLD / "prob_quantiles.csv")
    old_q = old_q.drop(columns=["protocol"], errors="ignore")
    old_q["ds"] = pd.to_datetime(old_q["ds"])
    # StockCode ids are mixed numeric/alphanumeric; read_csv infers ints for
    # the numeric ones, which silently breaks the join against the str ids.
    old_q["unique_id"] = old_q["unique_id"].astype(str)
    merged = eval_set[["unique_id", "ds", "y"]].merge(
        old_q, on=["unique_id", "ds"], how="inner")
    if merged.empty:
        raise SystemExit("ACCEPTANCE FAILED: stored quantiles do not join eval span")
    rescored = _scaled_pinball_table(merged, init_set=init_set, quantiles=QUANTILES)
    stored = pd.read_csv(OUT_OLD / "prob_pinball_scaled.csv")
    cmp = rescored.merge(stored, on=["model", "quantile"], suffixes=("_new", "_stored"))
    cmp["rel_diff"] = (cmp["scaled_pinball_new"] - cmp["scaled_pinball_stored"]).abs() / cmp["scaled_pinball_stored"]
    worst = float(cmp["rel_diff"].max())
    log(f"acceptance: worst relative diff vs stored Table-3 source = {worst:.2e}")
    if worst > 0.01:
        raise SystemExit(f"ACCEPTANCE FAILED: worst rel diff {worst:.4f} > 1% — stopping per spec.")
    return old_q


def zero_frame(eval_set) -> pd.DataFrame:
    z = eval_set[["unique_id", "ds"]].copy()
    for c in QCOLS:
        z[c] = 0.0
    z["model"] = "Zero"
    return z[["model", "unique_id", "ds"] + QCOLS]


def aa_at_frames(init_set, eval_set) -> tuple[pd.DataFrame, dict[str, float]]:
    frames, t0 = [], time.perf_counter()
    for ci, history, target in chunked_frames(init_set, eval_set):
        tchunk = time.perf_counter()
        q = _predict_non_hb_prob_models_once(
            history, target, quantiles=QUANTILES, baseline_mode="paper",
            freq="D", season_length=7)
        frames.append(q)
        log(f"AA/AT chunk {ci}: {time.perf_counter() - tchunk:.0f}s "
            f"({len(target)} target rows)")
    allq = pd.concat(frames, ignore_index=True)
    wall = time.perf_counter() - t0
    return allq, {"AutoARIMA": wall / 2, "AutoTheta": wall / 2}


def iets_frames(init_set, eval_set, budget_hours: float) -> tuple[pd.DataFrame | None, float, str]:
    from models.iets_baseline import fit_predict_iets_prob_panel
    frames, t0 = [], time.perf_counter()
    chunks = list(chunked_frames(init_set, eval_set))
    for ci, history, target in chunks:
        tchunk = time.perf_counter()
        q = fit_predict_iets_prob_panel(
            history, target, quantiles=QUANTILES,
            rscript=IETS_RSCRIPT, occurrence="auto",
            timeout_seconds=None, per_series_timeout_seconds=10,
            n_jobs=20, cache_path=None,
        )
        dt = time.perf_counter() - tchunk
        frames.append(q)
        elapsed = time.perf_counter() - t0
        projected = elapsed / (ci + 1) * len(chunks)
        log(f"iETS chunk {ci}: {dt:.0f}s; projected total {projected/3600:.2f}h")
        if projected > budget_hours * 3600 and ci + 1 < len(chunks):
            return None, elapsed, (
                f"aborted after chunk {ci}: projected total {projected/3600:.2f}h "
                f"exceeds the {budget_hours}h budget")
    return pd.concat(frames, ignore_index=True), time.perf_counter() - t0, "ok"


def tidy_rows(spl: pd.DataFrame, cov: pd.DataFrame, cov_pos: pd.DataFrame,
              n_series: int, walls: dict[str, float],
              refits: dict[str, str]) -> pd.DataFrame:
    rows = []

    def add(model, metric, quantile, value):
        rows.append({
            "dataset": "online_retail", "protocol": "walk_forward",
            "model": model, "metric": metric, "quantile": quantile,
            "value": value, "n_series": n_series,
            "wall_clock_sec": walls.get(model, np.nan), "seed": SEED,
            "refit_interval": refits.get(model, "1"),
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
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-iets", action="store_true")
    ap.add_argument("--iets-budget-hours", type=float, default=4.0)
    args = ap.parse_args()

    t_start = time.perf_counter()
    init_set, eval_set = load_or()
    n_series = int(init_set.unique_id.nunique())
    log(f"OR loaded: {n_series} series")

    old_q = acceptance_check(init_set, eval_set)

    walls: dict[str, float] = {}
    refits: dict[str, str] = {m: "1" for m in old_q["model"].unique()}
    refits["EB-Hurdle"] = "online(1)"

    t0 = time.perf_counter()
    zq = zero_frame(eval_set)
    walls["Zero"] = time.perf_counter() - t0
    refits["Zero"] = "NA"

    aaq, aa_walls = aa_at_frames(init_set, eval_set)
    walls.update(aa_walls)
    refits["AutoARIMA"] = refits["AutoTheta"] = str(REFIT_EVERY)

    new_frames = [zq, aaq]
    failures: list[str] = []
    if args.skip_iets:
        failures.append("iETS: skipped by flag")
    else:
        iq, iets_wall, status = iets_frames(init_set, eval_set, args.iets_budget_hours)
        if iq is None:
            failures.append(f"iETS walk-forward: {status}")
        else:
            iq["model"] = iq["model"].astype(str)
            new_frames.append(iq)
            walls[str(iq['model'].iloc[0])] = iets_wall
            refits[str(iq['model'].iloc[0])] = str(REFIT_EVERY)

    newq = pd.concat(new_frames, ignore_index=True)[["model", "unique_id", "ds"] + QCOLS]
    newq.to_csv(OUT / "t1_quantiles_new.csv", index=False)

    newq["ds"] = pd.to_datetime(newq["ds"])
    newq["unique_id"] = newq["unique_id"].astype(str)
    all_q = pd.concat([old_q[["model", "unique_id", "ds"] + QCOLS], newq],
                      ignore_index=True)
    merged = eval_set[["unique_id", "ds", "y"]].merge(
        all_q, on=["unique_id", "ds"], how="inner")

    spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=QUANTILES)
    cov = _coverage_summary(merged)
    cov_pos = _coverage_summary(merged[merged["y"] > 0].copy())

    tidy = tidy_rows(spl, cov, cov_pos, n_series, walls, refits)
    tidy.to_csv(OUT / "wf_online_retail.csv", index=False)
    log("wrote wf_online_retail.csv")
    print(spl.pivot_table(index="model", columns="quantile",
                          values="scaled_pinball").round(4).to_string())

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "t1_wf_online_retail",
        "git_commit": git,
        "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__,
        "wall_clock_sec": time.perf_counter() - t_start,
        "per_model_wall_clock_sec": walls,
        "hardware": platform.processor(),
        "platform": platform.platform(),
        "seed": SEED, "gpu": False,
        "walk_step_days": WALK_STEP, "refit_every_blocks": REFIT_EVERY,
        "failures": failures,
    }
    (OUT / "t1_run_meta.json").write_text(json.dumps(meta, indent=2))
    if failures:
        with open(OUT / "FAILURES.md", "a", encoding="utf-8") as f:
            for line in failures:
                f.write(f"- [task1] {line}\n")
    log(f"done in {(time.perf_counter() - t_start)/60:.1f} min; failures: {failures or 'none'}")


if __name__ == "__main__":
    main()
