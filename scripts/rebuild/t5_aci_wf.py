"""Spec Task 5: Adaptive Conformal Inference (ACI) wrappers, OR walk-forward.

ACI (Gibbs & Candes 2021) on top of the two strongest existing point
wrappers, TSB and ADIDA, under the identical 7-day-block walk-forward
protocol. Mirrors the repo's static split-conformal construction exactly
(chronological 80/20 fit/calibration split of the CURRENT history each
block, pooled signed-residual distribution per model), with one change:
the residual-quantile LEVEL is per-series x per-quantile adaptive,

    alpha_{t+1} = clip(alpha_t + gamma * (q - 1{y_t <= qhat_t}), 1e-3, 0.999)

updated day-by-day within each revealed block (gamma = 0.05, noted in
NOTES.md). alpha_0 = q reproduces the static wrapper at t=0.

Scoring via the existing run_prob functions. Output:
  outputs/aistats2027_rebuild/adaptive_conformal_wf.csv
  (+ t5_quantiles.csv audit, t5_run_meta.json)

Usage: py scripts/rebuild/t5_aci_wf.py   (set ARS_SF_NJOBS=1 outside)
"""

from __future__ import annotations

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
from experiments.run_prob import _coverage_summary, _qcols, _scaled_pinball_table
from models.baselines import _fit_predict_panel, _import_statsforecast, _prepare_horizons
from models.conformal import _split_calibration_chronological

OUT = ROOT / "outputs" / "aistats2027_rebuild"
OUT.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = _qcols(QUANTILES)
SEED = 42
WALK_STEP = 7
GAMMA = 0.05
CAL_RATIO = 0.2
BASE_MODELS = ["TSB", "ADIDA"]


def log(msg: str) -> None:
    print(f"[t5] {msg}", flush=True)


def fit_predict_points(train_df, target_df, freq="D"):
    """Point predictions for TSB + ADIDA in one StatsForecast call."""
    _, M = _import_statsforecast()
    models = [M["TSB"](alpha_d=0.5, alpha_p=0.45), M["ADIDA"]()]
    horizon_df = _prepare_horizons(target_df.groupby("unique_id").size())
    pred = _fit_predict_panel(
        train_df=train_df, horizon_df=horizon_df, models=models,
        freq=freq, probabilistic=False, levels=None, n_jobs=1)
    if pred.empty:
        return pred
    pred = pred.merge(target_df[["unique_id", "ds"]], on=["unique_id", "ds"],
                      how="inner")
    return pred


def main() -> None:
    t_start = time.perf_counter()
    df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    n_series = int(init_set.unique_id.nunique())
    log(f"OR: {n_series} series")

    uids = sorted(init_set["unique_id"].astype(str).unique())
    # alpha state: {model: DataFrame indexed by uid, columns per quantile}
    alpha = {m: pd.DataFrame({f"a_{q}": q for q in QUANTILES}, index=uids)
             for m in BASE_MODELS}

    frames = []
    n_blocks = 0
    for frame in iter_walk_forward_frames(init_set, eval_set, step_size=WALK_STEP):
        n_blocks += 1
        tb = time.perf_counter()
        fit_df, cal_df = _split_calibration_chronological(frame.history, cal_ratio=CAL_RATIO)
        cal_pred = fit_predict_points(fit_df, cal_df[["unique_id", "ds"]])
        cal_merged = cal_df[["unique_id", "ds", "y"]].merge(
            cal_pred, on=["unique_id", "ds"], how="inner")
        residuals = {}
        for m in BASE_MODELS:
            valid = cal_merged.dropna(subset=[m])
            residuals[m] = np.sort(
                valid["y"].to_numpy(float) - valid[m].to_numpy(float))
        ev_pred = fit_predict_points(frame.history, frame.target[["unique_id", "ds"]])

        tgt = frame.target[["unique_id", "ds", "y"]].merge(
            ev_pred, on=["unique_id", "ds"], how="left")
        tgt["unique_id"] = tgt["unique_id"].astype(str)
        for m in BASE_MODELS:
            res = residuals[m]
            if len(res) < 10:
                continue
            a_now = alpha[m].reindex(tgt["unique_id"].values)
            point = tgt[m].to_numpy(float)
            out = tgt[["unique_id", "ds"]].copy()
            for q in QUANTILES:
                lev = a_now[f"a_{q}"].to_numpy(float)
                adj = np.quantile(res, np.clip(lev, 1e-3, 0.999),
                                  method="linear") if len(res) else 0.0
                # np.quantile with vector levels returns per-level values
                out[f"q_{q}"] = np.maximum(point + adj, 0.0)
            arr = out[QCOLS].to_numpy(float)
            for j in range(1, arr.shape[1]):
                arr[:, j] = np.maximum(arr[:, j], arr[:, j - 1])
            out[QCOLS] = arr
            out["model"] = f"ACI-{m}"
            frames.append(out[["model", "unique_id", "ds"] + QCOLS].copy())

            # sequential day-by-day alpha update within the revealed block
            upd = out.merge(tgt[["unique_id", "ds", "y"]],
                            on=["unique_id", "ds"]).sort_values(["unique_id", "ds"])
            for q in QUANTILES:
                covered = (upd["y"].to_numpy(float)
                           <= upd[f"q_{q}"].to_numpy(float)).astype(float)
                g = pd.DataFrame({
                    "uid": upd["unique_id"].to_numpy(),
                    "delta": GAMMA * (q - covered),
                }).groupby("uid")["delta"].sum()
                col = f"a_{q}"
                alpha[m].loc[g.index, col] = np.clip(
                    alpha[m].loc[g.index, col].to_numpy(float) + g.to_numpy(float),
                    1e-3, 0.999)
        log(f"block {n_blocks}: {time.perf_counter()-tb:.0f}s")

    all_q = pd.concat(frames, ignore_index=True)
    all_q["ds"] = pd.to_datetime(all_q["ds"])
    all_q.to_csv(OUT / "t5_quantiles.csv", index=False)

    ev = eval_set[["unique_id", "ds", "y"]].copy()
    ev["ds"] = pd.to_datetime(ev["ds"])
    ev["unique_id"] = ev["unique_id"].astype(str)
    merged = ev.merge(all_q, on=["unique_id", "ds"], how="inner")
    spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=QUANTILES)
    cov = _coverage_summary(merged)
    cov_pos = _coverage_summary(merged[merged["y"] > 0].copy())

    rows = []

    def add(model, metric, quantile, value):
        rows.append({
            "dataset": "online_retail", "protocol": "walk_forward",
            "model": model, "metric": metric, "quantile": quantile,
            "value": value, "n_series": n_series,
            "wall_clock_sec": time.perf_counter() - t_start, "seed": SEED,
            "refit_interval": "1", "gamma": GAMMA,
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
    pd.DataFrame(rows).to_csv(OUT / "adaptive_conformal_wf.csv", index=False)
    log("wrote adaptive_conformal_wf.csv")
    print(spl.pivot_table(index="model", columns="quantile",
                          values="scaled_pinball").round(4).to_string())

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    (OUT / "t5_run_meta.json").write_text(json.dumps({
        "task": "t5_aci_wf", "git_commit": git, "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__,
        "wall_clock_sec": time.perf_counter() - t_start,
        "hardware": platform.processor(), "platform": platform.platform(),
        "seed": SEED, "gpu": False, "gamma": GAMMA,
        "base_models": BASE_MODELS, "cal_ratio": CAL_RATIO,
        "aci_definition": "per-series per-quantile adaptive level on the pooled "
                          "residual distribution; day-by-day updates within "
                          "revealed blocks; alpha_0 = q",
    }, indent=2))
    log(f"done in {(time.perf_counter()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
