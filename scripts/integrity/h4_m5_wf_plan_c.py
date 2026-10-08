"""Task 4 (plan c): M5 strict walk-forward, full 5,000 series.

Methods: EBB (mixture, w=0.95; selected once at the origin, online
sufficient-statistic updates), the five CP wrappers (refit each block),
ACI-ADIDA (gamma=0.05, per-series x per-quantile adaptive levels, residual
construction identical to the static wrapper), and the Zero diagnostic.
AutoARIMA / AutoTheta / iETS are deliberately excluded (plan c).

Protocol: seven-day blocks over the 647-day external span (93 blocks),
identical mechanics to the Online Retail walk-forward study.

The script prints a full-run projection after block 2 and aborts if the
projection exceeds ABORT_HOURS (safety fuse); otherwise it continues to
completion. Quantile frames are appended to disk per block to bound
memory; scoring runs model-by-model at the end through the existing
run_prob functions.

Outputs (outputs/<date>/): m5_walkforward.csv (tidy metrics),
m5_wf_quantiles.csv (audit), h4_run_meta.json

Usage: py scripts/integrity/h4_m5_wf_plan_c.py
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

from data_loading import load_m5_long, preprocess_m5, train_eval_split_fixed_origin
from experiments.protocols import iter_walk_forward_frames
from experiments.run_prob import _coverage_summary, _qcols, _scaled_pinball_table
from models.baselines import _fit_predict_panel, _import_statsforecast, _prepare_horizons
from models.conformal import (
    _split_calibration_chronological, fit_predict_conformal_baselines,
)
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import (
    initialize_online_eb_hurdle, predict_online_eb_hurdle, update_online_eb_hurdle,
)
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = _qcols(QUANTILES)
SEED = 42
WALK_STEP = 7
CAL_RATIO = 0.2
GAMMA = 0.05
ABORT_HOURS = 20.0
QUANT_PATH = OUT / "m5_wf_quantiles.csv"


def log(msg: str) -> None:
    print(f"[h4] {msg}", flush=True)


def adida_points(train_df, target_df):
    _, M = _import_statsforecast()
    horizon_df = _prepare_horizons(target_df.groupby("unique_id").size())
    pred = _fit_predict_panel(train_df=train_df, horizon_df=horizon_df,
                              models=[M["ADIDA"]()], freq="D",
                              probabilistic=False, levels=None, n_jobs=1)
    if pred.empty:
        return pred
    col = [c for c in pred.columns if c not in {"unique_id", "ds", "index"}][0]
    pred = pred.rename(columns={col: "ADIDA"})
    return pred.merge(target_df[["unique_id", "ds"]], on=["unique_id", "ds"],
                      how="inner")


def append_frames(frames: list[pd.DataFrame], first: bool) -> None:
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(QUANT_PATH, index=False, mode="w" if first else "a",
              header=first)


def main() -> None:
    t_start = time.perf_counter()
    set_seed(SEED)
    sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    df = preprocess_m5(sales, cal, sample_size=5000)
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
    n_series = int(init_set["unique_id"].nunique())
    span = int(eval_set.groupby("unique_id").size().max())
    n_blocks = int(np.ceil(span / WALK_STEP))
    log(f"{n_series} series, span {span}d, {n_blocks} blocks")

    t0 = time.perf_counter()
    labels = mixture_group_labels(init_set, k=0, fit_discount=0.95).labels
    state = initialize_online_eb_hurdle(
        init_set, group_labels=labels, bootstrap_draws=0, bootstrap_seed=SEED,
        group_shrink_strength=0.0, dynamic_occurrence=False,
        occurrence_discount=1.0, item_variance_mode="conjugate",
        item_variance_shrink_strength=20.0, fit_discount=0.95)
    log(f"EBB init (mixture, 0.95): {time.perf_counter()-t0:.0f}s")

    uids = sorted(init_set["unique_id"].astype(str).unique())
    alpha = pd.DataFrame({f"a_{q}": q for q in QUANTILES}, index=uids)

    walls = {"EBB": 0.0, "CP5": 0.0, "ACI": 0.0}
    bi = 0
    for frame in iter_walk_forward_frames(init_set, eval_set, step_size=WALK_STEP):
        bi += 1
        blk = []
        # --- EBB online ---
        tb = time.perf_counter()
        hbq = predict_online_eb_hurdle(state, frame.target, quantiles=QUANTILES,
                                    n_samples=2000, include_hyper_uncertainty=False)
        for c in QCOLS:
            if c not in hbq.columns:
                hbq[c] = np.nan
        hbq["model"] = "EB-Hurdle"
        blk.append(hbq[["model", "unique_id", "ds"] + QCOLS])
        state = update_online_eb_hurdle(state, frame.target)
        walls["EBB"] += time.perf_counter() - tb

        # --- five CP wrappers ---
        tb = time.perf_counter()
        cq = fit_predict_conformal_baselines(frame.history, frame.target,
                                             quantiles=QUANTILES,
                                             cal_ratio=CAL_RATIO, freq="D")
        if not cq.empty:
            blk.append(cq)
        walls["CP5"] += time.perf_counter() - tb

        # --- ACI-ADIDA ---
        tb = time.perf_counter()
        fit_df, cal_df = _split_calibration_chronological(frame.history,
                                                          cal_ratio=CAL_RATIO)
        cal_pred = adida_points(fit_df, cal_df[["unique_id", "ds"]])
        cal_merged = cal_df[["unique_id", "ds", "y"]].merge(
            cal_pred, on=["unique_id", "ds"], how="inner").dropna(subset=["ADIDA"])
        res = np.sort(cal_merged["y"].to_numpy(float)
                      - cal_merged["ADIDA"].to_numpy(float))
        ev_pred = adida_points(frame.history, frame.target[["unique_id", "ds"]])
        tgt = frame.target[["unique_id", "ds", "y"]].merge(
            ev_pred, on=["unique_id", "ds"], how="left")
        tgt["unique_id"] = tgt["unique_id"].astype(str)
        if len(res) >= 10:
            a_now = alpha.reindex(tgt["unique_id"].values)
            point = tgt["ADIDA"].to_numpy(float)
            out = tgt[["unique_id", "ds"]].copy()
            for q in QUANTILES:
                lev = np.clip(a_now[f"a_{q}"].to_numpy(float), 1e-3, 0.999)
                adj = np.quantile(res, lev, method="linear")
                out[f"q_{q}"] = np.maximum(point + adj, 0.0)
            arr = out[QCOLS].to_numpy(float)
            for j in range(1, arr.shape[1]):
                arr[:, j] = np.maximum(arr[:, j], arr[:, j - 1])
            out[QCOLS] = arr
            out["model"] = "ACI-ADIDA"
            blk.append(out[["model", "unique_id", "ds"] + QCOLS].copy())
            upd = out.merge(tgt[["unique_id", "ds", "y"]],
                            on=["unique_id", "ds"]).sort_values(["unique_id", "ds"])
            for q in QUANTILES:
                covered = (upd["y"].to_numpy(float)
                           <= upd[f"q_{q}"].to_numpy(float)).astype(float)
                g = pd.DataFrame({"uid": upd["unique_id"].to_numpy(),
                                  "delta": GAMMA * (q - covered)}).groupby("uid")["delta"].sum()
                alpha.loc[g.index, f"a_{q}"] = np.clip(
                    alpha.loc[g.index, f"a_{q}"].to_numpy(float) + g.to_numpy(float),
                    1e-3, 0.999)
        walls["ACI"] += time.perf_counter() - tb

        # --- Zero ---
        z = frame.target[["unique_id", "ds"]].copy()
        for c in QCOLS:
            z[c] = 0.0
        z["model"] = "Zero"
        blk.append(z[["model", "unique_id", "ds"] + QCOLS])

        append_frames(blk, first=(bi == 1))
        el = time.perf_counter() - t_start
        log(f"block {bi}/{n_blocks} done ({el/60:.0f} min elapsed)")
        if bi == 2:
            proj = el / 2 * n_blocks / 3600
            log(f"PROJECTION from 2 blocks: ~{proj:.1f}h for {n_blocks} blocks")
            if proj > ABORT_HOURS:
                log(f"ABORT: projection exceeds {ABORT_HOURS}h fuse")
                sys.exit(2)

    # ---------------- scoring ----------------
    ev = eval_set[["unique_id", "ds", "y"]].copy()
    ev["ds"] = pd.to_datetime(ev["ds"])
    ev["unique_id"] = ev["unique_id"].astype(str)
    all_q = pd.read_csv(QUANT_PATH)
    all_q["ds"] = pd.to_datetime(all_q["ds"])
    all_q["unique_id"] = all_q["unique_id"].astype(str)

    rows = []
    for model, mq in all_q.groupby("model"):
        merged = ev.merge(mq, on=["unique_id", "ds"], how="inner")
        spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=QUANTILES)
        spl = spl[spl["model"] == model] if "model" in spl.columns else spl
        cov = _coverage_summary(merged)
        cov_pos = _coverage_summary(merged[merged["y"] > 0].copy())
        for _, r in spl.iterrows():
            rows.append({"dataset": "m5", "protocol": "walk_forward",
                         "model": model, "metric": "scaled_pinball",
                         "quantile": r["quantile"],
                         "value": float(r["scaled_pinball"])})
        rows.append({"dataset": "m5", "protocol": "walk_forward", "model": model,
                     "metric": "scaled_pinball", "quantile": "mean",
                     "value": float(spl["scaled_pinball"].mean())})
        c = cov.iloc[0]
        cp = cov_pos.iloc[0]
        for name, val in (("coverage80", c["Coverage@80"]),
                          ("coverage80_positive", cp["Coverage@80"]),
                          ("aiw80", c["AIW@80"])):
            rows.append({"dataset": "m5", "protocol": "walk_forward",
                         "model": model, "metric": name, "quantile": "",
                         "value": float(val)})
        log(f"scored {model}: mean SPL "
            f"{float(spl['scaled_pinball'].mean()):.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "m5_walkforward.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "h4_m5_wf_plan_c", "commit": commit, "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "seed": SEED,
        "protocol": f"{WALK_STEP}-day blocks, {n_blocks} blocks, {n_series} series",
        "methods": "EBB (mixture,0.95, online) + CP x5 (refit/block) + "
                   "ACI-ADIDA (gamma 0.05) + Zero; AA/AT/iETS excluded (plan c)",
        "component_walls_s": {k: round(v, 1) for k, v in walls.items()},
        "wall_clock_total_s": round(time.perf_counter() - t_start, 1),
    }
    (OUT / "h4_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log("wrote m5_walkforward.csv")


if __name__ == "__main__":
    main()
