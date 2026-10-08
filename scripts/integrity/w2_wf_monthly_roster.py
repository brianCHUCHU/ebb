"""W2: full walk-forward roster on the monthly panels (Auto, RAF), plus
ACI-ADIDA for Carparts (the only monthly panel that already has a roster).

Pre-registered protocol: monthly blocks (h=6 Auto/Carparts, h=12 RAF); every
statistical baseline refits each block; EBB selected once at the origin at
its audited configuration (Auto global 0.99, RAF mixture 0.997, Carparts
global 0.90) and updates sufficient statistics online. ACI-ADIDA: gamma 0.05,
per-series x per-quantile adaptive levels, residual pool identical to the
static wrapper. iETS via R, 20 jobs, 10 s/series cap, recorded if it fails.

Outputs (outputs/<date>/): wf_roster_<panel>.csv, wf_roster_quantiles_<panel>.csv,
w2_run_meta_<panel>.json

Usage: py scripts/integrity/w2_wf_monthly_roster.py <auto|raf|carparts> [--aci-only]
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

from data_loading import load_generic_long, train_eval_split_last_h
from experiments.protocols import iter_walk_forward_frames
from experiments.run_prob import (_coverage_summary, _predict_non_hb_prob_models_once,
                                  _qcols, _scaled_pinball_table)
from models.baselines import _fit_predict_panel, _import_statsforecast, _prepare_horizons
from models.conformal import _split_calibration_chronological, fit_predict_conformal_baselines
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import initialize_online_eb_hurdle, predict_online_eb_hurdle, update_online_eb_hurdle

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
Q5 = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = _qcols(Q5)
H = {"auto": 6, "carparts": 6, "raf": 12}
CONFIG = {"auto": ("global", 0.99), "raf": ("mixture", 0.997), "carparts": ("global", 0.90)}
IETS_RSCRIPT = r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe"
GAMMA, CAL_RATIO = 0.05, 0.2


def log(m):
    print(f"[w2] {m}", flush=True)


def adida_points(train_df, target_df):
    _, M = _import_statsforecast()
    horizon_df = _prepare_horizons(target_df.groupby("unique_id").size())
    pred = _fit_predict_panel(train_df=train_df, horizon_df=horizon_df, models=[M["ADIDA"]()],
                              freq="MS", probabilistic=False, levels=None, n_jobs=1)
    if pred.empty:
        return pred
    col = [c for c in pred.columns if c not in {"unique_id", "ds", "index"}][0]
    return pred.rename(columns={col: "ADIDA"}).merge(target_df[["unique_id", "ds"]],
                                                     on=["unique_id", "ds"], how="inner")


def main():
    panel = sys.argv[1]
    aci_only = "--aci-only" in sys.argv
    t0 = time.perf_counter()
    df = load_generic_long(ROOT / f"data/{panel}_long.csv")
    init_set, eval_set = train_eval_split_last_h(df, h=H[panel])
    n_series = int(init_set["unique_id"].nunique())
    structure, w = CONFIG[panel]
    walls, failures, frames_out = {}, [], []

    if not aci_only:
        labels = (mixture_group_labels(init_set, k=0, fit_discount=w).labels
                  if structure == "mixture" else None)
        state = initialize_online_eb_hurdle(
            init_set, group_labels=labels, bootstrap_draws=0, bootstrap_seed=42,
            group_shrink_strength=0.0, dynamic_occurrence=False, occurrence_discount=1.0,
            item_variance_mode="conjugate", item_variance_shrink_strength=20.0, fit_discount=w)

    uids = sorted(init_set["unique_id"].astype(str).unique())
    alpha = pd.DataFrame({f"a_{q}": q for q in Q5}, index=uids)
    n_blocks = 0
    for frame in iter_walk_forward_frames(init_set, eval_set, step_size=1):
        n_blocks += 1
        if not aci_only:
            tb = time.perf_counter()
            hbq = predict_online_eb_hurdle(state, frame.target, quantiles=Q5, n_samples=2000,
                                        include_hyper_uncertainty=False)
            for c in QCOLS:
                if c not in hbq.columns:
                    hbq[c] = np.nan
            hbq["model"] = "EB-Hurdle"
            frames_out.append(hbq[["model", "unique_id", "ds"] + QCOLS])
            walls["EB-Hurdle"] = walls.get("EB-Hurdle", 0.0) + time.perf_counter() - tb

            tb = time.perf_counter()
            nq = _predict_non_hb_prob_models_once(frame.history, frame.target, quantiles=Q5,
                                                  baseline_mode="paper", freq="MS", season_length=12)
            frames_out.append(nq)
            half = (time.perf_counter() - tb) / 2
            walls["AutoARIMA"] = walls.get("AutoARIMA", 0.0) + half
            walls["AutoTheta"] = walls.get("AutoTheta", 0.0) + half

            tb = time.perf_counter()
            cq = fit_predict_conformal_baselines(frame.history, frame.target, quantiles=Q5,
                                                 cal_ratio=CAL_RATIO, freq="MS")
            if not cq.empty:
                frames_out.append(cq)
            fifth = (time.perf_counter() - tb) / 5
            for m in ("CP-CrostonClassic", "CP-CrostonSBA", "CP-TSB", "CP-ADIDA", "CP-IMAPA"):
                walls[m] = walls.get(m, 0.0) + fifth

        # --- ACI-ADIDA ---
        tb = time.perf_counter()
        fit_df, cal_df = _split_calibration_chronological(frame.history, cal_ratio=CAL_RATIO)
        cal_pred = adida_points(fit_df, cal_df[["unique_id", "ds"]])
        cal_m = cal_df[["unique_id", "ds", "y"]].merge(cal_pred, on=["unique_id", "ds"],
                                                       how="inner").dropna(subset=["ADIDA"])
        res = np.sort(cal_m["y"].to_numpy(float) - cal_m["ADIDA"].to_numpy(float))
        ev_pred = adida_points(frame.history, frame.target[["unique_id", "ds"]])
        tgt = frame.target[["unique_id", "ds", "y"]].merge(ev_pred, on=["unique_id", "ds"], how="left")
        tgt["unique_id"] = tgt["unique_id"].astype(str)
        if len(res) >= 10:
            a_now = alpha.reindex(tgt["unique_id"].values)
            point = tgt["ADIDA"].to_numpy(float)
            out = tgt[["unique_id", "ds"]].copy()
            for q in Q5:
                lev = np.clip(a_now[f"a_{q}"].to_numpy(float), 1e-3, 0.999)
                out[f"q_{q}"] = np.maximum(point + np.quantile(res, lev, method="linear"), 0.0)
            arr = out[QCOLS].to_numpy(float)
            for j in range(1, arr.shape[1]):
                arr[:, j] = np.maximum(arr[:, j], arr[:, j - 1])
            out[QCOLS] = arr
            out["model"] = "ACI-ADIDA"
            frames_out.append(out[["model", "unique_id", "ds"] + QCOLS].copy())
            upd = out.merge(tgt[["unique_id", "ds", "y"]], on=["unique_id", "ds"]).sort_values(["unique_id", "ds"])
            for q in Q5:
                covered = (upd["y"].to_numpy(float) <= upd[f"q_{q}"].to_numpy(float)).astype(float)
                g = pd.DataFrame({"uid": upd["unique_id"].to_numpy(),
                                  "delta": GAMMA * (q - covered)}).groupby("uid")["delta"].sum()
                alpha.loc[g.index, f"a_{q}"] = np.clip(
                    alpha.loc[g.index, f"a_{q}"].to_numpy(float) + g.to_numpy(float), 1e-3, 0.999)
        walls["ACI-ADIDA"] = walls.get("ACI-ADIDA", 0.0) + time.perf_counter() - tb

        if not aci_only:
            state = update_online_eb_hurdle(state, frame.target)
        log(f"{panel} block {n_blocks}: done ({(time.perf_counter()-t0)/60:.1f} min)")

    if not aci_only:
        from models.iets_baseline import fit_predict_iets_prob_panel
        tb = time.perf_counter()
        try:
            iets_frames = []
            for frame in iter_walk_forward_frames(init_set, eval_set, step_size=1):
                iets_frames.append(fit_predict_iets_prob_panel(
                    frame.history, frame.target, quantiles=Q5, rscript=IETS_RSCRIPT,
                    occurrence="auto", timeout_seconds=None, per_series_timeout_seconds=10,
                    n_jobs=20, cache_path=None))
            iq = pd.concat(iets_frames, ignore_index=True)
            frames_out.append(iq)
            walls[str(iq["model"].iloc[0])] = time.perf_counter() - tb
        except Exception as e:  # noqa: BLE001
            failures.append(f"iETS {panel} walk-forward: {e!r}")
            log(f"iETS failed: {e!r}")
        zq = eval_set[["unique_id", "ds"]].copy()
        for c in QCOLS:
            zq[c] = 0.0
        zq["model"] = "Zero"
        frames_out.append(zq[["model", "unique_id", "ds"] + QCOLS])

    aq = pd.concat(frames_out, ignore_index=True)
    aq["ds"] = pd.to_datetime(aq["ds"])
    aq["unique_id"] = aq["unique_id"].astype(str)
    tag = "_aci" if aci_only else ""
    aq.to_csv(OUT / f"wf_roster_quantiles_{panel}{tag}.csv", index=False)
    ev = eval_set[["unique_id", "ds", "y"]].copy()
    ev["ds"] = pd.to_datetime(ev["ds"])
    ev["unique_id"] = ev["unique_id"].astype(str)
    merged = ev.merge(aq, on=["unique_id", "ds"], how="inner")
    spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=Q5)
    cov = _coverage_summary(merged).set_index("model")
    cov_pos = _coverage_summary(merged[merged["y"] > 0].copy()).set_index("model")
    rows = []
    for m, g in spl.groupby("model"):
        for _, r in g.iterrows():
            rows.append({"dataset": panel, "protocol": "walk_forward", "model": m,
                         "metric": "scaled_pinball", "quantile": r["quantile"],
                         "value": float(r["scaled_pinball"])})
        rows.append({"dataset": panel, "protocol": "walk_forward", "model": m,
                     "metric": "scaled_pinball", "quantile": "mean",
                     "value": float(g["scaled_pinball"].mean())})
        cp = float(cov_pos.loc[m, "Coverage@80"]) if m in cov_pos.index else float("nan")
        for name, val in (("coverage80", cov.loc[m, "Coverage@80"]), ("coverage80_positive", cp),
                          ("aiw80", cov.loc[m, "AIW@80"]), ("wall_clock_s", walls.get(m, float("nan")))):
            rows.append({"dataset": panel, "protocol": "walk_forward", "model": m,
                         "metric": name, "quantile": "", "value": float(val)})
    pd.DataFrame(rows).to_csv(OUT / f"wf_roster_{panel}{tag}.csv", index=False)
    meta = {"task": f"w2_wf_roster_{panel}{tag}", "panel": panel, "n_series": n_series,
            "blocks": n_blocks, "ebb_config": CONFIG[panel], "aci_gamma": GAMMA,
            "wall_clock_total_s": round(time.perf_counter() - t0, 1), "failures": failures,
            "platform": platform.platform(), "python": sys.version.split()[0], "seed": 42}
    (OUT / f"w2_run_meta_{panel}{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for m, g in spl.groupby("model"):
        log(f"{panel} {m}: mean SPL {float(g['scaled_pinball'].mean()):.4f}")


if __name__ == "__main__":
    main()
