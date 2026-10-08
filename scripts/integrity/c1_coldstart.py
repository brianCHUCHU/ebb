"""C1: short-history (cold-start) scaling experiment, per docs/DESIGN_coldstart.md.

For each panel and truncation length L, the initialization window is cut to
its last min(L, n_i) observations per series; the external evaluation span
is unchanged. EBB (audited structure/discount, labels re-learned on the
truncated window, B=20), TweedieGP (released defaults via the venv runner),
CP-ADIDA / CP-Croston (split conformal, cal_ratio 0.2) and Zero are scored
with the shared functions; the SPL scale is taken from the FULL
initialization window so levels are comparable. Median credibility weights
come from scripts/analysis/leverage_dual_lambda.dual_lambda (unchanged).

Outputs (outputs/<date>/): coldstart_<panel>.csv, c1_run_meta_<panel>.json,
  coldstart_tmp/<panel>_L<L>_{in,out}.csv
Usage: py scripts/integrity/c1_coldstart.py <panel> [panel ...]
"""
from __future__ import annotations

import importlib.util
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

from data_loading import (load_generic_long, load_m5_long, load_online_retail, preprocess_m5,
                          preprocess_online_retail, train_eval_split_fixed_origin,
                          train_eval_split_last_h)
from experiments.run_prob import _enforce_monotonic_quantiles, _scaled_pinball_table
from models.conformal import fit_predict_conformal_baselines
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import fit_eb_hurdle, predict_eb_hurdle
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

spec = importlib.util.spec_from_file_location("ldl", ROOT / "scripts" / "analysis" / "leverage_dual_lambda.py")
ldl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ldl)

OUT = ROOT / "outputs" / date.today().isoformat()
TMP = OUT / "coldstart_tmp"
TMP.mkdir(parents=True, exist_ok=True)
VENV_PY = ROOT / "external/TweedieGP/.venv/Scripts/python.exe"
RUNNER = ROOT / "scripts/integrity/h2_tweediegp_full.py"
Q5 = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = [f"q_{q}" for q in Q5]
CONFIG = {"online_retail": ("global", 0.95, "D"), "m5": ("mixture", 0.95, "D"),
          "auto": ("global", 0.99, "MS"), "carparts": ("global", 0.90, "MS"), "raf": ("mixture", 0.997, "MS")}
LEVELS = {"online_retail": [14, 28, 56, None], "m5": [28, 56, 112, 224, 448, None],
          "auto": [6, 12, None], "carparts": [6, 12, 24, None], "raf": [6, 12, 24, 48, None]}


def log(m):
    print(f"[c1] {m}", flush=True)


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


def truncate(init: pd.DataFrame, L):
    if L is None:
        return init
    d = init.sort_values(["unique_id", "ds"])
    return d.groupby("unique_id", group_keys=False).tail(L)


def score(pred: pd.DataFrame, model: str, ev: pd.DataFrame, init_full: pd.DataFrame):
    pred = pred.copy(); pred["unique_id"] = pred["unique_id"].astype(str); pred["ds"] = pd.to_datetime(pred["ds"])
    pred = _enforce_monotonic_quantiles(pred, quantiles=Q5)
    pred["model"] = model
    merged = ev.merge(pred[["model", "unique_id", "ds"] + QCOLS], on=["unique_id", "ds"], how="inner")
    spl = _scaled_pinball_table(merged, init_set=init_full, quantiles=Q5)
    spl = spl[spl["model"] == model]
    return (float(spl["scaled_pinball"].mean()),
            float(spl.loc[np.isclose(spl["quantile"].astype(float), 0.9), "scaled_pinball"].iloc[0]),
            int(merged["unique_id"].nunique()))


def run(panel):
    t0 = time.perf_counter()
    structure, w, freq = CONFIG[panel]
    init_full, ev = load_panel(panel)
    init_full = init_full.copy(); init_full["unique_id"] = init_full["unique_id"].astype(str)
    ev = ev.copy(); ev["unique_id"] = ev["unique_id"].astype(str); ev["ds"] = pd.to_datetime(ev["ds"])
    ev3 = ev[["unique_id", "ds", "y"]]
    rows, failures = [], []
    done = set()
    for d in sorted((ROOT / "outputs").glob("2026-09-*"), reverse=True):   # resume
        f = d / f"coldstart_{panel}.csv"
        if f.exists():
            prev = pd.read_csv(f)
            prev["L"] = prev["L"].astype(str)
            done = set(prev.loc[prev["model"] == "TweedieGP", "L"])
            rows = prev[prev["L"].isin(done)].to_dict("records")
            log(f"{panel}: resuming, levels already complete: {sorted(done)}")
            break
    for L in LEVELS[panel]:
        Ltag = "full" if L is None else str(L)
        if Ltag in done:
            continue
        init = truncate(init_full, L)
        n_med = int(init.groupby("unique_id").size().median())
        lam = ldl.dual_lambda(init, w)
        base = {"panel": panel, "L": Ltag, "median_len": n_med,
                "median_lambda_occ": round(lam["median_lambda_occ"], 4),
                "median_lambda_size": round(lam["median_lambda_size"], 4)}
        # EBB
        tb = time.perf_counter()
        labels = mixture_group_labels(init, k=0, fit_discount=w).labels if structure == "mixture" else None
        if labels is not None:
            labels.index = labels.index.astype(str)
        params = fit_eb_hurdle(init, group_labels=labels, item_variance_mode="conjugate",
                            item_variance_shrink_strength=20.0, fit_discount=w,
                            bootstrap_draws=20, bootstrap_seed=42)
        pred = predict_eb_hurdle(params, ev, quantiles=Q5, include_hyper_uncertainty=True)
        m_, q_, n_ = score(pred, "EBB", ev3, init_full)
        rows.append({**base, "model": "EBB", "spl_mean": round(m_, 4), "spl_q90": round(q_, 4), "n_series": n_,
                     "wall_s": round(time.perf_counter() - tb, 1)})
        log(f"{panel} L={Ltag} (median len {n_med}, lam+ {lam['median_lambda_size']:.3f}) EBB {m_:.4f}")
        # Zero
        z = ev3[["unique_id", "ds"]].copy()
        for c in QCOLS:
            z[c] = 0.0
        m_, q_, n_ = score(z, "Zero", ev3, init_full)
        rows.append({**base, "model": "Zero", "spl_mean": round(m_, 4), "spl_q90": round(q_, 4), "n_series": n_, "wall_s": 0.0})
        # CP wrappers
        tb = time.perf_counter()
        try:
            cq = fit_predict_conformal_baselines(init, ev, quantiles=Q5, cal_ratio=0.2, freq=freq)
            for m in ("CP-ADIDA", "CP-CrostonClassic"):
                sub = cq[cq["model"] == m]
                if sub.empty:
                    failures.append(f"L={Ltag}: {m} empty"); continue
                m_, q_, n_ = score(sub, m, ev3, init_full)
                rows.append({**base, "model": {"CP-CrostonClassic": "CP-Croston"}.get(m, m), "spl_mean": round(m_, 4),
                             "spl_q90": round(q_, 4), "n_series": n_, "wall_s": round((time.perf_counter() - tb) / 2, 1)})
                log(f"{panel} L={Ltag} {m} {m_:.4f}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"L={Ltag}: conformal {e!r}"); log(f"conformal failed at L={Ltag}: {e!r}")
        # TweedieGP
        tb = time.perf_counter()
        in_path, out_path = TMP / f"{panel}_L{Ltag}_in.csv", TMP / f"{panel}_L{Ltag}_out.csv"
        a = init[["unique_id", "ds", "y"]].copy(); a["is_train"] = 1
        b = ev3.copy(); b["is_train"] = 0
        pd.concat([a, b], ignore_index=True).to_csv(in_path, index=False)
        r = subprocess.run([str(VENV_PY), str(RUNNER), panel, str(in_path), str(out_path)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not out_path.exists():
            failures.append(f"L={Ltag}: TweedieGP rc={r.returncode} {r.stderr[-300:]}")
            log(f"TweedieGP failed at L={Ltag}")
        else:
            q = pd.read_csv(out_path, dtype={"unique_id": str})
            m_, q_, n_ = score(q, "TweedieGP", ev3, init_full)
            rows.append({**base, "model": "TweedieGP", "spl_mean": round(m_, 4), "spl_q90": round(q_, 4), "n_series": n_,
                         "wall_s": round(time.perf_counter() - tb, 1), "tg_fallback_zero": int(q.get("fallback_zero", pd.Series([0])).sum())})
            log(f"{panel} L={Ltag} TweedieGP {m_:.4f} ({(time.perf_counter()-tb)/60:.1f} min)")
        pd.DataFrame(rows).to_csv(OUT / f"coldstart_{panel}.csv", index=False)
    meta = {"task": f"c1_coldstart_{panel}", "panel": panel, "design": "docs/DESIGN_coldstart.md",
            "levels": [("full" if L is None else L) for L in LEVELS[panel]], "ebb_config": [structure, w],
            "scale": "naive scale from the FULL initialization window", "failures": failures,
            "wall_clock_total_s": round(time.perf_counter() - t0, 1),
            "platform": platform.platform(), "python": sys.version.split()[0], "seed": 42}
    (OUT / f"c1_run_meta_{panel}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["auto", "carparts", "raf", "online_retail", "m5"]):
        run(p)
