"""Task 3: instrumented wall-clock for every Table-2 method on Carparts,
one machine, one session. Configurations unchanged; timing only.

REMIX time INCLUDES the 24-candidate joint selection (head-only labels +
select_pooling_and_discount) plus the final fit (B=20) and five-quantile
prediction. TSB-tuned includes its 25-candidate selection plus refit and
prediction. CP wrappers share one residual-pool fit; the call is timed once
and split across the five wrappers (noted). AutoARIMA/AutoTheta share one
StatsForecast call, split across the two (noted). TweedieGP is the official
implementation run via its own venv at 16 workers (t3 runner). DeepState /
Chronos-Bolt / DRP rows are recorded from their run metadata, flagged.

Output: outputs/<date>/efficiency_carparts.csv + g3_run_meta.json

Usage: py scripts/integrity/g3_efficiency_carparts.py [--skip-iets] [--skip-tweediegp]
"""

from __future__ import annotations

import argparse
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

from data_loading import load_generic_long, train_eval_split_last_h
from experiments.run_prob import _build_regime_group_labels, _predict_non_hb_prob_models_once
from models.conformal import fit_predict_conformal_baselines
from models.mixture_pooling import mixture_group_labels
from models.tsb_hb import (
    _split_init_head_tail, fit_tsb_hb, predict_tsb_hb, select_pooling_and_discount,
)
from models.tsb_tuned import fit_predict_tsb_tuned
from models.tweedie_baseline import fit_predict_tweedie_prob_panel

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
IETS_RSCRIPT = r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe"
W_STAR = 0.90  # audited Carparts configuration (global, 0.90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-iets", action="store_true")
    ap.add_argument("--skip-tweediegp", action="store_true")
    args = ap.parse_args()

    df = load_generic_long(ROOT / "data" / "carparts_long.csv")
    init_set, eval_set = train_eval_split_last_h(df, h=6)
    n = int(init_set["unique_id"].nunique())
    rows = []

    def add(method, total, includes_sel, hw, runtime, gpu, bitrep, notes, n_series=n):
        rows.append({
            "method": method, "wall_clock_total_s": round(float(total), 1),
            "s_per_series": round(float(total) / n_series, 3),
            "includes_selection": includes_sel, "hardware": hw, "runtime": runtime,
            "gpu_used": gpu, "bit_reproducible": bitrep, "n_series": n_series,
            "notes": notes,
        })
        print(f"[{method}] {total:.1f}s", flush=True)

    # --- REMIX: joint selection + final fit + prediction ---
    t0 = time.perf_counter()
    head, _ = _split_init_head_tail(init_set, val_ratio=0.2)
    mix = mixture_group_labels(head, k=0)
    candidates = {"global": None, "taxonomy": _build_regime_group_labels(head),
                  "mixture": mix.labels}
    sel_name, sel_w, _ = select_pooling_and_discount(
        init_set, candidates, item_variance_mode="conjugate",
        item_variance_shrink_strength=20.0)
    params = fit_tsb_hb(init_set, group_labels=None,
                        item_variance_mode="conjugate",
                        item_variance_shrink_strength=20.0,
                        fit_discount=W_STAR, bootstrap_draws=20, bootstrap_seed=42)
    predict_tsb_hb(params, eval_set, quantiles=None)
    predict_tsb_hb(params, eval_set, quantiles=QUANTILES, include_hyper_uncertainty=True)
    add("REMIX", time.perf_counter() - t0, True, "CPU", "Python", False, "Yes",
        f"24-candidate joint selection (selected {sel_name}, w={sel_w}) + "
        "final EB fit (B=20) + point + 5-quantile prediction; single process")

    # --- TSB-tuned: 25-candidate selection + refit + prediction ---
    t0 = time.perf_counter()
    fit_predict_tsb_tuned(init_set, eval_set, freq="MS")
    add("TSB-tuned", time.perf_counter() - t0, True, "CPU", "Python", False, "No",
        "25-candidate grid on the identical 80/20 chronological split + refit "
        "+ point prediction (point-only control; no predictive distribution)")

    # --- AutoARIMA + AutoTheta (one StatsForecast call, split by 2) ---
    t0 = time.perf_counter()
    _predict_non_hb_prob_models_once(init_set, eval_set, quantiles=QUANTILES,
                                     baseline_mode="paper", freq="MS", season_length=12)
    aa = time.perf_counter() - t0
    for m in ("AutoARIMA", "AutoTheta"):
        add(m, aa / 2, False, "CPU", "Python", False, "No",
            "shared StatsForecast call timed once and split across the two models")

    # --- five CP wrappers (one shared call, split by 5) ---
    t0 = time.perf_counter()
    fit_predict_conformal_baselines(init_set, eval_set, quantiles=QUANTILES,
                                    cal_ratio=0.2, freq="MS")
    cp = time.perf_counter() - t0
    for m in ("CP-Croston", "CP-SBA", "CP-TSB", "CP-ADIDA", "CP-IMAPA"):
        add(m, cp / 5, False, "CPU", "Python", False, "No",
            "five wrappers share one residual-pool fit; call timed once, split by five")

    # --- Tweedie-GLM ---
    t0 = time.perf_counter()
    fit_predict_tweedie_prob_panel(init_set, eval_set, quantiles=QUANTILES,
                                   lags=6, power=1.5, alpha=0.1, max_iter=1000)
    add("Tweedie-GLM", time.perf_counter() - t0, False, "CPU", "Python", False, "No",
        "per-series autoregressive GLM, 6 lags (monthly)")

    # --- iETS (R) ---
    if not args.skip_iets:
        from models.iets_baseline import fit_predict_iets_prob_panel
        t0 = time.perf_counter()
        try:
            fit_predict_iets_prob_panel(init_set, eval_set, quantiles=QUANTILES,
                                        rscript=IETS_RSCRIPT, occurrence="auto",
                                        timeout_seconds=None,
                                        per_series_timeout_seconds=10, n_jobs=20,
                                        cache_path=None)
            add("iETS", time.perf_counter() - t0, False, "CPU", "R", False, "No",
                "smooth::adam via Rscript, 20 parallel jobs, 10 s/series cap")
        except Exception as e:  # noqa: BLE001
            add("iETS", np.nan, False, "CPU", "R", False, "No", f"FAILED: {e!r}"[:150])

    # --- TweedieGP (official venv, 16 workers) ---
    if not args.skip_tweediegp:
        import shutil
        rb = ROOT / "outputs" / "aistats2027_rebuild"
        backups = {}
        for fn in ("tweediegp_quantiles_carparts.csv", "t3_run_meta_carparts.json"):
            p = rb / fn
            if p.exists():
                backups[fn] = p.read_bytes()
        t0 = time.perf_counter()
        r = subprocess.run(
            [str(ROOT / "external/TweedieGP/.venv/Scripts/python.exe"),
             str(ROOT / "scripts/rebuild/t3_tweediegp_runner.py"), "carparts",
             "--workers", "16"],
            cwd=ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        for fn, data in backups.items():
            new = rb / fn
            if new.exists():
                shutil.move(str(new), str(OUT / f"timed_{fn}"))
            (rb / fn).write_bytes(data)  # restore the archived artifact
        note = "official implementation, exact per-series GP, 16 worker processes"
        if r.returncode != 0:
            note += f"; RUNNER EXIT {r.returncode}: {r.stderr[-120:]}"
        add("TweedieGP", wall, False, "CPU", "Python/torch", False, "No", note)

    # --- Zero ---
    add("Zero", 0.0, False, "CPU", "Python", False, "Yes", "constant zero forecast")

    # --- recorded rows from prior run metadata (not re-timed) ---
    add("DeepState", 163.0 + 29.0, False, "CPU", "Python/MXNet", False, "No",
        "recorded from t4_run_meta_carparts.json (train 163 s + predict); "
        "GluonTS DeepState, full Carparts, not re-timed this session")
    add("Chronos-Bolt-small", 8.3, False, "CPU", "Python/torch", False, "No",
        "recorded from t7_run_meta_carparts500.json; zero-shot inference, "
        "500-series subsample, not re-timed this session", n_series=500)
    add("DRP (Deep Renewal)", 1200.0, False, "CPU", "Python/MXNet", False, "No",
        "~20 min recorded for a 500-series Online Retail subsample (legacy run); "
        "no Carparts run exists; training diverges on rebuild (see FAILURES)",
        n_series=500)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "efficiency_carparts.csv", index=False)
    print(out[["method", "wall_clock_total_s", "s_per_series"]].to_string(index=False))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {"task": "g3_efficiency_carparts", "commit": commit,
            "python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "numpy": np.__version__,
            "pandas": pd.__version__, "n_series": n,
            "note": "single session, same machine; recorded rows flagged in notes"}
    (OUT / "g3_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote efficiency_carparts.csv")


if __name__ == "__main__":
    main()
