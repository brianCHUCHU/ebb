"""W1: TweedieGP under the strict walk-forward protocols.

Pre-registered protocol (fixed before any result is seen):
  carparts / auto / raf : monthly blocks, TweedieGP refit at every block
  online_retail         : seven-day blocks, refit every 4 blocks (the cadence
                          already used for AutoARIMA / AutoTheta / iETS on
                          this panel); within a refit chunk it forecasts the
                          concatenated 4-block target multi-step ahead
Each refit uses the authors' released defaults via the venv runner
(h2_tweediegp_full.py) on a temporary panel file (history = is_train 1,
block target = is_train 0). Scoring uses the same pipeline functions as
the published walk-forward tables.

Outputs (outputs/<date>/): wf_tweediegp_<panel>.csv (tidy metrics),
wf_tweediegp_quantiles_<panel>.csv, w1_run_meta_<panel>.json

Usage: py scripts/integrity/w1_wf_tweediegp.py <panel>
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

from data_loading import (load_generic_long, load_online_retail,
                          preprocess_online_retail, train_eval_split_fixed_origin,
                          train_eval_split_last_h)
from experiments.protocols import iter_walk_forward_frames
from experiments.run_prob import _coverage_summary, _enforce_monotonic_quantiles, _qcols, _scaled_pinball_table

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
TMP = OUT / "wf_tmp"
TMP.mkdir(exist_ok=True)
VENV_PY = ROOT / "external/TweedieGP/.venv/Scripts/python.exe"
RUNNER = ROOT / "scripts/integrity/h2_tweediegp_full.py"
Q5 = [0.1, 0.25, 0.5, 0.75, 0.9]
QCOLS = _qcols(Q5)
STEP = {"carparts": 1, "auto": 1, "raf": 1, "online_retail": 7}
REFIT_EVERY = {"carparts": 1, "auto": 1, "raf": 1, "online_retail": 4}
H = {"auto": 6, "carparts": 6, "raf": 12}


def load_panel(name):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data/online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    return train_eval_split_last_h(load_generic_long(ROOT / f"data/{name}_long.csv"), h=H[name])


def main():
    panel = sys.argv[1]
    t0 = time.perf_counter()
    init_set, eval_set = load_panel(panel)
    frames = list(iter_walk_forward_frames(init_set, eval_set, step_size=STEP[panel]))
    k = REFIT_EVERY[panel]
    all_q = []
    n_refits = 0
    for ci in range(0, len(frames), k):
        chunk = frames[ci:ci + k]
        history = chunk[0].history
        target = pd.concat([f.target for f in chunk], ignore_index=True)
        uids = target["unique_id"].unique()
        hist = history[history["unique_id"].isin(uids)][["unique_id", "ds", "y"]].copy()
        hist["is_train"] = 1
        tg = target[["unique_id", "ds", "y"]].copy()
        tg["is_train"] = 0
        pan = pd.concat([hist, tg], ignore_index=True)
        pan["unique_id"] = pan["unique_id"].astype(str)
        in_path = TMP / f"{panel}_refit{n_refits:03d}_in.csv"
        out_path = TMP / f"{panel}_refit{n_refits:03d}_out.csv"
        pan.to_csv(in_path, index=False)
        r = subprocess.run([str(VENV_PY), str(RUNNER), panel, str(in_path), str(out_path)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-800:], flush=True)
            raise SystemExit(f"runner failed at refit {n_refits}")
        q = pd.read_csv(out_path, dtype={"unique_id": str})
        q["model"] = "TweedieGP"
        all_q.append(q[["model", "unique_id", "ds"] + QCOLS + ["wall_seconds", "fallback_zero"]])
        n_refits += 1
        print(f"[{panel}] refit {n_refits}/{int(np.ceil(len(frames)/k))} done "
              f"({(time.perf_counter()-t0)/60:.0f} min)", flush=True)

    aq = pd.concat(all_q, ignore_index=True)
    aq["ds"] = pd.to_datetime(aq["ds"])
    for c in QCOLS:
        aq[c] = np.maximum(aq[c].astype(float), 0.0)
    aq = _enforce_monotonic_quantiles(aq, quantiles=Q5)
    aq.to_csv(OUT / f"wf_tweediegp_quantiles_{panel}.csv", index=False)

    ev = eval_set[["unique_id", "ds", "y"]].copy()
    ev["ds"] = pd.to_datetime(ev["ds"])
    ev["unique_id"] = ev["unique_id"].astype(str)
    merged = ev.merge(aq, on=["unique_id", "ds"], how="inner")
    spl = _scaled_pinball_table(merged, init_set=init_set, quantiles=Q5)
    spl = spl[spl["model"] == "TweedieGP"]
    cov = _coverage_summary(merged).iloc[0]
    cov_pos = _coverage_summary(merged[merged["y"] > 0].copy()).iloc[0]
    wall = time.perf_counter() - t0
    rows = [{"dataset": panel, "protocol": "walk_forward", "model": "TweedieGP",
             "metric": "scaled_pinball", "quantile": r["quantile"], "value": float(r["scaled_pinball"]),
             "refit_interval": k} for _, r in spl.iterrows()]
    rows.append({"dataset": panel, "protocol": "walk_forward", "model": "TweedieGP",
                 "metric": "scaled_pinball", "quantile": "mean",
                 "value": float(spl["scaled_pinball"].mean()), "refit_interval": k})
    for name, val in (("coverage80", cov["Coverage@80"]), ("coverage80_positive", cov_pos["Coverage@80"]),
                      ("aiw80", cov["AIW@80"]), ("wall_clock_s", wall)):
        rows.append({"dataset": panel, "protocol": "walk_forward", "model": "TweedieGP",
                     "metric": name, "quantile": "", "value": float(val), "refit_interval": k})
    pd.DataFrame(rows).to_csv(OUT / f"wf_tweediegp_{panel}.csv", index=False)
    meta = {"task": f"w1_wf_tweediegp_{panel}", "panel": panel, "blocks": len(frames),
            "refit_every": k, "n_refits": n_refits, "wall_clock_s": round(wall, 1),
            "config": "released defaults; inducing T>200->200/log else all points; 12 workers",
            "platform": platform.platform(), "python": sys.version.split()[0], "seed": 42}
    (OUT / f"w1_run_meta_{panel}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{panel}] TweedieGP wf mean SPL {float(spl['scaled_pinball'].mean()):.4f} "
          f"in {wall/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
