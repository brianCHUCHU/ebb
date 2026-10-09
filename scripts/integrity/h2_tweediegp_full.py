"""Task 2: TweedieGP full runs on Online Retail and M5, released code as-is.

Configuration = authors' released defaults (train_longer active), with the
paper's inducing rule applied per series: T <= 200 -> all training points
(num_inducing_points=None); T > 200 -> 200 inducing points, init 'log'.
Daily period 365 (authors' map). Seed 42 per worker, 3 retry attempts per
series (restart on numerical failure), zero-training-window fallback to an
all-zero forecast, flagged. 12 worker processes.

Output: outputs/<date>/tweediegp_quantiles_{panel}.csv
  columns: unique_id, ds, mean, q_0.1 ... q_0.99, train_seconds,
           wall_seconds, fallback_zero, n_attempts

Usage:
  external/TweedieGP/.venv/Scripts/python.exe scripts/integrity/h2_tweediegp_full.py online_retail
  external/TweedieGP/.venv/Scripts/python.exe scripts/integrity/h2_tweediegp_full.py m5
"""

from __future__ import annotations

import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "TweedieGP" / "src"))

import numpy as np
import pandas as pd

OUT = ROOT / "outputs" / date.today().isoformat()
PERIOD = {"online_retail": 365.0, "m5": 365.0, "carparts": 12.0, "auto": 12.0, "raf": 12.0}  # authors' map
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
SEED = 42
WORKERS = 12
INDUCING_THRESHOLD = 200


def _fit_one(args):
    uid, y_all, ds_tail, h, period = args
    import random

    import torch

    torch.set_num_threads(1)
    random.seed(SEED)
    torch.manual_seed(SEED)
    from tweediegp.intermittent_gp import intermittentGP

    y_all = np.asarray(y_all, dtype=np.float32)
    T = len(y_all) - h
    train_y = torch.tensor(y_all[:T], dtype=torch.float32)
    x = torch.arange(len(y_all), dtype=torch.float32) / period
    train_x, test_x = x[:T], x[-h:]
    num_ind = INDUCING_THRESHOLD if T > INDUCING_THRESHOLD else None
    init_ind = "log" if num_ind else None

    base = {"unique_id": uid, "ds": list(ds_tail)}
    if float(train_y.abs().sum()) == 0.0:
        out = dict(base, mean=[0.0] * h, fallback_zero=1, train_seconds=0.0,
                   wall_seconds=0.0, n_attempts=0)
        for q in QUANTILES:
            out[f"q_{q}"] = [0.0] * h
        return out

    t0 = time.time()
    last_err = "unknown"
    for attempt in range(3):
        try:
            model = intermittentGP(
                "tweedie", None, "median-demand", 1.0,
                num_ind, init_ind, 5e4, 100, 25, 1, 1e-4, False,
            )
            model.build(train_x=train_x, train_y=train_y)
            t_fit0 = time.time()
            model.fit(train_x=train_x, train_y=train_y)
            fit_s = time.time() - t_fit0
            if model._iter < model.min_iter:
                last_err = f"stopped at iter {model._iter} < min_iter"
                continue
            mean_pred, samples = model.predict(test_x)
            qs = torch.quantile(
                samples, torch.tensor(QUANTILES, dtype=samples.dtype), dim=0)
            out = dict(base,
                       mean=[float(v) for v in mean_pred.detach().numpy()],
                       fallback_zero=0, train_seconds=round(fit_s, 3),
                       wall_seconds=round(time.time() - t0, 3),
                       n_attempts=attempt + 1)
            for k, q in enumerate(QUANTILES):
                out[f"q_{q}"] = [float(v) for v in qs[k].detach().numpy()]
            return out
        except Exception as e:  # noqa: BLE001 - per-series retry, recorded
            last_err = f"{type(e).__name__}: {e}"[:120]
    out = dict(base, mean=[0.0] * h, fallback_zero=2,
               train_seconds=0.0, wall_seconds=round(time.time() - t0, 3),
               n_attempts=3, error=last_err)
    for q in QUANTILES:
        out[f"q_{q}"] = [0.0] * h
    return out


def main() -> None:
    panel = sys.argv[1]
    in_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / f"tweediegp_panel_{panel}.csv"
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT / f"tweediegp_quantiles_{panel}.csv"
    t_start = time.time()
    inp = pd.read_csv(in_path, dtype={"unique_id": str})
    inp["unique_id"] = inp["unique_id"].astype(str)
    jobs = []
    for uid, grp in inp.groupby("unique_id", sort=True):
        grp = grp.sort_values(["is_train", "ds"], ascending=[False, True])
        h = int((grp["is_train"] == 0).sum())
        jobs.append((uid, grp["y"].to_numpy(dtype=float),
                     grp.loc[grp["is_train"] == 0, "ds"].tolist(), h,
                     PERIOD[panel]))
    print(f"[{panel}] {len(jobs)} series, workers={WORKERS}", flush=True)

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(_fit_one, jobs, chunksize=8):
            rows.append(res)
            done += 1
            if done % 250 == 0:
                el = time.time() - t_start
                print(f"[{panel}] {done}/{len(jobs)} ({el/60:.0f} min, "
                      f"eta {el/done*(len(jobs)-done)/60:.0f} min)", flush=True)

    recs = []
    for r in rows:
        n = len(r["ds"])
        rec = {"unique_id": [r["unique_id"]] * n, "ds": r["ds"], "mean": r["mean"]}
        for q in QUANTILES:
            rec[f"q_{q}"] = r[f"q_{q}"]
        df = pd.DataFrame(rec)
        df["train_seconds"] = r["train_seconds"]
        df["wall_seconds"] = r["wall_seconds"]
        df["fallback_zero"] = r["fallback_zero"]
        df["n_attempts"] = r["n_attempts"]
        recs.append(df)
    out = pd.concat(recs, ignore_index=True)
    out.to_csv(out_path, index=False)

    per = pd.DataFrame([{"uid": r["unique_id"], "w": r["wall_seconds"],
                         "fb": r["fallback_zero"]} for r in rows])
    n_fail = int((per["fb"] == 2).sum())
    ok = per[per["fb"] == 0]
    meta = {
        "task": f"h2_tweediegp_full_{panel}", "panel": panel,
        "n_series": len(jobs), "n_zero_fallback": int((per["fb"] == 1).sum()),
        "n_failed_after_3_attempts": n_fail,
        "wall_clock_total_s": round(time.time() - t_start, 1),
        "per_series_wall_median_s": round(float(ok["w"].median()), 2) if len(ok) else None,
        "workers": WORKERS, "seed": SEED,
        "config": "released defaults (train_longer active); inducing: T>200 -> 200/'log', else all points; period 365",
        "platform": platform.platform(), "python": sys.version.split()[0],
    }
    (out_path.with_name(out_path.stem + "_meta.json")).write_text(json.dumps(meta, indent=2),
                                                   encoding="utf-8")
    print(f"[{panel}] done in {(time.time()-t_start)/60:.0f} min; "
          f"failed {n_fail}; wrote {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
