"""Spec Task 3 runner: official TweedieGP under OUR fixed-origin protocol.

Runs inside external/TweedieGP/.venv (py3.11 + torch 2.2.0 + gpytorch 1.11).
Uses the authors' own model class (external/TweedieGP/src) with the paper's
TweedieGP configuration (--likelihood tweedie --scaling median-demand, RBF
kernel, defaults otherwise), on OUR panel CSVs and OUR last-h split, so the
comparison is protocol-identical to the paper's tables.

Deviation log (mirrored in NOTES.md):
- The authors drop series whose TRAINING window is all zeros; our protocol
  forecasts every series, so those get an all-zero forecast here (flagged in
  the output as fallback_zero=1).
- The authors seed once globally then loop; we run series in parallel worker
  processes, seeding 42 in each worker. With 50k predictive samples the
  sampling noise on quantiles is negligible.

Output: outputs/aistats2027_rebuild/tweediegp_quantiles_{dataset}.csv
  columns: unique_id, ds, mean, q_0.1 ... q_0.99, train_seconds, fallback_zero

Usage (from repo root, venv python):
  external/TweedieGP/.venv/Scripts/python.exe scripts/rebuild/t3_tweediegp_runner.py auto --workers 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "TweedieGP" / "src"))

import numpy as np
import pandas as pd

H = {"auto": 6, "carparts": 6, "raf": 12}
PERIOD = 12.0  # monthly, matching the authors' period map
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
SEED = 42


def _fit_one(args):
    """Fit TweedieGP on one series; returns dict of forecasts."""
    uid, y_all, ds_tail, h = args
    import random

    import torch

    torch.set_num_threads(1)
    random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        torch.use_deterministic_algorithms(mode=True)
    except Exception:
        pass

    from tweediegp.intermittent_gp import intermittentGP

    y_all = np.asarray(y_all, dtype=np.float32)
    T = len(y_all) - h
    train_y = torch.tensor(y_all[:T], dtype=torch.float32)
    x = torch.arange(len(y_all), dtype=torch.float32) / PERIOD
    train_x, test_x = x[:T], x[-h:]

    base = {"unique_id": uid, "ds": list(ds_tail)}
    if float(train_y.abs().sum()) == 0.0:
        out = dict(base, mean=[0.0] * h, fallback_zero=1, train_seconds=0.0)
        for q in QUANTILES:
            out[f"q_{q}"] = [0.0] * h
        return out

    t0 = time.time()
    last_err = "unknown"
    for attempt in range(3):
        try:
            model = intermittentGP(
                "tweedie", None, "median-demand", 1.0,
                None, None, 5e4, 100, 25, 1, 1e-4, False,
            )
            model.build(train_x=train_x, train_y=train_y)
            model.fit(train_x=train_x, train_y=train_y)
            if model._iter < model.min_iter:
                last_err = f"stopped at iter {model._iter} < min_iter"
                continue
            mean_pred, samples = model.predict(test_x)
            qs = torch.quantile(
                samples, torch.tensor(QUANTILES, dtype=samples.dtype), dim=0)
            out = dict(
                base,
                mean=[float(v) for v in mean_pred.detach().numpy()],
                fallback_zero=0,
                train_seconds=time.time() - t0,
            )
            for k, q in enumerate(QUANTILES):
                out[f"q_{q}"] = [float(v) for v in qs[k].detach().numpy()]
            return out
        except Exception as e:  # noqa: BLE001 - per-series retry, recorded
            last_err = repr(e)
            continue
    out = dict(base, mean=[np.nan] * h, fallback_zero=0,
               train_seconds=time.time() - t0, error=last_err)
    for q in QUANTILES:
        out[f"q_{q}"] = [np.nan] * h
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(H))
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N series")
    args = ap.parse_args()

    h = H[args.dataset]
    df = pd.read_csv(ROOT / "data" / f"{args.dataset}_long.csv")
    df["unique_id"] = df["unique_id"].astype(str)
    df = df.sort_values(["unique_id", "ds"])
    jobs = []
    for uid, g in df.groupby("unique_id", sort=True):
        y = g["y"].to_numpy(dtype=np.float32)
        ds_tail = g["ds"].astype(str).tolist()[-h:]
        jobs.append((uid, y, ds_tail, h))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[t3] {args.dataset}: {len(jobs)} series, h={h}, "
          f"workers={args.workers}", flush=True)

    out_dir = ROOT / "outputs" / "aistats2027_rebuild"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tweediegp_quantiles_{args.dataset}.csv"

    rows: list[dict] = []
    t0 = time.time()
    done = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_fit_one, jobs, chunksize=8):
            n = len(res["ds"])
            for i in range(n):
                row = {
                    "unique_id": res["unique_id"], "ds": res["ds"][i],
                    "mean": res["mean"][i],
                    "fallback_zero": res["fallback_zero"],
                    "train_seconds": res["train_seconds"],
                }
                for q in QUANTILES:
                    row[f"q_{q}"] = res[f"q_{q}"][i]
                rows.append(row)
            if "error" in res:
                errors += 1
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                proj = el / done * len(jobs)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f"[t3] {done}/{len(jobs)} elapsed {el/60:.1f}m "
                      f"projected {proj/60:.1f}m errors={errors}", flush=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    meta = {
        "task": "t3_tweediegp", "dataset": args.dataset,
        "n_series": len(jobs), "errors": errors,
        "wall_clock_sec": time.time() - t0, "seed": SEED, "gpu": False,
        "config": "likelihood=tweedie scaling=median-demand kernel=RBF "
                  "max_iter=100 n_samples=5e4 (paper defaults)",
        "source": "official repo StefanoDamato/TweedieGP",
    }
    (out_dir / f"t3_run_meta_{args.dataset}.json").write_text(
        json.dumps(meta, indent=2))
    print(f"[t3] wrote {out_path} ({errors} errored series) "
          f"in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
