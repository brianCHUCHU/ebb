"""Task 1: TweedieGP configuration alignment against Damato et al. (2025).

Part A (M5, 3 series): build with the paper's inducing rule (T>200 -> 200
inducing points, init 'log' i.e. p(i) proportional to log(1+i/T)) and verify
the variational strategy actually holds a 200-point inducing set; record
status / wall / iterations. Expectation: no Cholesky NaN.

Part B (Carparts, 12 series): decompose build/fit/predict wall-clock and
iteration counts under three environment variants to locate the cost gap
vs the paper's 0.24 +/- 0.12 s/series (M3 MacBook Pro):
    A_current   = torch.use_deterministic_algorithms(True) + 1 thread
    B_nodet     = deterministic OFF + 1 thread
    C_nodet_mt  = deterministic OFF + default threads
Model configuration itself is the authors' (T=45 <= 200 -> all training
points as inducing set, identical to their rule; no change needed).

Outputs (outputs/<date>/): tweediegp_alignment_m5.csv,
tweediegp_alignment_carparts.csv

Usage:
  external/TweedieGP/.venv/Scripts/python.exe scripts/integrity/h1_tweediegp_alignment.py
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "TweedieGP" / "src"))

import numpy as np
import pandas as pd

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
IN = ROOT / "outputs" / "2026-08-07" / "tweediegp_probe_input.csv"
PERIOD_DAILY = 365.0
PERIOD_MONTHLY = 12.0
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
SEED = 42


def run_one(y_all, h, period, num_inducing, deterministic, single_thread):
    import random

    import torch

    if single_thread:
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(torch.get_num_interop_threads() * 2 or 8)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(mode=True)
        except Exception:
            pass
    else:
        try:
            torch.use_deterministic_algorithms(mode=False)
        except Exception:
            pass
    from tweediegp.intermittent_gp import intermittentGP

    y_all = np.asarray(y_all, dtype=np.float32)
    T = len(y_all) - h
    train_y = torch.tensor(y_all[:T], dtype=torch.float32)
    x = torch.arange(len(y_all), dtype=torch.float32) / period
    train_x, test_x = x[:T], x[-h:]
    if float(train_y.abs().sum()) == 0.0:
        return {"status": "zero_fallback"}

    res = {}
    last_err = "unknown"
    for _attempt in range(3):
        try:
            model = intermittentGP(
                "tweedie", None, "median-demand", 1.0,
                num_inducing, "log" if num_inducing else None,
                5e4, 100, 25, 1, 1e-4, False,
            )
            t0 = time.time()
            model.build(train_x=train_x, train_y=train_y)
            t1 = time.time()
            try:
                ip = model._gp.variational_strategy.inducing_points
                res["inducing_shape"] = "x".join(str(s) for s in ip.shape)
            except Exception:
                res["inducing_shape"] = "n/a"
            model.fit(train_x=train_x, train_y=train_y)
            t2 = time.time()
            mean_pred, samples = model.predict(test_x)
            _ = np.quantile(samples.detach().numpy(), QUANTILES, axis=0)
            t3 = time.time()
            res.update({"status": "ok", "build_s": round(t1 - t0, 3),
                        "fit_s": round(t2 - t1, 3), "predict_s": round(t3 - t2, 3),
                        "total_s": round(t3 - t0, 3),
                        "iterations": int(model._iter) + 1,
                        "stop_reason": str(model._stop)})
            return res
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"[:140]
    res["status"] = f"FAILED {last_err}"
    return res


def main() -> None:
    inp = pd.read_csv(IN)

    # ---- Part A: M5 with the paper's inducing rule ----
    rows = []
    m5 = inp[inp["panel"] == "m5"]
    for uid in sorted(m5["unique_id"].unique())[:3]:
        grp = m5[m5["unique_id"] == uid].sort_values(
            ["is_train", "ds"], ascending=[False, True])
        y = grp["y"].to_numpy(dtype=float)
        h = int((grp["is_train"] == 0).sum())
        r = run_one(y, h, PERIOD_DAILY, num_inducing=200,
                    deterministic=True, single_thread=True)
        r.update({"series_id": uid, "n_train": len(y) - h, "n_eval": h})
        rows.append(r)
        print("[m5 m=200]", uid, r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "tweediegp_alignment_m5.csv", index=False)

    # ---- Part B: Carparts variants ----
    df = pd.read_csv(ROOT / "data" / "carparts_long.csv")
    uids = np.array(sorted(df["unique_id"].astype(str).unique()))
    rng = np.random.default_rng(42)
    chosen = uids[np.sort(rng.choice(len(uids), 12, replace=False))]
    rows = []
    for variant, det, st in (("A_current", True, True),
                             ("B_nodet", False, True),
                             ("C_nodet_mt", False, False)):
        for uid in chosen:
            g = df[df["unique_id"].astype(str) == uid].sort_values("ds")
            y = g["y"].to_numpy(dtype=float)
            r = run_one(y, 6, PERIOD_MONTHLY, num_inducing=None,
                        deterministic=det, single_thread=st)
            r.update({"variant": variant, "series_id": uid})
            rows.append(r)
        ok = [r for r in rows if r["variant"] == variant and r.get("status") == "ok"]
        med = np.median([r["total_s"] for r in ok]) if ok else float("nan")
        print(f"[carparts {variant}] median total {med:.2f}s over {len(ok)} ok "
              f"(fit {np.median([r['fit_s'] for r in ok]):.2f}, "
              f"predict {np.median([r['predict_s'] for r in ok]):.3f}, "
              f"iters median {np.median([r['iterations'] for r in ok]):.0f})",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT / "tweediegp_alignment_carparts.csv", index=False)
    print("wrote tweediegp_alignment_{m5,carparts}.csv")


if __name__ == "__main__":
    main()
