"""Gate 1 (probe step): TweedieGP wall-clock on 12 daily series per panel.

Runs inside external/TweedieGP/.venv (py3.11 + torch). Configuration is
identical to the monthly runs in scripts/rebuild/t3_tweediegp_runner.py
(tweedie likelihood, median-demand scaling, RBF/default kernel, 5e4
predictive samples, seed 42, single torch thread), with the authors' daily
period (365, from tweediegp/main.py). Series run SEQUENTIALLY so each
wall-clock is a clean single-series measurement; full-panel extrapolations
are reported both sequential and at the 16 workers used by the monthly runs.

Output: outputs/<date>/tweediegp_scaling_probe.csv
  columns: panel, series_id, n_obs, n_train, n_eval, wall_clock_s, fallback_zero

Usage:
  external/TweedieGP/.venv/Scripts/python.exe scripts/integrity/g1_tweediegp_probe.py [panel]
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
PERIOD_DAILY = 365.0  # authors' period map: {'D': 365, 'W': 52.1, 'M': 12}
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
SEED = 42
WORKERS_REFERENCE = 16


def fit_one(y_all: np.ndarray, h: int) -> tuple[float, int]:
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
    x = torch.arange(len(y_all), dtype=torch.float32) / PERIOD_DAILY
    train_x, test_x = x[:T], x[-h:]
    if float(train_y.abs().sum()) == 0.0:
        return 0.0, "zero_fallback"
    t0 = time.time()
    last_err = "unknown"
    for _attempt in range(3):  # same retry budget as the t3 runner
        try:
            model = intermittentGP(
                "tweedie", None, "median-demand", 1.0,
                None, None, 5e4, 100, 25, 1, 1e-4, False,
            )
            model.build(train_x=train_x, train_y=train_y)
            model.fit(train_x=train_x, train_y=train_y)
            mean_pred, samples = model.predict(test_x)
            _ = np.quantile(samples.detach().numpy(), QUANTILES, axis=0)
            return time.time() - t0, "ok"
        except Exception as e:  # noqa: BLE001 - recorded, not silent
            last_err = f"{type(e).__name__}: {e}"[:160]
    return time.time() - t0, f"FAILED {last_err}"


def main() -> None:
    panels = sys.argv[1:] or ["online_retail", "m5"]
    inp = pd.read_csv(OUT / "tweediegp_probe_input.csv")
    sizes = pd.read_csv(OUT / "tweediegp_probe_panelsizes.csv").set_index("panel")["n_series"]
    rows = []
    for panel in panels:
        sub = inp[inp["panel"] == panel]
        for uid, grp in sub.groupby("unique_id", sort=True):
            grp = grp.sort_values(["is_train", "ds"], ascending=[False, True])
            y_all = grp["y"].to_numpy(dtype=float)
            h = int((grp["is_train"] == 0).sum())
            wall, status = fit_one(y_all, h)
            rows.append({"panel": panel, "series_id": uid, "n_obs": len(y_all),
                         "n_train": len(y_all) - h, "n_eval": h,
                         "wall_clock_s": round(wall, 2), "status": status})
            print(f"[{panel}] {uid}: n={len(y_all)} (train {len(y_all)-h}, "
                  f"h {h}) wall={wall:.1f}s status={status}", flush=True)
        dfp = pd.DataFrame([r for r in rows if r["panel"] == panel and r["status"] == "ok"])
        if len(dfp):
            med, mx = dfp.wall_clock_s.median(), dfp.wall_clock_s.max()
            n = int(sizes[panel])
            print(f"[{panel}] median={med:.1f}s max={mx:.1f}s | full panel n={n}: "
                  f"sequential ~{med*n/3600:.1f}h (median-based), "
                  f"at {WORKERS_REFERENCE} workers ~{med*n/WORKERS_REFERENCE/3600:.2f}h "
                  f"(worst-case {mx*n/WORKERS_REFERENCE/3600:.2f}h)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "tweediegp_scaling_probe.csv", index=False)
    print("wrote tweediegp_scaling_probe.csv")


if __name__ == "__main__":
    main()
