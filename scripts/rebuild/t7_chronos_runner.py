"""Spec Task 7: Chronos-Bolt zero-shot reference point (500-series subsamples).

Runs amazon/chronos-bolt-small (CPU, zero-shot, no fine-tune) on seed-42
500-series subsamples of Online Retail and Carparts under the fixed-origin
protocol, chaining 64-step blocks with revealed history only (the model's
maximum prediction length is 64; deviation recorded). Quantiles at the
pipeline grid {0.1..0.9} -> SPL via the existing scorer downstream.

Output: outputs/aistats2027_rebuild/chronos_quantiles_{name}.csv
Usage: tweedie venv python scripts/rebuild/t7_chronos_runner.py or500
       tweedie venv python scripts/rebuild/t7_chronos_runner.py carparts500
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "aistats2027_rebuild"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
SEED = 42
BLOCK = 64


def main() -> None:
    name = sys.argv[1]
    src = {"or500": "or1000", "carparts500": "carparts"}[name]
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small", device_map="cpu",
        torch_dtype=torch.float32)
    print(f"[t7] model loaded in {time.time()-t0:.0f}s", flush=True)

    init = pd.read_csv(OUT / f"panel_{src}_init.csv")
    ev = pd.read_csv(OUT / f"panel_{src}_eval.csv")
    for df in (init, ev):
        df["unique_id"] = df["unique_id"].astype(str)
        df["ds"] = pd.to_datetime(df["ds"])
    rng = np.random.default_rng(SEED)
    uids = np.sort(init.unique_id.unique())
    if len(uids) > 500:
        keep = set(rng.choice(uids, size=500, replace=False))
        init = init[init.unique_id.isin(keep)]
        ev = ev[ev.unique_id.isin(keep)]
    print(f"[t7] {name}: {init.unique_id.nunique()} series", flush=True)

    ev_idx = ev.sort_values(["unique_id", "ds"]).copy()
    ev_idx["k"] = ev_idx.groupby("unique_id").cumcount()
    n_blocks = int(np.ceil((ev_idx.k.max() + 1) / BLOCK))
    hist = init.copy()
    rows = []
    qlv = torch.tensor(QUANTILES)
    for b in range(n_blocks):
        tb = time.time()
        block = ev_idx[(ev_idx.k >= b * BLOCK) & (ev_idx.k < (b + 1) * BLOCK)]
        if block.empty:
            continue
        h_uids = sorted(hist.unique_id.unique())
        contexts = [torch.tensor(
            hist[hist.unique_id == u].sort_values("ds").y.to_numpy(dtype=np.float32))
            for u in h_uids]
        hmax = int(block.groupby("unique_id").size().max())
        q, _ = pipe.predict_quantiles(
            contexts, prediction_length=hmax,
            quantile_levels=QUANTILES)
        for ui, u in enumerate(h_uids):
            tgt = block[block.unique_id == u].sort_values("ds")
            for i, (_, r) in enumerate(tgt.iterrows()):
                row = {"unique_id": u, "ds": str(r.ds.date())}
                for qi, qq in enumerate(QUANTILES):
                    row[f"q_{qq}"] = float(q[ui, i, qi])
                row["mean"] = float(q[ui, i, QUANTILES.index(0.5)])
                rows.append(row)
        hist = pd.concat([hist, block[["unique_id", "ds", "y"]]],
                         ignore_index=True)
        print(f"[t7] block {b+1}/{n_blocks}: {time.time()-tb:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(OUT / f"chronos_quantiles_{name}.csv", index=False)
    (OUT / f"t7_run_meta_{name}.json").write_text(json.dumps({
        "task": "t7_chronos", "panel": name, "model": "amazon/chronos-bolt-small",
        "wall_clock_sec": time.time() - t0, "seed": SEED, "gpu": False,
        "zero_shot": True, "block_chaining": BLOCK,
        "note": "median used as point forecast; 64-step max prediction length "
                "chained with revealed history only"}, indent=2))
    print(f"[t7] wrote chronos_quantiles_{name}.csv in "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
