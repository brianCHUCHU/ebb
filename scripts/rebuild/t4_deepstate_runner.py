"""Spec Task 4: DeepState (GluonTS/MXNet) as the hierarchical prior-work arm.

Seeger et al. 2016 has no public implementation (recorded in FAILURES.md);
DeepState (Rangapuram et al. 2018, gluonts.mx.model.deepstate) is the closest
runnable published descendant of the same Amazon ISSM line: per-series linear
state-space models with shared neural parameterization, Bayesian filtering
for the states, sample-based predictive distributions. We label it as such,
never as Seeger-2016 itself.

Runs under external/mxnet_env (py3.8 + mxnet 1.7 + gluonts 0.11.12), on
panel CSVs exported by the caller (plain unique_id, ds, y). Predicts the
whole evaluation span in horizon-length blocks by prediction-only chaining
(history extended with REVEALED targets only, mirroring run_prob's DeepAR/DRP
protocol; no training on the evaluation span).

Output: outputs/aistats2027_rebuild/deepstate_quantiles_{name}.csv
Usage:
  mxnet_env python scripts/rebuild/t4_deepstate_runner.py or1000 D 30
  mxnet_env python scripts/rebuild/t4_deepstate_runner.py carparts M 6
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "aistats2027_rebuild"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
SEED = 42


def main() -> None:
    name, freq, horizon = sys.argv[1], sys.argv[2], int(sys.argv[3])
    t0 = time.time()
    import mxnet as mx

    mx.random.seed(SEED)
    np.random.seed(SEED)
    from gluonts.dataset.common import ListDataset
    from gluonts.mx.model.deepstate import DeepStateEstimator
    from gluonts.mx.trainer import Trainer

    init = pd.read_csv(OUT / f"panel_{name}_init.csv")
    ev = pd.read_csv(OUT / f"panel_{name}_eval.csv")
    for df in (init, ev):
        df["unique_id"] = df["unique_id"].astype(str)
        df["ds"] = pd.to_datetime(df["ds"])
    print(f"[t4] {name}: {init.unique_id.nunique()} series, freq={freq}, "
          f"h_block={horizon}", flush=True)

    gfreq = "D" if freq == "D" else "M"

    def to_ds(frame):
        entries = []
        for uid, g in frame.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
            entries.append({"item_id": uid, "start": pd.Timestamp(g.ds.iloc[0]),
                            "target": g.y.to_numpy(dtype=float)})
        return ListDataset(entries, freq=gfreq)

    estimator = DeepStateEstimator(
        freq=gfreq,
        prediction_length=horizon,
        cardinality=[1],
        use_feat_static_cat=False,
        trainer=Trainer(epochs=20, num_batches_per_epoch=50, learning_rate=1e-3,
                        hybridize=False),
    )
    predictor = estimator.train(to_ds(init))
    train_secs = time.time() - t0
    print(f"[t4] trained in {train_secs/60:.1f} min", flush=True)

    ev_idx = ev.sort_values(["unique_id", "ds"]).copy()
    ev_idx["k"] = ev_idx.groupby("unique_id").cumcount()
    n_blocks = int(np.ceil((ev_idx.k.max() + 1) / horizon))
    hist = init.copy()
    rows = []
    for b in range(n_blocks):
        tb = time.time()
        block = ev_idx[(ev_idx.k >= b * horizon) & (ev_idx.k < (b + 1) * horizon)]
        if block.empty:
            continue
        preds = predictor.predict(to_ds(hist), num_samples=400)
        uids = hist.sort_values(["unique_id", "ds"]).unique_id.unique()
        for uid, fc in zip(uids, preds):
            tgt = block[block.unique_id == uid].sort_values("ds")
            if tgt.empty:
                continue
            n = len(tgt)
            samp = fc.samples[:, :n]
            qs = np.quantile(samp, QUANTILES, axis=0)
            for i, (_, r) in enumerate(tgt.iterrows()):
                row = {"unique_id": uid, "ds": str(r.ds.date()),
                       "mean": float(samp[:, i].mean())}
                for qi, q in enumerate(QUANTILES):
                    row[f"q_{q}"] = float(qs[qi, i])
                rows.append(row)
        hist = pd.concat([hist, block[["unique_id", "ds", "y"]]],
                         ignore_index=True)
        print(f"[t4] block {b+1}/{n_blocks}: {time.time()-tb:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"deepstate_quantiles_{name}.csv", index=False)
    meta = {"task": "t4_deepstate", "panel": name,
            "wall_clock_sec": time.time() - t0, "train_seconds": train_secs,
            "seed": SEED, "gpu": False,
            "config": "gluonts 0.11.12 DeepStateEstimator defaults, epochs=20, "
                      "batches/epoch=50, lr=1e-3, 400 samples, "
                      "prediction-only chaining",
            "label": "closest runnable descendant of the Amazon ISSM line "
                     "(Rangapuram et al. 2018); NOT Seeger et al. 2016 itself"}
    (OUT / f"t4_run_meta_{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[t4] wrote deepstate_quantiles_{name}.csv "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
