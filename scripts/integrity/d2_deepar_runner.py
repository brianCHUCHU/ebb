"""D2: DeepAR under the paper protocols (docs/DESIGN_deepar.md). Runs inside external/mxnet_env
(Python 3.8, MXNet 1.7, GluonTS 0.11.12, CPU).

  fixed : train on the initialization window, forecast the whole evaluation span once
          (prediction_length = H). Monthly panels only.
  wf    : train once with prediction_length = block, then forecast block by block from the history
          extended with revealed targets only (no refit).

Usage:
  external/mxnet_env/Scripts/python.exe scripts/integrity/d2_deepar_runner.py <panel> <fixed|wf> [epochs]
Output: outputs/deepar/deepar_<protocol>_<panel>.csv, deepar_run_meta_<protocol>_<panel>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "deepar"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
SEED = 42
FREQ = {"online_retail": "D", "m5": "D", "auto": "M", "carparts": "M", "raf": "M"}
BLOCK = {"online_retail": 7, "m5": 7, "auto": 1, "carparts": 1, "raf": 1}
NUM_SAMPLES = 200


def log(m):
    print(f"[d2] {m}", flush=True)


def main() -> None:
    panel, protocol = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    t0 = time.time()
    import mxnet as mx

    mx.random.seed(SEED)
    np.random.seed(SEED)
    from gluonts.dataset.common import ListDataset
    from gluonts.mx.distribution import NegativeBinomialOutput
    from gluonts.mx.model.deepar import DeepAREstimator
    from gluonts.mx.trainer import Trainer

    init = pd.read_csv(OUT / f"panel_{panel}_init.csv", dtype={"unique_id": str})
    ev = pd.read_csv(OUT / f"panel_{panel}_eval.csv", dtype={"unique_id": str})
    for df in (init, ev):
        df["ds"] = pd.to_datetime(df["ds"])
    freq = FREQ[panel]
    ev = ev.sort_values(["unique_id", "ds"]).copy()
    ev["k"] = ev.groupby("unique_id").cumcount()
    H = int(ev["k"].max() + 1)
    pl = H if protocol == "fixed" else BLOCK[panel]
    log(f"{panel} {protocol}: {init.unique_id.nunique()} series, freq={freq}, H={H}, "
        f"prediction_length={pl}, epochs={epochs}")

    def to_ds(frame):
        entries = []
        for uid, g in frame.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
            entries.append({"item_id": uid, "start": pd.Timestamp(g.ds.iloc[0]),
                            "target": g.y.to_numpy(dtype=float)})
        return ListDataset(entries, freq=freq), [e["item_id"] for e in entries]

    estimator = DeepAREstimator(
        freq=freq, prediction_length=pl, distr_output=NegativeBinomialOutput(),
        trainer=Trainer(epochs=epochs, num_batches_per_epoch=50, learning_rate=1e-3, hybridize=False),
    )
    train_ds, _ = to_ds(init)
    predictor = estimator.train(train_ds)
    train_s = time.time() - t0
    log(f"trained in {train_s / 60:.1f} min")

    rows = []
    hist = init[["unique_id", "ds", "y"]].copy()
    n_blocks = int(np.ceil(H / pl))
    for b in range(n_blocks):
        tb = time.time()
        block = ev[(ev.k >= b * pl) & (ev.k < (b + 1) * pl)]
        if block.empty:
            continue
        ds_b, uids = to_ds(hist)
        tgt_by_uid = {u: g.sort_values("ds") for u, g in block.groupby("unique_id", sort=False)}
        for uid, fc in zip(uids, predictor.predict(ds_b, num_samples=NUM_SAMPLES)):
            tgt = tgt_by_uid.get(uid)
            if tgt is None:
                continue
            n = len(tgt)
            samp = np.asarray(fc.samples)[:, :n]
            qs = np.quantile(samp, QUANTILES, axis=0)
            mean = samp.mean(axis=0)
            for i, d in enumerate(tgt["ds"].to_numpy()):
                row = {"unique_id": uid, "ds": str(pd.Timestamp(d).date()), "mean": float(mean[i])}
                for qi, q in enumerate(QUANTILES):
                    row[f"q_{q}"] = float(qs[qi, i])
                rows.append(row)
        hist = pd.concat([hist, block[["unique_id", "ds", "y"]]], ignore_index=True)
        if b < 3 or (b + 1) % 10 == 0 or b == n_blocks - 1:
            log(f"block {b + 1}/{n_blocks}: {time.time() - tb:.0f}s")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"deepar_{protocol}_{panel}.csv", index=False)
    bad = int((~np.isfinite(out[[f"q_{q}" for q in QUANTILES]].to_numpy(float))).sum())
    meta = {"panel": panel, "protocol": protocol, "prediction_length": pl, "H": H, "blocks": n_blocks,
            "epochs": epochs, "num_samples": NUM_SAMPLES, "seed": SEED, "gpu": False,
            "train_seconds": round(train_s, 1), "wall_clock_seconds": round(time.time() - t0, 1),
            "non_finite_quantiles": bad,
            "config": "gluonts 0.11.12 DeepAREstimator defaults, NegativeBinomialOutput, "
                      "50 batches/epoch, lr 1e-3; no refit on the evaluation span"}
    (OUT / f"deepar_run_meta_{protocol}_{panel}.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote deepar_{protocol}_{panel}.csv ({len(out)} rows, {bad} non-finite) in "
        f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
