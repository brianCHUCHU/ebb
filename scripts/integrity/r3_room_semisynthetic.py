"""R3: semi-synthetic transfer test of the pre-fit pooling-room diagnostic
(docs/DESIGN_room_diagnostic.md). Real series (Online Retail, Carparts, M5
5,000-series sample) keep their zeros, noise, lengths and scales; a known
group structure is injected by multiplying every positive value of item i by
exp(offset_{g(i)}), offsets symmetric (+-0.5 delta, +-1.5 delta) over four
randomly assigned groups. Occurrence is not touched. Features come from the
global fit on the (injected) initialization window; the target is the oracle
gain of the injected partition on the real fixed-origin evaluation span.

Outputs (outputs/<date>/): room_semisynthetic.csv, r3_run_meta.json
Usage: py scripts/integrity/r3_room_semisynthetic.py [panel ...]
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "integrity"))

import numpy as np
import pandas as pd

from data_loading import (load_generic_long, load_m5_long, load_online_retail, preprocess_m5,
                          preprocess_online_retail, train_eval_split_fixed_origin,
                          train_eval_split_last_h)
from experiments.run_prob import _enforce_monotonic_quantiles, _scaled_pinball_table
from models.eb_hurdle import fit_eb_hurdle, predict_eb_hurdle
from room_features import room_features
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
W_STAR = {"online_retail": 0.95, "carparts": 0.90, "m5": 0.95}
DELTAS = (0.0, 0.25, 0.5, 1.0, 2.0)
N_ASSIGN = 3
SEED = 20260915
SEL_Q = [0.5, 0.75, 0.9]
K_IDX = np.array([-1.5, -0.5, 0.5, 1.5])


def log(m):
    print(f"[r3] {m}", flush=True)


def load_panel(name):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data/online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    if name == "m5":
        set_seed(42)
        s, c = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        return train_eval_split_fixed_origin(preprocess_m5(s, c, sample_size=5000), init_ratio=2 / 3, min_len=1)
    return train_eval_split_last_h(load_generic_long(ROOT / f"data/{name}_long.csv"), h=6)


def pinball(init, ev, labels, w):
    params = fit_eb_hurdle(init, group_labels=labels, item_variance_mode="conjugate",
                        item_variance_shrink_strength=20.0, fit_discount=w)
    pred = predict_eb_hurdle(params, ev, quantiles=SEL_Q, include_hyper_uncertainty=False)
    pred = _enforce_monotonic_quantiles(pred, quantiles=SEL_Q)
    pred["model"] = "x"
    merged = ev[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="inner")
    spl = _scaled_pinball_table(merged, init_set=init, quantiles=SEL_Q)
    return float(spl[spl["model"] == "x"]["scaled_pinball"].mean())


def main():
    args = sys.argv[1:]
    w_override = None
    if "--w" in args:                      # regime sweep: strong forgetting on the same real series
        i = args.index("--w"); w_override = float(args[i + 1]); args = args[:i] + args[i + 2:]
    panels = args or ["carparts", "online_retail", "m5"]
    t0 = time.perf_counter(); rows = []
    out_path = OUT / ("room_semisynthetic.csv" if w_override is None else f"room_semisynthetic_w{w_override:g}.csv")
    if out_path.exists():
        rows = [r for r in pd.read_csv(out_path).to_dict("records") if r["panel"] not in panels]
    for panel in panels:
        init0, ev0 = load_panel(panel)
        init0 = init0.copy(); ev0 = ev0.copy()
        for d in (init0, ev0):
            d["unique_id"] = d["unique_id"].astype(str)
        uids = np.array(sorted(init0["unique_id"].unique()))
        w = W_STAR[panel] if w_override is None else w_override
        for a in range(N_ASSIGN):
            rng = np.random.default_rng(SEED + a)
            grp = pd.Series(rng.integers(0, 4, len(uids)), index=uids)
            for delta in DELTAS:
                offset = grp.map(lambda g: K_IDX[g] * delta).astype(float)
                init = init0.copy(); ev = ev0.copy()
                for d in (init, ev):
                    d["y"] = d["y"].astype(float)
                    mult = np.exp(d["unique_id"].map(offset).to_numpy(float))
                    pos = d["y"].to_numpy(float) > 0
                    d.loc[pos, "y"] = d.loc[pos, "y"].to_numpy(float) * mult[pos]
                labels = grp.map(lambda g: f"g{g}").astype(str)
                tb = time.perf_counter()
                feats = room_features(init, w)
                pg = pinball(init, ev, None, w)
                pt = pinball(init, ev, labels, w)
                rows.append({"panel": panel, "assign": a, "delta": delta, "w": w, "n_items": int(len(uids)),
                             "pin_global": pg, "pin_true": pt, "delta_oracle": 100 * (pg - pt) / pg, **feats})
                log(f"{panel} assign={a} delta={delta}: oracle gain {rows[-1]['delta_oracle']:+.2f}% R1 {feats['R1']:.4f} "
                    f"lev {feats['lev_size']:.2f}/{feats['lev_occ']:.2f} het {feats['het_size']:.3f} ({(time.perf_counter()-tb):.0f}s)")
                pd.DataFrame(rows).to_csv(out_path, index=False)
    meta = {"task": "r3_room_semisynthetic", "design": "docs/DESIGN_room_diagnostic.md", "seed": SEED,
            "deltas": DELTAS, "assignments": N_ASSIGN, "offsets": "K_IDX*delta on log size, positives only",
            "wall_clock_total_s": round(time.perf_counter() - t0, 1), "platform": platform.platform(),
            "python": sys.version.split()[0]}
    (OUT / "r3_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
