"""R1: expanded synthetic grid for the pre-fit pooling-room diagnostic
(docs/DESIGN_room_diagnostic.md). Per panel: global-pool features (no labels
used) and the oracle target Delta = 100*(pin_global - pin_true)/pin_global.

Grid: separation x w x T x occurrence mean x items/group x balance; 5 reps;
seed 20260915. Output: outputs/<date>/room_synthetic.csv (one row per panel),
r1_run_meta.json. Resumable (skips cells already written).
Usage: py scripts/integrity/r1_room_synthetic.py
"""
from __future__ import annotations

import importlib.util
import itertools
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

from models.eb_hurdle import fit_eb_hurdle
from room_features import room_features

spec = importlib.util.spec_from_file_location("rss", ROOT / "src" / "experiments" / "run_synthetic_sanity.py")
rss = importlib.util.module_from_spec(spec); spec.loader.exec_module(rss)

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260915
GRID = dict(separation=(0.0, 0.25, 0.5, 1.0, 2.0), discount=(1.0, 0.95, 0.90), T=(30, 60, 120),
            occ_mean=(0.10, 0.25, 0.50), items=(20, 60), balance=("balanced", "skewed"))
N_REPS = 5
SKEW = (0.70, 0.10, 0.10, 0.10)


def log(m):
    print(f"[r1] {m}", flush=True)


def pinball(train, test, labels, w):
    params = fit_eb_hurdle(train, group_labels=labels, item_variance_mode="conjugate",
                        item_variance_shrink_strength=20.0, fit_discount=w)
    p, mu, s2 = rss.params_to_triple(params, test)
    return rss.score_predictive(test, p, mu, s2)["pinball"]


def skew_panel(panel, labels, rng):
    """Keep 70/10/10/10 of the items per group (relative to equal size) by dropping items."""
    keep = []
    groups = sorted(labels.unique())
    per = labels.value_counts()
    base = int(per.min())
    for share, g in zip(SKEW, groups):
        ids = labels.index[labels == g].to_numpy()
        n_keep = max(2, int(round(share * 4 * base)))   # share of the balanced total, capped by availability
        n_keep = min(n_keep, len(ids))
        keep.extend(rng.choice(ids, size=n_keep, replace=False))
    keep = set(map(str, keep))
    return panel[panel["unique_id"].astype(str).isin(keep)].copy(), labels[labels.index.astype(str).isin(keep)]


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    out_path = OUT / "room_synthetic.csv"
    done = set()
    rows = []
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = set(zip(prev.separation, prev.discount, prev["T"], prev.occ_mean, prev["items"], prev.balance, prev.rep))
        log(f"resuming with {len(rows)} rows")
    cells = list(itertools.product(*GRID.values()))
    for ci, (sep, w, T, occ, items, bal) in enumerate(cells):
        for rep in range(N_REPS):
            key = (sep, w, T, occ, items, bal, rep)
            # always advance the RNG identically so resume is reproducible
            h = max(T // 8, 6)
            panel, labels, path, hyp = rss.simulate(rng, n_groups=4, items_per_group=items, T=T, horizon=h,
                                                   separation=sep, tau2=0.15, sigma2=1.0, occurrence_mean=occ,
                                                   occurrence_spread=0.8 * min(sep, 1.0), grand_mu=1.0, drift=0.5)
            labels = labels.astype(str); labels.index = labels.index.astype(str)
            panel["unique_id"] = panel["unique_id"].astype(str)
            if bal == "skewed":
                panel, labels = skew_panel(panel, labels, rng)
            if key in done:
                continue
            train, test = rss.split(panel, h)
            feats = room_features(train, w)
            pg = pinball(train, test, None, w)
            pt = pinball(train, test, labels, w)
            rows.append({"separation": sep, "discount": w, "T": T, "occ_mean": occ, "items": items, "balance": bal,
                         "rep": rep, "n_items": int(train["unique_id"].nunique()), "pin_global": pg, "pin_true": pt,
                         "delta_oracle": 100 * (pg - pt) / pg, **feats})
        if (ci + 1) % 10 == 0 or ci == len(cells) - 1:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            last = pd.DataFrame(rows[-N_REPS:]) if rows else None
            log(f"cell {ci+1}/{len(cells)} sep={sep} w={w} T={T} occ={occ} items={items} {bal}: "
                f"delta {last.delta_oracle.mean():+.2f} R1 {last.R1.mean():.4f} lev {last.lev_size.mean():.2f}/{last.lev_occ.mean():.2f} "
                f"het {last.het_size.mean():.3f}/{last.het_occ.mean():.4f} ({(time.perf_counter()-t0)/60:.1f} min)")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    meta = {"task": "r1_room_synthetic", "design": "docs/DESIGN_room_diagnostic.md", "seed": SEED,
            "grid": {k: list(v) for k, v in GRID.items()}, "reps": N_REPS, "n_rows": len(rows),
            "wall_clock_total_s": round(time.perf_counter() - t0, 1), "platform": platform.platform(),
            "python": sys.version.split()[0]}
    (OUT / "r1_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"done: {len(rows)} panels")


if __name__ == "__main__":
    main()
