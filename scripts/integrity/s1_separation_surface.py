"""S1: true-partition return as a surface over (credibility leverage, size-block
separation), per docs/DESIGN_separation.md. Generator and arms imported from
src/experiments/run_synthetic_sanity.py unchanged; adds a `mixture-learned`
arm (the paper's partition objective run on the simulated panel).

Outputs (outputs/<date>/): sep_surface.csv (per replicate), sep_surface_cells.csv
  (per cell: mean/SE gains, median lambda, R2), s1_run_meta.json
Usage: py scripts/integrity/s1_separation_surface.py
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

import numpy as np
import pandas as pd

from models.mixture_pooling import mixture_group_labels
from models.tsb_hb import _compute_series_stats, fit_tsb_hb

spec = importlib.util.spec_from_file_location("rss", ROOT / "src" / "experiments" / "run_synthetic_sanity.py")
rss = importlib.util.module_from_spec(spec); spec.loader.exec_module(rss)

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260913
SEPARATIONS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
DISCOUNTS = (1.0, 0.95, 0.90)
LENGTHS = (30, 120)
N_REPS = 9
GEN = dict(n_groups=4, items_per_group=60, tau2=0.15, sigma2=1.0,
           occurrence_mean=0.25, occurrence_spread=0.8, grand_mu=1.0, drift=0.5)


def log(m):
    print(f"[s1] {m}", flush=True)


def r2_size(stats: pd.DataFrame, labels: pd.Series) -> float:
    """ANOVA R^2 of item mean log-size across a partition (items with a finite mean)."""
    m = stats["mean_log"]
    ok = np.isfinite(m)
    m = m[ok]; g = labels.reindex(m.index).astype(str)
    if m.empty or g.nunique() < 2:
        return 0.0
    tot = float(((m - m.mean()) ** 2).sum())
    within = float(sum(((m[g == k] - m[g == k].mean()) ** 2).sum() for k in g.unique()))
    return 0.0 if tot <= 0 else 1.0 - within / tot


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    rows = []
    cells = list(itertools.product(SEPARATIONS, DISCOUNTS, LENGTHS))
    for ci, (sep, w, T) in enumerate(cells):
        for rep in range(N_REPS):
            h = max(T // 8, 6)
            panel, labels, path, hyp = rss.simulate(rng, T=T, horizon=h, separation=sep, **GEN)
            train, test = rss.split(panel, h)
            labels = labels.astype(str); labels.index = labels.index.astype(str)
            train = train.copy(); train["unique_id"] = train["unique_id"].astype(str)
            test = test.copy(); test["unique_id"] = test["unique_id"].astype(str)
            recs = rss.run_arms(train, test, labels, path, hyp, sigma2=GEN["sigma2"], w=w)
            pin = {r["arm"]: r["pinball"] for r in recs}
            lam = [r for r in recs if r["arm"] == "global-estimated"][0]["median_lambda_size"]
            # learned partition (paper objective) on the simulated panel
            mix = mixture_group_labels(train, k=0, fit_discount=w)
            ml = mix.labels.astype(str); ml.index = ml.index.astype(str)
            params = fit_tsb_hb(train, group_labels=ml, item_variance_mode="conjugate",
                                item_variance_shrink_strength=20.0, fit_discount=w)
            p, mu, s2 = rss.params_to_triple(params, test)
            pin["mixture-learned"] = rss.score_predictive(test, p, mu, s2)["pinball"]
            stats = _compute_series_stats(train, fit_discount=w)
            rows.append({"separation": sep, "discount": w, "T": T, "rep": rep,
                         "pin_global": pin["global-estimated"],
                         "pin_true": pin["oracle-labels+estimated-hypers"],
                         "pin_true_hypers": pin["oracle-labels+true-hypers"],
                         "pin_learned": pin["mixture-learned"],
                         "gain_true_pct": 100 * (pin["global-estimated"] - pin["oracle-labels+estimated-hypers"]) / pin["global-estimated"],
                         "gain_learned_pct": 100 * (pin["global-estimated"] - pin["mixture-learned"]) / pin["global-estimated"],
                         "median_lambda_size": lam, "k_learned": int(mix.k),
                         "r2_true": r2_size(stats, labels), "r2_learned": r2_size(stats, ml)})
        c = pd.DataFrame(rows[-N_REPS:])
        log(f"cell {ci+1}/{len(cells)} sep={sep} w={w} T={T}: gain_true {c.gain_true_pct.mean():+.2f}% "
            f"learned {c.gain_learned_pct.mean():+.2f}% lam+ {c.median_lambda_size.median():.3f} "
            f"R2_true {c.r2_true.mean():.3f} R2_learned {c.r2_learned.mean():.3f} K {c.k_learned.median():.0f} "
            f"({(time.perf_counter()-t0)/60:.1f} min)")
        pd.DataFrame(rows).to_csv(OUT / "sep_surface.csv", index=False)
    df = pd.DataFrame(rows)
    agg = df.groupby(["separation", "discount", "T"]).agg(
        gain_true=("gain_true_pct", "mean"), gain_true_se=("gain_true_pct", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        gain_learned=("gain_learned_pct", "mean"), gain_learned_se=("gain_learned_pct", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        lambda_size=("median_lambda_size", "median"), r2_true=("r2_true", "mean"), r2_learned=("r2_learned", "mean"),
        k_learned=("k_learned", "median")).reset_index()
    agg.round(4).to_csv(OUT / "sep_surface_cells.csv", index=False)
    print(agg.round(3).to_string())
    meta = {"task": "s1_separation_surface", "design": "docs/DESIGN_separation.md", "seed": SEED,
            "grid": {"separation": SEPARATIONS, "discount": DISCOUNTS, "T": LENGTHS, "reps": N_REPS}, "generator": GEN,
            "wall_clock_total_s": round(time.perf_counter() - t0, 1), "platform": platform.platform(),
            "python": sys.version.split()[0]}
    (OUT / "s1_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
