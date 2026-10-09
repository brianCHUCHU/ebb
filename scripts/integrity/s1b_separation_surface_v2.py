"""S1b: separation surface, version 2 (pre-registered addendum to
docs/DESIGN_separation.md, 2026-09-13):
  (3) occurrence separation scaled with size separation, occurrence_spread =
      0.8 * min(sep, 1), so zero separation means zero structure in both blocks
      (cells with sep >= 1 are identical to version 1);
  (2) two remedies for the learned mixture, already in the released code, run
      as extra arms: label sample-splitting (`split_for_hyper_estimation`,
      parity) and credibility hyper-shrinkage (`hyper_shrink='credibility'`),
      alone and combined.
Same grid, seed and generator otherwise. Outputs sep_surface_v2.csv,
sep_surface_v2_cells.csv, s1b_run_meta.json.
Usage: py scripts/integrity/s1b_separation_surface_v2.py
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
from models.eb_hurdle import _compute_series_stats, fit_eb_hurdle, split_for_hyper_estimation

spec = importlib.util.spec_from_file_location("rss", ROOT / "src" / "experiments" / "run_synthetic_sanity.py")
rss = importlib.util.module_from_spec(spec); spec.loader.exec_module(rss)

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260913
SEPARATIONS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
DISCOUNTS = (1.0, 0.95, 0.90)
LENGTHS = (30, 120)
N_REPS = 9
GEN = dict(n_groups=4, items_per_group=60, tau2=0.15, sigma2=1.0, occurrence_mean=0.25, grand_mu=1.0, drift=0.5)


def log(m):
    print(f"[s1b] {m}", flush=True)


def r2_size(stats, labels):
    m = stats["mean_log"]; ok = np.isfinite(m); m = m[ok]
    g = labels.reindex(m.index).astype(str)
    if m.empty or g.nunique() < 2:
        return 0.0
    tot = float(((m - m.mean()) ** 2).sum())
    within = float(sum(((m[g == k] - m[g == k].mean()) ** 2).sum() for k in g.unique()))
    return 0.0 if tot <= 0 else 1.0 - within / tot


def pinball_of(train, test, labels, w, hyper_df=None, hyper_shrink="off"):
    params = fit_eb_hurdle(train, group_labels=labels, item_variance_mode="conjugate",
                        item_variance_shrink_strength=20.0, fit_discount=w,
                        hyper_train_df=hyper_df, hyper_shrink=hyper_shrink)
    p, mu, s2 = rss.params_to_triple(params, test)
    return rss.score_predictive(test, p, mu, s2)["pinball"]


def main():
    t0 = time.perf_counter(); rng = np.random.default_rng(SEED); rows = []
    cells = list(itertools.product(SEPARATIONS, DISCOUNTS, LENGTHS))
    for ci, (sep, w, T) in enumerate(cells):
        occ_spread = 0.8 * min(sep, 1.0)
        for rep in range(N_REPS):
            h = max(T // 8, 6)
            panel, labels, path, hyp = rss.simulate(rng, T=T, horizon=h, separation=sep, occurrence_spread=occ_spread, **GEN)
            train, test = rss.split(panel, h)
            labels = labels.astype(str); labels.index = labels.index.astype(str)
            train = train.copy(); train["unique_id"] = train["unique_id"].astype(str)
            test = test.copy(); test["unique_id"] = test["unique_id"].astype(str)
            recs = rss.run_arms(train, test, labels, path, hyp, sigma2=GEN["sigma2"], w=w)
            pin = {r["arm"]: r["pinball"] for r in recs}
            lam = [r for r in recs if r["arm"] == "global-estimated"][0]["median_lambda_size"]
            g = pin["global-estimated"]
            # learned mixture on the full window (as in the paper)
            mix = mixture_group_labels(train, k=0, fit_discount=w)
            ml = mix.labels.astype(str); ml.index = ml.index.astype(str)
            pin["learned"] = pinball_of(train, test, ml, w)
            pin["learned+cred"] = pinball_of(train, test, ml, w, hyper_shrink="credibility")
            # labels on the structure half, hypers on the other half (parity)
            s_half, h_half = split_for_hyper_estimation(train, mode="parity")
            mix_s = mixture_group_labels(s_half, k=0, fit_discount=w)
            ms = mix_s.labels.astype(str); ms.index = ms.index.astype(str)
            pin["learned+split"] = pinball_of(train, test, ms, w, hyper_df=h_half)
            pin["learned+split+cred"] = pinball_of(train, test, ms, w, hyper_df=h_half, hyper_shrink="credibility")
            stats = _compute_series_stats(train, fit_discount=w)
            rows.append({"separation": sep, "discount": w, "T": T, "rep": rep, "occ_spread": occ_spread,
                         "pin_global": g, "pin_true": pin["oracle-labels+estimated-hypers"],
                         "gain_true_pct": 100 * (g - pin["oracle-labels+estimated-hypers"]) / g,
                         "gain_learned_pct": 100 * (g - pin["learned"]) / g,
                         "gain_learned_cred_pct": 100 * (g - pin["learned+cred"]) / g,
                         "gain_learned_split_pct": 100 * (g - pin["learned+split"]) / g,
                         "gain_learned_split_cred_pct": 100 * (g - pin["learned+split+cred"]) / g,
                         "median_lambda_size": lam, "k_learned": int(mix.k), "k_learned_split": int(mix_s.k),
                         "r2_true": r2_size(stats, labels), "r2_learned": r2_size(stats, ml)})
        c = pd.DataFrame(rows[-N_REPS:])
        log(f"cell {ci+1}/{len(cells)} sep={sep} w={w} T={T}: true {c.gain_true_pct.mean():+.2f} learned {c.gain_learned_pct.mean():+.2f} "
            f"cred {c.gain_learned_cred_pct.mean():+.2f} split {c.gain_learned_split_pct.mean():+.2f} split+cred {c.gain_learned_split_cred_pct.mean():+.2f} "
            f"| lam+ {c.median_lambda_size.median():.3f} R2 {c.r2_true.mean():.3f} ({(time.perf_counter()-t0)/60:.1f} min)")
        pd.DataFrame(rows).to_csv(OUT / "sep_surface_v2.csv", index=False)
    df = pd.DataFrame(rows)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    agg = df.groupby(["separation", "discount", "T"]).agg(
        gain_true=("gain_true_pct", "mean"), gain_true_se=("gain_true_pct", se),
        gain_learned=("gain_learned_pct", "mean"), gain_learned_se=("gain_learned_pct", se),
        gain_learned_cred=("gain_learned_cred_pct", "mean"), gain_learned_split=("gain_learned_split_pct", "mean"),
        gain_learned_split_cred=("gain_learned_split_cred_pct", "mean"), gain_learned_split_cred_se=("gain_learned_split_cred_pct", se),
        lambda_size=("median_lambda_size", "median"), r2_true=("r2_true", "mean"), r2_learned=("r2_learned", "mean"),
        k_learned=("k_learned", "median"), k_learned_split=("k_learned_split", "median")).reset_index()
    agg.round(4).to_csv(OUT / "sep_surface_v2_cells.csv", index=False)
    print(agg.round(3).to_string())
    meta = {"task": "s1b_separation_surface_v2", "design": "docs/DESIGN_separation.md (addendum 2026-09-13)", "seed": SEED,
            "grid": {"separation": SEPARATIONS, "discount": DISCOUNTS, "T": LENGTHS, "reps": N_REPS},
            "occurrence_spread": "0.8*min(sep,1)", "remedy_arms": ["learned+cred", "learned+split (parity)", "learned+split+cred"],
            "wall_clock_total_s": round(time.perf_counter() - t0, 1), "platform": platform.platform(), "python": sys.version.split()[0]}
    (OUT / "s1b_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
