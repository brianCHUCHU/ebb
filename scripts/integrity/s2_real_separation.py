"""S2: size-block separation of the real panels, per docs/DESIGN_separation.md.

For each panel at its audited (structure, w) on the same initialization
window as the leverage table: item mean log-size from the discounted
sufficient statistics; ANOVA R^2 of that quantity across (i) the learned
mixture labels used in the paper and (ii) the ADI/CV^2 taxonomy, plus the
median size-block credibility from leverage_dual_lambda (unchanged).

Outputs (outputs/<date>/): real_separation.csv, real_separation_groups.csv, s2_run_meta.json
Usage: py scripts/integrity/s2_real_separation.py [panel ...]
"""
from __future__ import annotations

import importlib.util
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

from experiments.run_prob import _build_regime_group_labels
from models.mixture_pooling import mixture_group_labels
from models.tsb_hb import _compute_series_stats

ldl_spec = importlib.util.spec_from_file_location("ldl", ROOT / "scripts" / "analysis" / "leverage_dual_lambda.py")
ldl = importlib.util.module_from_spec(ldl_spec); ldl_spec.loader.exec_module(ldl)

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
W_STAR = {"online_retail": 0.95, "m5": 0.95, "auto": 0.99, "carparts": 0.90, "raf": 0.997}
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]


def r2_size(stats, labels):
    m = stats["mean_log"]; ok = np.isfinite(m); m = m[ok]
    g = labels.reindex(m.index).astype(str)
    keep = g.notna() & (g != "nan"); m, g = m[keep], g[keep]
    if m.empty or g.nunique() < 2:
        return 0.0, m.shape[0]
    tot = float(((m - m.mean()) ** 2).sum())
    within = float(sum(((m[g == k] - m[g == k].mean()) ** 2).sum() for k in g.unique()))
    return (0.0 if tot <= 0 else 1.0 - within / tot), int(m.shape[0])


def main():
    panels = sys.argv[1:] or PANELS
    t0 = time.perf_counter(); rows, grows = [], []
    for p in panels:
        init = ldl.load_init(p)
        init = init.copy(); init["unique_id"] = init["unique_id"].astype(str)
        w = W_STAR[p]
        lam = ldl.dual_lambda(init, w)
        stats = _compute_series_stats(init, fit_discount=w)
        mix = mixture_group_labels(init, k=0, fit_discount=w)
        ml = mix.labels.astype(str); ml.index = ml.index.astype(str)
        tax = _build_regime_group_labels(init); tax.index = tax.index.astype(str)
        r2_l, n_l = r2_size(stats, ml)
        r2_t, n_t = r2_size(stats, tax)
        rows.append({"panel": p, "w": w, "median_lambda_size": round(lam["median_lambda_size"], 4),
                     "median_lambda_occ": round(lam["median_lambda_occ"], 4),
                     "k_learned": int(mix.k), "r2_learned": round(r2_l, 4), "n_items_r2": n_l,
                     "k_taxonomy": int(tax.nunique()), "r2_taxonomy": round(r2_t, 4),
                     "sd_item_mean_log": round(float(stats["mean_log"].dropna().std()), 4)})
        m = stats["mean_log"]
        for name, lab in (("learned", ml), ("taxonomy", tax)):
            g = lab.reindex(m.index).astype(str)
            for k in sorted(g.dropna().unique()):
                sub = m[(g == k) & np.isfinite(m)]
                grows.append({"panel": p, "partition": name, "group": k, "n_items": int((g == k).sum()),
                              "share": round(float((g == k).mean()), 4), "center_mean_log": round(float(sub.mean()), 4) if len(sub) else np.nan,
                              "sd_mean_log": round(float(sub.std()), 4) if len(sub) > 1 else np.nan})
        print(f"[s2] {p}: lam+ {lam['median_lambda_size']:.3f} | learned K={mix.k} R2={r2_l:.3f} | "
              f"taxonomy K={tax.nunique()} R2={r2_t:.3f} | n={n_l} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "real_separation.csv", index=False)
        pd.DataFrame(grows).to_csv(OUT / "real_separation_groups.csv", index=False)
    meta = {"task": "s2_real_separation", "design": "docs/DESIGN_separation.md",
            "r2": "1 - SS_within/SS_total of item mean log-size (discounted stats at w*) across the partition",
            "wall_clock_total_s": round(time.perf_counter() - t0, 1), "platform": platform.platform(),
            "python": sys.version.split()[0], "seed": 42}
    (OUT / "s2_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
