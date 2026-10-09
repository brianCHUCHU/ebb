"""D4: paired per-series comparison of DeepAR with EBB at fixed origin (docs/DESIGN_deepar.md).

EBB quantiles are recomputed exactly as in c1_coldstart.py at full length (audited structure and
discount, B=20), which reproduces the published fixed-origin numbers; per-series mean scaled
pinball losses of the two methods are then compared with a paired t-test and a Wilcoxon test.

Output: outputs/deepar/deepar_paired_fixed.csv
Usage : py scripts/integrity/d4_deepar_paired.py [panel ...]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import fit_eb_hurdle, predict_eb_hurdle

spec = importlib.util.spec_from_file_location("c1", ROOT / "scripts" / "integrity" / "c1_coldstart.py")
c1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c1)

OUT = ROOT / "outputs" / "deepar"
Q = c1.Q5


def per_series_spl(pred: pd.DataFrame, ev: pd.DataFrame, init: pd.DataFrame) -> pd.Series:
    scale = init.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s)))).replace(0.0, np.nan)
    scale = scale.fillna(float(np.nanmedian(scale.to_numpy())))
    m = ev[["unique_id", "ds", "y"]].merge(pred[["unique_id", "ds"] + c1.QCOLS], on=["unique_id", "ds"], how="inner")
    loss = np.zeros(len(m))
    for q in Q:
        e = m["y"].to_numpy(float) - m[f"q_{q}"].to_numpy(float)
        loss += np.maximum(q * e, (q - 1) * e)
    loss /= len(Q)
    m["l"] = loss / m["unique_id"].map(scale).to_numpy(float)
    return m.groupby("unique_id")["l"].mean()


def main() -> None:
    rows = []
    for p in (sys.argv[1:] or ["auto", "carparts", "raf"]):
        f = OUT / f"deepar_fixed_{p}.csv"
        if not f.exists():
            continue
        structure, w, _ = c1.CONFIG[p]
        init, ev = c1.load_panel(p)
        init = init.copy(); init["unique_id"] = init["unique_id"].astype(str)
        ev = ev.copy(); ev["unique_id"] = ev["unique_id"].astype(str); ev["ds"] = pd.to_datetime(ev["ds"])
        labels = mixture_group_labels(init, k=0, fit_discount=w).labels if structure == "mixture" else None
        if labels is not None:
            labels.index = labels.index.astype(str)
        prm = fit_eb_hurdle(init, group_labels=labels, item_variance_mode="conjugate",
                         item_variance_shrink_strength=20.0, fit_discount=w, bootstrap_draws=20, bootstrap_seed=42)
        e = predict_eb_hurdle(prm, ev, quantiles=Q, include_hyper_uncertainty=True)
        e["unique_id"] = e["unique_id"].astype(str); e["ds"] = pd.to_datetime(e["ds"])
        e = c1._enforce_monotonic_quantiles(e, quantiles=Q)
        d = pd.read_csv(f, dtype={"unique_id": str}); d["ds"] = pd.to_datetime(d["ds"])
        for c in c1.QCOLS:
            d[c] = np.maximum(d[c].astype(float), 0.0)
        d = c1._enforce_monotonic_quantiles(d, quantiles=Q)
        se, sd = per_series_spl(e, ev, init), per_series_spl(d, ev, init)
        j = pd.concat([se.rename("ebb"), sd.rename("deepar")], axis=1).dropna()
        diff = j["ebb"] - j["deepar"]
        t, pt = stats.ttest_1samp(diff, 0.0)
        try:
            _, pw = stats.wilcoxon(diff[diff != 0])
        except ValueError:
            pw = np.nan
        ebb_head = c1.score(e, "EBB", ev[["unique_id", "ds", "y"]], init)[0]
        rows.append({"panel": p, "n_series": len(j), "EBB_spl_table_scale": round(ebb_head, 4),
                     "mean_diff_ebb_minus_deepar": float(diff.mean()), "t": float(t), "p_t": float(pt),
                     "p_wilcoxon": float(pw), "ebb_better": bool(diff.mean() < 0),
                     "significant_5pct": bool(pt < 0.05)})
        print(f"[d4] {p}: EBB {ebb_head:.4f} | mean diff {diff.mean():+.5f} t={t:.2f} p={pt:.3g} wilcoxon p={pw:.3g}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT / "deepar_paired_fixed.csv", index=False)


if __name__ == "__main__":
    main()
