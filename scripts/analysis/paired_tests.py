"""Paired per-series significance tests for point forecasts.

Usage:
  py scripts/analysis/paired_tests.py \
      --predictions outputs/paper_runs/or_point_fixed/point_predictions.csv.gz \
      --focal EB-Hurdle-Discount EB-Hurdle-Mixture EB-Hurdle-Mix-Disc EB-Hurdle \
      --out outputs/paper_runs/or_point_fixed/paired_tests.csv

For every focal model and every other model, computes per-series MAE and RMSE,
then paired t-test and Wilcoxon signed-rank test on the per-series losses,
with Holm adjustment within each (focal, metric) family.
Negative mean_diff means the focal model has LOWER (better) loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj


def per_series_losses(df: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = df[["unique_id", "y", model]].dropna(subset=[model])
    err = sub[model].astype(float) - sub["y"].astype(float)
    tmp = pd.DataFrame({"unique_id": sub["unique_id"], "abs_err": err.abs(), "sq_err": err**2})
    g = tmp.groupby("unique_id")
    return pd.DataFrame({"mae": g["abs_err"].mean(), "rmse": np.sqrt(g["sq_err"].mean())})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--focal", nargs="+", default=["EB-Hurdle", "EB-Hurdle-Discount", "EB-Hurdle-Mixture", "EB-Hurdle-Mix-Disc"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    model_cols = [c for c in df.columns if c not in {"unique_id", "ds", "y", "index"}]
    losses = {m: per_series_losses(df, m) for m in model_cols}

    rows: list[dict[str, object]] = []
    for focal in args.focal:
        if focal not in losses:
            continue
        for metric in ["mae", "rmse"]:
            fam: list[dict[str, object]] = []
            for other in model_cols:
                if other == focal:
                    continue
                joined = losses[focal][[metric]].join(
                    losses[other][[metric]], how="inner", lsuffix="_f", rsuffix="_o"
                ).dropna()
                if len(joined) < 10:
                    continue
                diff = joined[f"{metric}_f"] - joined[f"{metric}_o"]
                t_stat, t_p = stats.ttest_rel(joined[f"{metric}_f"], joined[f"{metric}_o"])
                nonzero = diff[diff != 0]
                if len(nonzero) >= 10:
                    w_stat, w_p = stats.wilcoxon(nonzero)
                else:
                    w_stat, w_p = np.nan, np.nan
                fam.append(
                    {
                        "focal": focal,
                        "other": other,
                        "metric": metric,
                        "n_series": int(len(joined)),
                        "mean_diff": float(diff.mean()),
                        "median_diff": float(diff.median()),
                        "t_stat": float(t_stat),
                        "p_t": float(t_p),
                        "p_wilcoxon": float(w_p) if np.isfinite(w_p) else np.nan,
                    }
                )
            if fam:
                p_t = holm_adjust(np.array([r["p_t"] for r in fam], dtype=float))
                pw_raw = np.array(
                    [r["p_wilcoxon"] if np.isfinite(r["p_wilcoxon"]) else 1.0 for r in fam], dtype=float
                )
                p_w = holm_adjust(pw_raw)
                for r, a, b in zip(fam, p_t, p_w):
                    r["p_t_holm"] = float(a)
                    r["p_wilcoxon_holm"] = float(b)
                rows.extend(fam)

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    focal_wins = out[(out["mean_diff"] < 0) & (out["p_wilcoxon_holm"] < 0.05)]
    print(f"Wrote {len(out)} comparisons to {args.out}")
    print(f"Significant focal wins (Holm-adjusted Wilcoxon p<0.05): {len(focal_wins)}")
    if not out.empty:
        show = out[["focal", "other", "metric", "mean_diff", "p_t_holm", "p_wilcoxon_holm"]]
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
