"""Add TSB-tuned to the OR per-series paired-significance family (W3 follow-up).

Generates TSB-tuned predictions on the identical Online Retail split, merges
them into the saved per-timestamp prediction frame from or_point_fixed, and
re-runs the paired-test family (focal = TSB-HB-Discount, the paper's
forgetting configuration) so the Holm adjustment covers the enlarged family.

Usage: py scripts/analysis/tsb_tuned_paired.py
Outputs:
  outputs/aistats2027/or_point_fixed/point_predictions_with_tuned.csv.gz
  outputs/aistats2027/paired_tests_with_tuned.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import numpy as np
import pandas as pd

from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from models.tsb_tuned import fit_predict_tsb_tuned, TSB_TUNED_COL
from paired_tests import holm_adjust, per_series_losses  # noqa: E402
from scipy import stats


def main() -> None:
    pred_path = ROOT / "outputs" / "aistats2027" / "or_point_fixed" / "point_predictions.csv.gz"
    base = pd.read_csv(pred_path, dtype={"unique_id": str})
    base["unique_id"] = base["unique_id"].astype(str)
    base["ds"] = pd.to_datetime(base["ds"])

    df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    tuned_pred, chosen, _ = fit_predict_tsb_tuned(
        init_set[["unique_id", "ds", "y"]].copy(),
        eval_set[["unique_id", "ds", "y"]].copy(),
        freq="D",
    )
    print(f"TSB-tuned OR: alpha_d={chosen['alpha_d']}, alpha_p={chosen['alpha_p']}")
    tuned_pred["unique_id"] = tuned_pred["unique_id"].astype(str)
    tuned_pred["ds"] = pd.to_datetime(tuned_pred["ds"])

    merged = base.merge(tuned_pred, on=["unique_id", "ds"], how="left")
    n_match = merged[TSB_TUNED_COL].notna().sum()
    print(f"matched {n_match}/{len(merged)} rows")
    out_pred = pred_path.with_name("point_predictions_with_tuned.csv.gz")
    merged.to_csv(out_pred, index=False, compression="gzip")

    focal = "TSB-HB-Discount"
    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y", "index"}]
    losses = {m: per_series_losses(merged, m) for m in model_cols}
    rows = []
    for metric in ["mae", "rmse"]:
        fam = []
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
            w_p = stats.wilcoxon(nonzero)[1] if len(nonzero) >= 10 else np.nan
            fam.append({
                "focal": focal, "other": other, "metric": metric,
                "n_series": int(len(joined)),
                "mean_diff": float(diff.mean()), "median_diff": float(diff.median()),
                "p_t": float(t_p), "p_wilcoxon": float(w_p) if np.isfinite(w_p) else np.nan,
            })
        p_t = holm_adjust(np.array([r["p_t"] for r in fam]))
        p_w = holm_adjust(np.array([r["p_wilcoxon"] if np.isfinite(r["p_wilcoxon"]) else 1.0 for r in fam]))
        for r, a, b in zip(fam, p_t, p_w):
            r["p_t_holm"], r["p_wilcoxon_holm"] = float(a), float(b)
        rows.extend(fam)

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "aistats2027" / "paired_tests_with_tuned.csv"
    out.to_csv(out_path, index=False)
    show = out[["focal", "other", "metric", "n_series", "mean_diff", "median_diff", "p_t_holm", "p_wilcoxon_holm"]]
    print(show.to_string(index=False))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
