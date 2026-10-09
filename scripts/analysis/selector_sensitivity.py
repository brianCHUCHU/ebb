"""Sensitivity of the EBB selector to the validation ratio (Online Retail).

For val_ratio in {0.1, 0.2, 0.3}: rerun the joint selection, then fit the
selected configuration on the full initialization window and report
out-of-sample point metrics. Also reports the learned mixture components
(K, per-component occurrence mean and median size) for interpretability.

Usage: py scripts/analysis/selector_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from metrics import compute_adi_cv2, classify_adi_cv2, rmsse
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import fit_eb_hurdle, predict_eb_hurdle, select_pooling_and_discount


def taxonomy_labels(df: pd.DataFrame) -> pd.Series:
    feats = compute_adi_cv2(df)
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)
    return feats.set_index("unique_id")["category"].astype(str)


def main() -> None:
    df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    mix = mixture_group_labels(init_set, k=0)
    cands = {"global": None, "taxonomy": taxonomy_labels(init_set), "mixture": mix.labels}

    print("=== validation-ratio sensitivity (Online Retail) ===")
    rows = []
    for vr in [0.1, 0.2, 0.3]:
        name, w, _ = select_pooling_and_discount(init_set, cands, val_ratio=vr)
        params = fit_eb_hurdle(init_set, group_labels=cands[name], item_variance_mode="conjugate",
                            item_variance_shrink_strength=20.0, fit_discount=w)
        preds = predict_eb_hurdle(params, eval_set, quantiles=None)
        merged = eval_set[["unique_id", "ds", "y"]].merge(preds, on=["unique_id", "ds"])
        err = merged["yhat"] - merged["y"]
        mae = float(err.abs().mean())
        rm = rmsse(init_set, merged.rename(columns={"yhat": "y_pred"}))
        rows.append({"val_ratio": vr, "selected": name, "w": w, "MAE": round(mae, 4), "RMSSE": round(rm, 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== learned mixture components (Online Retail, K=%d, BIC) ===" % mix.k)
    stats_rows = []
    for j, comp in enumerate(mix.components):
        occ_mean = comp.alpha / (comp.alpha + comp.beta)
        stats_rows.append({
            "component": j,
            "weight": round(comp.weight, 3),
            "occ_prior_mean": round(occ_mean, 4),
            "prior_strength_phi": round(comp.alpha + comp.beta, 2),
            "median_size": round(float(np.exp(comp.size_mu)), 2),
            "tau_sq": round(comp.size_tau_sq, 3),
        })
    print(pd.DataFrame(stats_rows).to_string(index=False))

    # cross-tab learned components vs classical taxonomy
    tax = cands["taxonomy"]
    ct = pd.crosstab(mix.labels, tax.reindex(mix.labels.index))
    print("\n=== learned components x ADI/CV2 taxonomy (item counts) ===")
    print(ct.to_string())


if __name__ == "__main__":
    main()
