"""P0-2: the pooling x forgetting point grid on M5 (backfills tab:leverage).

Reproduces, on the M5 5,000-series sample, the same six-cell grid the paper
reports for the other panels (tab:ablation / ablate_*): each cell is an
independent configuration with its own discount resolution and a full-window
refit, mirroring the non-M5 CLI semantics of run_point exactly:

  - global   : labels None; 'auto' selects w with labels=None
  - taxonomy : ADI/CV^2 labels; 'auto' selects w WITH the taxonomy labels
               (run_point resolves the discount under the same base labels)
  - mixture  : 'auto' selects w with labels=None first, then learns the
               mixture on the discounted statistics (fit_discount=w)

Usage: py scripts/analysis/m5_partition_grid.py
Output: outputs/paper_runs/ablate_m5_point_grid.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from data_loading import load_m5_long, preprocess_m5, train_eval_split_fixed_origin
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed
from metrics import rmsse, compute_adi_cv2, classify_adi_cv2
from models.eb_hurdle import fit_eb_hurdle, predict_eb_hurdle, select_fit_discount
from models.mixture_pooling import mixture_group_labels

ITEM_VARIANCE_MODE = "conjugate"
VARIANCE_PRIOR_DF = 20.0


def build_taxonomy_labels(df: pd.DataFrame) -> pd.Series | None:
    feats = compute_adi_cv2(df)
    if feats.empty:
        return None
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)
    return feats.set_index("unique_id")["category"].astype(str)


def evaluate(init_set: pd.DataFrame, eval_set: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    m = eval_set[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
    m = m.rename(columns={"yhat": "y_pred"})
    mae = float((m.y_pred - m.y).abs().mean())
    rmse = float(np.sqrt(((m.y_pred - m.y) ** 2).mean()))
    rm = rmsse(init_set, m)
    return {"MAE": mae, "RMSE": rmse, "RMSSE": rm}


def main() -> None:
    set_seed(42)
    sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
    df = preprocess_m5(sales, cal, sample_size=5000)
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
    print(f"M5 sample: {init_set.unique_id.nunique()} series")

    taxonomy_labels = build_taxonomy_labels(init_set)

    rows = []
    for grouping in ("global", "taxonomy", "mixture"):
        for spec in ("1.0", "auto"):
            t0 = time.perf_counter()
            if spec == "auto":
                base_labels = taxonomy_labels if grouping == "taxonomy" else None
                w, _ = select_fit_discount(
                    init_set,
                    group_labels=base_labels,
                    item_variance_mode=ITEM_VARIANCE_MODE,
                    item_variance_shrink_strength=VARIANCE_PRIOR_DF,
                )
            else:
                w = 1.0
            mixture_k = None
            if grouping == "global":
                labels = None
            elif grouping == "taxonomy":
                labels = taxonomy_labels
            else:
                mix = mixture_group_labels(init_set, k=0, fit_discount=w)
                labels = mix.labels
                mixture_k = mix.k
            params = fit_eb_hurdle(
                init_set,
                group_labels=labels,
                group_shrink_strength=0.0,
                item_variance_mode=ITEM_VARIANCE_MODE,
                item_variance_shrink_strength=VARIANCE_PRIOR_DF,
                fit_discount=w,
            )
            pred = predict_eb_hurdle(params, eval_set, quantiles=None)
            r = evaluate(init_set, eval_set, pred)
            r.update({
                "dataset": "m5", "grouping": grouping, "discount_spec": spec,
                "selected_w": w, "mixture_k": mixture_k,
                "seconds": time.perf_counter() - t0,
            })
            rows.append(r)
            print(f"{grouping:9s} w_spec={spec:4s} -> w={w:.4g} K={mixture_k} "
                  f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} RMSSE={r['RMSSE']:.4f} "
                  f"({r['seconds']:.0f}s)", flush=True)

    out = pd.DataFrame(rows)[[
        "dataset", "grouping", "discount_spec", "selected_w", "mixture_k",
        "MAE", "RMSE", "RMSSE", "seconds",
    ]]
    out_path = ROOT / "outputs" / "paper_runs" / "ablate_m5_point_grid.csv"
    out.to_csv(out_path, index=False)
    print(out.round(4).to_string(index=False))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
