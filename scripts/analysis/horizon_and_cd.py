"""C3: horizon-wise error decomposition (Online Retail).
B6: Friedman + Nemenyi critical-difference diagram and Diebold-Mariano tests
across datasets, from saved per-timestamp predictions.

Usage: py scripts/analysis/horizon_and_cd.py
Outputs: paper_v2/figs/horizon_decomp.pdf, paper_v2/figs/cd_diagram.pdf,
         outputs/paper_runs/dm_tests.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper_runs"
FIGS = ROOT / "paper_v2" / "figs"

plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "pdf.fonttype": 42,
})
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999"]

MODELS = ["CrostonClassic", "CrostonSBA", "TSB", "ADIDA", "IMAPA", "AutoARIMA", "AutoTheta"]

PRED_SOURCES = {
    "Online Retail": ("or_point_fixed", "or_point_ebb"),
    "M5": ("m5_point_fixed_saved", "m5_point_ebb"),
    "Auto": ("auto_point_fixed", "auto_point_ebb"),
    "Carparts": ("carparts_point_fixed", "carparts_point_ebb"),
    "RAF": ("raf_point_fixed", "raf_point_ebb"),
}


def load_joint(ds: str):
    base_d, ebb_d = PRED_SOURCES[ds]
    base_p = OUT / base_d / "point_predictions.csv.gz"
    ebb_p = OUT / ebb_d / "point_predictions.csv.gz"
    if not base_p.exists() or not ebb_p.exists():
        return None
    base = pd.read_csv(base_p)
    ebb = pd.read_csv(ebb_p)[["unique_id", "ds", "EB-Hurdle"]].rename(columns={"EB-Hurdle": "EBB"})
    joint = base.merge(ebb, on=["unique_id", "ds"], how="inner")
    return joint


def fig_horizon(joint: pd.DataFrame) -> None:
    joint = joint.sort_values(["unique_id", "ds"]).copy()
    joint["h"] = joint.groupby("unique_id").cumcount() + 1
    joint["bucket"] = pd.cut(joint["h"], bins=[0, 7, 14, 28, 56, 112, 10**9],
                             labels=["1-7", "8-14", "15-28", "29-56", "57-112", "113+"])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    show = [("EBB", "EBB", C[1], "-", 2.0), ("TSB", "TSB", C[0], "--", 1.2),
            ("ADIDA", "ADIDA", C[2], "--", 1.2), ("IMAPA", "IMAPA", C[5], ":", 1.2),
            ("AutoARIMA", "AutoARIMA", C[3], ":", 1.2)]
    for colname, label, color, ls, lw in show:
        g = joint.groupby("bucket", observed=True).apply(
            lambda d: float(np.sqrt(((d[colname] - d.y) ** 2).mean())), include_groups=False)
        ax.plot(range(len(g)), g.values, ls, marker="o", ms=2.5, lw=lw, color=color, label=label)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(list(g.index), fontsize=7)
    ax.set_xlabel("forecast horizon bucket (days ahead)")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=7, ncol=2, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(FIGS / "horizon_decomp.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote horizon_decomp.pdf")


def per_series_scaled_sqerr(joint: pd.DataFrame, models: list[str], init_naive_mse: pd.Series) -> pd.DataFrame:
    """Per-series RMSSE^2 (scaled squared error), the paper's primary point metric."""
    out = {}
    for m in models:
        se = (joint[m] - joint.y) ** 2
        mse = se.groupby(joint.unique_id).mean()
        out[m] = mse / init_naive_mse.reindex(mse.index)
    df = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).dropna()
    return df


def load_init_naive_mse(ds: str) -> pd.Series:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from data_loading import (load_online_retail, preprocess_online_retail,
                              train_eval_split_fixed_origin, load_generic_long,
                              train_eval_split_last_h)
    if ds == "Online Retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data" / "online_retail.csv"))
        init, _ = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    elif ds == "M5":
        from data_loading import load_m5_long, preprocess_m5
        from utils import set_seed, default_m5_sales_file, default_m5_calendar_file
        set_seed(42)
        sales, cal = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        df = preprocess_m5(sales, cal, sample_size=5000)
        init, _ = train_eval_split_fixed_origin(df, init_ratio=2 / 3, min_len=1)
    else:
        h = {"Auto": 6, "Carparts": 6, "RAF": 12}[ds]
        df = load_generic_long(ROOT / "data" / f"{ds.lower()}_long.csv")
        init, _ = train_eval_split_last_h(df, h=h)
    s = init.sort_values(["unique_id", "ds"]).copy()
    s["d2"] = (s.y - s.groupby("unique_id").y.shift(1)) ** 2
    denom = s.groupby("unique_id")["d2"].mean()
    return denom[denom > 0]


def cd_and_dm() -> None:
    rank_blocks = []
    dm_rows = []
    models = MODELS + ["EBB"]
    for ds in PRED_SOURCES:
        joint = load_joint(ds)
        if joint is None:
            print(f"{ds}: predictions missing, skipped")
            continue
        denom = load_init_naive_mse(ds)
        if denom.empty:
            print(f"{ds}: init denominators unavailable yet, skipped")
            continue
        ps = per_series_scaled_sqerr(joint, [m for m in models if m in joint.columns], denom)
        rank_blocks.append(ps.rank(axis=1))
        # DM-style test per dataset on per-series scaled squared errors
        for m in MODELS:
            if m not in ps.columns:
                continue
            d = (ps["EBB"] - ps[m]).dropna()
            t, p = stats.ttest_1samp(d, 0.0)
            dm_rows.append({"dataset": ds, "vs": m, "mean_scaled_loss_diff": float(d.mean()),
                            "t": float(t), "p": float(p), "n_series": int(len(d))})
        print(f"{ds}: ranked over {len(ps)} series")

    if not rank_blocks:
        return
    ranks_all = pd.concat(rank_blocks, ignore_index=True).dropna()
    avg_rank = ranks_all.mean(axis=0).sort_values()
    k = ranks_all.shape[1]
    n_blocks = len(ranks_all)
    fried_stat, fried_p = stats.friedmanchisquare(*[ranks_all[c].values for c in ranks_all.columns])
    # Nemenyi CD with per-series blocks pooled across datasets
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}[k]
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    n_datasets = len(rank_blocks)

    fig, ax = plt.subplots(figsize=(3.6, 1.9))
    y = 0
    for m, r in avg_rank.items():
        color = C[1] if m == "EBB" else "#444444"
        ax.plot([r], [y], "o", ms=5, color=color)
        ax.annotate(f" {m} ({r:.2f})", (r, y), fontsize=7, va="center", color=color)
        y -= 1
    ax.errorbar([avg_rank.min()], [y], xerr=[[0], [cd]], fmt="none", capsize=3, color="#000000")
    ax.annotate(f" CD={cd:.2f} (Nemenyi, $\\alpha$=0.05)", (avg_rank.min(), y), fontsize=7, va="center")
    ax.set_yticks([])
    ax.set_xlabel(
        f"avg.\\ rank of per-series scaled sq.\\ error "
        f"({n_blocks} series, {n_datasets} datasets; Friedman p={fried_p:.2g})",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "cd_diagram.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(dm_rows).to_csv(OUT / "dm_tests.csv", index=False)
    print("wrote cd_diagram.pdf, dm_tests.csv")
    print(avg_rank.round(3).to_string())
    print(f"Friedman chi2={fried_stat:.2f} p={fried_p:.4g}; CD={cd:.3f}")


if __name__ == "__main__":
    joint_or = load_joint("Online Retail")
    if joint_or is not None:
        fig_horizon(joint_or)
    cd_and_dm()
