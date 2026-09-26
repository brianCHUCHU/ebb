"""R4: transfer of the pre-fit room diagnostic from the synthetic grid to the
semi-synthetic real-series panels (docs/DESIGN_room_diagnostic.md).

R1b needs no fitting. R2 (logistic on log1p features, Delta > eps) is fitted
once on ALL synthetic panels and applied unchanged to every semi-synthetic
row. Reports, per semi-synthetic file: predicted probability, R1b, realized
oracle gain, and AUC / calibration where both classes occur.

Outputs (outputs/<date>/): room_transfer.csv, room_transfer_summary.csv,
  fig16_room_transfer.{pdf,png}
Usage: py scripts/integrity/r4_room_transfer.py [semi csv ...]
"""
from __future__ import annotations

import glob
import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / date.today().isoformat()
FIGS = ROOT / "paper_v2" / "figs_v3"
FEATS = ["lev_size", "het_size", "z_size", "lev_occ", "het_occ", "z_occ", "median_n_pos", "median_n_obs", "m_items"]
EPS = (0.5, 1.0)
INK, MUTED, ACCENT, BLUE = "#222222", "#9a9a9a", "#D55E00", "#0072B2"
PNL = {"online_retail": "Online Retail", "carparts": "Carparts", "m5": "M5"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def design(df):
    return np.log1p(df[FEATS].clip(lower=0).to_numpy(float))


def main():
    syn = pd.read_csv(OUT / "room_synthetic.csv").dropna(subset=["delta_oracle"])
    files = sys.argv[1:] or sorted(glob.glob(str(OUT / "room_semisynthetic*.csv")))
    semi = pd.concat([pd.read_csv(f).assign(source=Path(f).stem) for f in files], ignore_index=True)
    semi["R1b"] = semi["lev_size"] ** 2 * semi["tau2_global"]
    Xs = design(syn); sc = StandardScaler().fit(Xs)
    for eps in EPS:
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xs), (syn["delta_oracle"] > eps).astype(int))
        semi[f"p_R2_{eps}"] = clf.predict_proba(sc.transform(design(semi)))[:, 1]
    semi.to_csv(OUT / "room_transfer.csv", index=False)
    rows = []
    for (src, panel), g in semi.groupby(["source", "panel"]):
        rec = {"source": src, "panel": panel, "n": len(g), "w": float(g["w"].iloc[0]),
               "delta_min": g["delta_oracle"].min(), "delta_max": g["delta_oracle"].max(),
               "share_gt1": float((g["delta_oracle"] > 1).mean()), "share_gt05": float((g["delta_oracle"] > 0.5).mean()),
               "p_R2_1.0_mean": g["p_R2_1.0"].mean(), "p_R2_1.0_max": g["p_R2_1.0"].max(), "R1b_mean": g["R1b"].mean()}
        for eps in EPS:
            yb = (g["delta_oracle"] > eps).astype(int)
            rec[f"auc_R2_{eps}"] = roc_auc_score(yb, g[f"p_R2_{eps}"]) if yb.nunique() > 1 else np.nan
            rec[f"auc_R1b_{eps}"] = roc_auc_score(yb, g["R1b"]) if yb.nunique() > 1 else np.nan
        rows.append(rec)
    summ = pd.DataFrame(rows); summ.round(4).to_csv(OUT / "room_transfer_summary.csv", index=False)
    pd.set_option("display.width", 220); print(summ.round(3).to_string(index=False))
    yb = (semi["delta_oracle"] > 1).astype(int)
    if yb.nunique() > 1:
        print("pooled semi-synthetic AUC(>1%): R2", round(roc_auc_score(yb, semi["p_R2_1.0"]), 3), "R1b", round(roc_auc_score(yb, semi["R1b"]), 3))
    # figure: predicted P(Delta>1%) vs realized gain, by panel and w
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    marks = {"carparts": "D", "online_retail": "o", "m5": "s"}
    for (panel, w), g in semi.groupby(["panel", "w"]):
        ax.scatter(g["p_R2_1.0"], g["delta_oracle"], marker=marks.get(panel, "o"), s=22,
                   color=ACCENT if w < 0.95 else BLUE, edgecolor="white", linewidth=0.6, label=f"{PNL.get(panel, panel)}, $w$={w:g}", alpha=0.9)
    ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 3))); ax.axhline(0, color=MUTED, lw=0.4)
    ax.set_xlabel(r"predicted $\Pr(\Delta_{\rm oracle}>1\%)$ from the synthetic grid")
    ax.set_ylabel("realized oracle gain (% SPL)")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="best")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig16_room_transfer.pdf", bbox_inches="tight"); fig.savefig(FIGS / "fig16_room_transfer.png", bbox_inches="tight", dpi=200)
    for d in ("v6_aistats", "v5_aistats"):
        if (ROOT / "paper_v2" / d / "figs").exists():
            shutil.copy(FIGS / "fig16_room_transfer.pdf", ROOT / "paper_v2" / d / "figs" / "fig16_room_transfer.pdf")
    print("wrote room_transfer.csv, room_transfer_summary.csv, fig16_room_transfer")


if __name__ == "__main__":
    main()
