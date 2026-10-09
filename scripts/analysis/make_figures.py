"""Generate the main-text figures for paper_v2 (PDF, consistent style).

Outputs to paper_v2/figs/:
  discount_curves.pdf — validation scaled pinball vs discount w, all datasets
  spl_profile.pdf     — SPL by quantile, Online Retail, all probabilistic models
  drift_tradeoff.pdf  — theory: bias/variance bounds and their sum vs 1-w

Usage: py scripts/analysis/make_figures.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper_runs"
FIGS = ROOT / "paper_v2" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)

# Okabe-Ito colorblind-safe palette
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999", "#F0E442"]


def fig_discount_curves() -> None:
    runs = {
        "Online Retail": "or_point_fixed",
        "M5": "m5_point_fixed",
        "Auto": "auto_point_fixed",
        "Carparts": "carparts_point_fixed",
        "RAF": "raf_point_fixed",
    }
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, (name, d) in enumerate(runs.items()):
        p = OUT / d / "ebhurdle_variant_diagnostics.csv"
        if not p.exists():
            continue
        diag = pd.read_csv(p)
        row = diag[diag["variant"] == "EB-Hurdle-Discount"]
        if row.empty or pd.isna(row.iloc[0].get("discount_grid")):
            continue
        grid = ast.literal_eval(row.iloc[0]["discount_grid"])
        g = pd.DataFrame(grid).sort_values("discount")
        rel = g["scaled_pinball"] / g.loc[g["discount"] == 1.0, "scaled_pinball"].iloc[0]
        ax.plot(1.0 - g["discount"], rel, marker="o", ms=2.5, lw=1.2, color=C[i], label=name)
        w_star = g.loc[g["scaled_pinball"].idxmin(), "discount"]
        rel_star = g["scaled_pinball"].min() / g.loc[g["discount"] == 1.0, "scaled_pinball"].iloc[0]
        ax.plot([1.0 - w_star], [rel_star], marker="*", ms=9, color=C[i], zorder=5)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"forgetting strength $1-w$")
    ax.set_ylabel("validation SPL (rel. to $w{=}1$)")
    ax.legend(fontsize=7, loc="upper left", ncol=1, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(FIGS / "discount_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote discount_curves.pdf")


def fig_spl_profile() -> None:
    base = pd.read_csv(OUT / "or_prob_fixed_paper" / "prob_pinball_scaled.csv")
    ebb = pd.read_csv(OUT / "or_prob_select" / "prob_pinball_scaled.csv")
    ebb = ebb[ebb["model"] == "EB-Hurdle"]
    spl = pd.concat([base[base["model"] != "EB-Hurdle"], ebb], ignore_index=True)
    spl["quantile"] = spl["quantile"].astype(str)
    spl = spl[spl["quantile"] != "mean"]
    spl["q"] = spl["quantile"].astype(float)
    order = [
        ("EB-Hurdle", "EBB (ours)", C[1], "-", 2.0),
        ("AutoARIMA", "AutoARIMA", C[0], "--", 1.1),
        ("AutoTheta", "AutoTheta", C[2], "--", 1.1),
        ("CP-TSB", "CP-TSB", C[3], ":", 1.1),
        ("CP-ADIDA", "CP-ADIDA", C[4], ":", 1.1),
        ("CP-IMAPA", "CP-IMAPA", C[5], ":", 1.1),
        ("CP-CrostonSBA", "CP-SBA", C[6], ":", 1.1),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for model, label, color, ls, lw in order:
        sub = spl[spl["model"] == model].sort_values("q")
        if sub.empty:
            continue
        ax.plot(sub["q"], sub["scaled_pinball"], ls, color=color, lw=lw, marker="o", ms=2.5, label=label)
    ax.set_xlabel("quantile level $q$")
    ax.set_ylabel("scaled pinball loss")
    ax.set_xticks([0.1, 0.25, 0.5, 0.75, 0.9])
    ax.legend(fontsize=6.5, loc="upper left", ncol=2, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIGS / "spl_profile.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote spl_profile.pdf")


def fig_drift_tradeoff() -> None:
    b = np.linspace(1e-4, 0.5, 400)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, delta in enumerate([0.0005, 0.002, 0.008]):
        bias2 = (delta * (1 - b) / b) ** 2
        var = 0.25 * b / (2 - b)
        mse = bias2 + var
        ax.plot(b, mse, color=C[i], lw=1.3, label=rf"$\delta={delta}$")
        bstar = b[np.argmin(mse)]
        ax.plot([bstar], [mse.min()], marker="*", ms=9, color=C[i], zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"forgetting strength $1-w$")
    ax.set_ylabel("worst-case MSE bound")
    ax.legend(fontsize=7, handlelength=1.4)
    fig.tight_layout()
    fig.savefig(FIGS / "drift_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote drift_tradeoff.pdf")


if __name__ == "__main__":
    fig_discount_curves()
    fig_spl_profile()
    fig_drift_tradeoff()
