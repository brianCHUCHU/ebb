"""W11: regret-versus-cost frontier figure (two panels, shared y).

Left  (fixed origin): x = measured Carparts wall-clock per series
      (outputs/2026-09-05/efficiency_carparts.csv), y = largest shortfall to
      the per-panel best over the five panels (regret_fixed.csv).
Right (walk-forward): x = total walk-forward wall-clock summed over the three
      monthly panels (Carparts + Auto + RAF; every method has a measured wall
      on all three), y = largest shortfall over the same three panels
      (recomputed from regret_wf.csv).
Style follows make_figures_v3 (accent for the EBB family, grey for others,
text in ink). Writes paper_v2/figs_v3/fig12_frontier.{pdf,png} and copies the
PDF into both manuscript figs/ directories.
Usage: py scripts/integrity/w11_fig_frontier.py [YYYY-MM-DD]
"""
from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
FIGS = ROOT / "paper_v2" / "figs_v3"
ACCENT, GREY, INK, MUTED = "#D55E00", "#7f7f7f", "#222222", "#9a9a9a"
MONTHLY = ["carparts", "auto", "raf"]
SHOW = ["AutoARIMA", "AutoTheta", "Tweedie-GLM", "iETS", "CP-Croston", "CP-SBA", "CP-TSB", "CP-ADIDA",
        "CP-IMAPA", "ACI-ADIDA", "TweedieGP", "EBB", "ACI-EBB", "EBB-rule"]
LABEL = {"EBB": "EBB", "ACI-EBB": "ACI-EBB", "EBB-rule": "EBB (rule)"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK})


def load():
    eff = pd.read_csv(ROOT / "outputs" / "2026-09-05" / "efficiency_carparts.csv").set_index("method")
    rf = pd.read_csv(OUT / "regret_fixed.csv").set_index("model")
    rw = pd.read_csv(OUT / "regret_wf.csv").set_index("model")
    cost = pd.read_csv(OUT / "wf_cost.csv").set_index("model")
    fixed = {}
    for m in SHOW:
        if m in rf.index and m in eff.index:
            fixed[m] = (float(eff.loc[m, "s_per_series"]), float(rf.loc[m, "max_regret"]))
    wf = {}
    for m in SHOW:
        if m in rw.index and m in cost.index and cost.loc[m, MONTHLY].notna().all() and rw.loc[m, MONTHLY].notna().all():
            wf[m] = (float(cost.loc[m, MONTHLY].sum()), float(rw.loc[m, MONTHLY].max()))
    return fixed, wf


def panel(ax, pts, xlabel, title, ymax, offsets):
    for m, (x, y) in pts.items():
        ours = m in ("EBB", "ACI-EBB", "EBB-rule")
        yy = min(y, ymax * 0.97)
        ax.scatter([x], [yy], s=34 if ours else 22, color=ACCENT if ours else GREY, zorder=3,
                   edgecolor="white", linewidth=0.8, clip_on=False)
        dx, dy, ha = offsets.get(m, (6, 0, "left"))
        txt = LABEL.get(m, m) + (f" ({y:.0f}, off scale)" if y > ymax else "")
        ax.annotate(txt, (x, yy), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=7.2, color=ACCENT if ours else INK, fontweight="bold" if ours else "normal")
    ax.set_xscale("log")
    ax.set_ylim(-2, ymax)
    ax.axhline(5, color=MUTED, lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.text(ax.get_xlim()[0] if False else None, 5, "", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.grid(True, which="major", axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    fixed, wf = load()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    off_f = {"EBB": (6, 6, "left"), "CP-Croston": (6, 5, "left"), "CP-SBA": (6, -5, "left"),
             "CP-ADIDA": (6, 5, "left"), "CP-IMAPA": (6, -5, "left"), "CP-TSB": (6, 0, "left"),
             "AutoARIMA": (6, 6, "left"), "AutoTheta": (6, -6, "left"), "iETS": (-6, 0, "right"),
             "TweedieGP": (-6, 7, "right"), "Tweedie-GLM": (6, 0, "left")}
    off_w = {"EBB": (-6, 7, "right"), "ACI-EBB": (6, -7, "left"), "EBB-rule": (6, 7, "left"), "CP-Croston": (6, 5, "left"),
             "CP-SBA": (6, -5, "left"), "CP-IMAPA": (6, 6, "left"), "CP-ADIDA": (6, -5, "left"),
             "ACI-ADIDA": (-6, 0, "right"), "CP-TSB": (6, 0, "left"), "iETS": (6, 0, "left"),
             "AutoARIMA": (6, -6, "left"), "AutoTheta": (6, 6, "left"), "TweedieGP": (-6, 7, "right")}
    panel(axes[0], fixed, "Carparts wall-clock per series (s)", "Fixed origin, five panels", 80, off_f)
    panel(axes[1], wf, "walk-forward wall-clock, monthly panels (s)", "Walk-forward, three monthly panels", 80, off_w)
    axes[0].set_ylabel("largest shortfall to per-panel best (%)")
    for ax, xf in zip(axes, (0.72, 0.5)):
        ax.text(xf, 5, "5%", transform=ax.get_yaxis_transform(), ha="center", va="bottom", fontsize=6.8, color=MUTED)
    fig.tight_layout(w_pad=1.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig12_frontier.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig12_frontier.png", bbox_inches="tight", dpi=200)
    for d in ("v5", "v5_aistats", "v6_aistats"):
        if (ROOT / "paper_v2" / d / "figs").exists():
            shutil.copy(FIGS / "fig12_frontier.pdf", ROOT / "paper_v2" / d / "figs" / "fig12_frontier.pdf")
    # single-column variant: walk-forward panel only (for the condensed manuscript)
    fig2, ax2 = plt.subplots(figsize=(3.4, 2.7))
    panel(ax2, wf, "walk-forward wall-clock, monthly panels (s)", "Walk-forward, three monthly panels", 80, off_w)
    ax2.set_ylabel("largest shortfall to per-panel best (%)")
    ax2.text(0.5, 5, "5%", transform=ax2.get_yaxis_transform(), ha="center", va="bottom", fontsize=6.8, color=MUTED)
    fig2.tight_layout()
    fig2.savefig(FIGS / "fig12b_frontier_wf.pdf", bbox_inches="tight")
    fig2.savefig(FIGS / "fig12b_frontier_wf.png", bbox_inches="tight", dpi=200)
    for d in ("v5", "v5_aistats", "v6_aistats"):
        if (ROOT / "paper_v2" / d / "figs").exists():
            shutil.copy(FIGS / "fig12b_frontier_wf.pdf", ROOT / "paper_v2" / d / "figs" / "fig12b_frontier_wf.pdf")
    print("fixed:", {k: (round(v[0], 3), round(v[1], 1)) for k, v in fixed.items()})
    print("wf:", {k: (round(v[0], 1), round(v[1], 1)) for k, v in wf.items()})


if __name__ == "__main__":
    main()
