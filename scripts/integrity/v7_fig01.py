"""v7 Figure 1: the diagnostic is right in both directions (both panels on the
credibility axis).

(a) Cold-start: EBB advantage over TweedieGP vs the median size-block credibility
    lambda+ at each retained history length L (outputs/2026-09-09/coldstart_wide.csv).
    Each panel is one polyline from full length (right, open marker) to L=6 (left).
(b) Controlled surface over credibility x separation: 36 simulated cells (squares)
    over a lightly interpolated background, with the five real panels projected via
    their learned-partition R^2 (outputs/2026-09-13/sep_surface_v2_cells.csv,
    separation_placement.csv).

Writes paper_v2/v7_paper/figs/fig01_pooling_pays.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "paper_v2" / "v7_paper" / "figs"
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}
COLORS = {"online_retail": "#0072B2", "m5": "#009E73", "auto": "#D55E00", "carparts": "#CC79A7", "raf": "#E69F00"}
MARK = {"online_retail": "o", "m5": "s", "auto": "^", "carparts": "D", "raf": "v"}
INK, MUTED = "#222222", "#9a9a9a"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK})


def main() -> None:
    wide = pd.read_csv(ROOT / "outputs" / "2026-09-09" / "coldstart_wide.csv")
    cells = pd.read_csv(ROOT / "outputs" / "2026-09-13" / "sep_surface_v2_cells.csv")
    place = pd.read_csv(ROOT / "outputs" / "2026-09-13" / "separation_placement.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), gridspec_kw={"width_ratios": [1, 1.18]})

    # ---- (a) cold start vs retained history (credibility is not monotone in L on
    # Carparts and collapses to 0 on RAF at L=6, so the history axis reads better) ----
    # Solid: within-model gain from the shared prior (EBB vs EBB with the prior off, both B=0;
    # outputs/2026-09-27/coldstart_nopool_wide.csv). Dashed: EBB's advantage over TweedieGP.
    ax = axes[0]
    nopool = pd.read_csv(ROOT / "outputs" / "2026-09-27" / "coldstart_nopool_wide.csv")
    for p in PANELS:
        sub = nopool[nopool["panel"] == p].sort_values("median_len")
        ax.plot(sub["median_len"], sub["adv_vs_TweedieGP"], lw=0.9, ls=(0, (3, 2)), color=COLORS[p],
                alpha=0.45)
        ax.plot(sub["median_len"], sub["pool_gain_mean_pct"], marker=MARK[p], ms=4.2, lw=1.3,
                color=COLORS[p], label=PNL[p], markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("history retained per series (periods, log)")
    ax.set_ylabel("gain from the shared prior (% SPL)")
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.5)
    ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=1.6, columnspacing=0.9)
    ax.set_title("(a) Pooling gains as credibility falls", fontsize=8, loc="left")

    # ---- (b) room surface: smooth field inside the simulated hull, faint cells,
    # zero contour, real panels labelled with their realized refinement gain ----
    ax = axes[1]
    z = cells["gain_true"].to_numpy(float)
    lam = cells["lambda_size"].to_numpy(float)
    r2 = cells["r2_true"].to_numpy(float)
    vmax = float(np.nanmax(np.abs(z)))
    gx, gy = np.meshgrid(np.linspace(0, 1, 161), np.linspace(0, 1.05, 161))
    gz = griddata(np.c_[lam, r2], z, (gx, gy), method="linear")   # NaN outside the hull
    ax.pcolormesh(gx, gy, gz, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.75, shading="auto",
                  zorder=1, rasterized=True)
    sc = ax.scatter(lam, r2, c=z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=14, marker="s",
                    edgecolor=MUTED, linewidth=0.3, alpha=0.9, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("oracle-partition gain over one pool (% SPL)", fontsize=7.2)
    cb.ax.tick_params(labelsize=7)
    OFF = {"online_retail": (7, 0, "left"), "carparts": (0, 10, "center"), "m5": (-7, -8, "right"),
           "auto": (-4, -28, "right"), "raf": (-7, 8, "right")}
    for _, r in place.iterrows():
        p = r["panel"]
        ax.scatter([r["lambda_size"]], [r["r2_learned"]], color=COLORS[p], marker=MARK[p], s=48,
                   zorder=5, edgecolor="white", linewidth=0.8)
        dx, dy, ha = OFF[p]
        arrow = dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=3) if p == "auto" else None
        ax.annotate(f"{PNL[p]} {r['realized_gain_pct']:+.2f}%", (r["lambda_size"], r["r2_learned"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.6,
                    color=COLORS[p], fontweight="bold", arrowprops=arrow, zorder=6)
    ax.text(0.02, 0.97, "most panels: little room to refine", transform=ax.transAxes, ha="left",
            va="top", fontsize=6.6, color=MUTED, style="italic")
    ax.set_xlabel(r"median size-block credibility $\lambda^{(+)}$")
    ax.set_ylabel(r"separation explained by partition $R^2$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.10)
    ax.set_title("(b) Little room for refinement", fontsize=8, loc="left")

    for a in axes:
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.tight_layout(w_pad=1.4)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig01_pooling_pays.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig01_pooling_pays.png", bbox_inches="tight", dpi=200)
    print("wrote", FIGS / "fig01_pooling_pays.pdf")


if __name__ == "__main__":
    main()
