"""Task 6: two-panel mechanism figure (drops the old panel (b), whose
content is superseded by the simulated envelope in fig10a).

Panel (a): schematic -- one discount, two consequences.
Panel (b): collapse probability vs resolution statistic, from
outputs/synthetic_resolution/collapse_calibration.csv (unchanged data).

Output: paper_v2/v5/fig01_mechanism_v2.pdf (+ .png preview)

Usage: py scripts/integrity/g6_mechanism_two_panel.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_v2" / "v5"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, ORANGE, PINK = "#0072B2", "#D55E00", "#CC79A7"

fig, (ax0, ax1) = plt.subplots(
    1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.05, 1.0]})

# ---------------------------------------------------------------- panel (a)
ax0.set_axis_off()
ax0.set_xlim(0, 10)
ax0.set_ylim(0, 10)
ax0.text(-0.6, 10.3, r"$\mathbf{(a)}$", fontsize=11)
ax0.text(5.0, 10.3, "One discount, two consequences", ha="center", fontsize=10)

xs = np.linspace(2.4, 8.2, 9)
ax0.scatter(xs, np.full_like(xs, 8.9), s=28, color=BLUE, alpha=0.9, zorder=3)
ax0.text(1.9, 8.9, "weak forgetting", ha="right", va="center", fontsize=8.5)
alphas = np.linspace(0.15, 1.0, len(xs))
sizes = np.linspace(10, 42, len(xs))
for x, a, s in zip(xs, alphas, sizes):
    ax0.scatter([x], [7.0], s=s, color=ORANGE, alpha=a, zorder=3)
ax0.text(1.9, 7.0, "strong forgetting", ha="right", va="center", fontsize=8.5)
ax0.annotate("", xy=(8.2, 6.1), xytext=(2.4, 6.1),
             arrowprops=dict(arrowstyle="->", color="grey", lw=1.0))
ax0.text(2.4, 5.7, "older observations", color="grey", fontsize=8, va="top")
ax0.text(8.2, 5.7, "recent", color="grey", fontsize=8, va="top", ha="right")

ax0.text(5.2, 4.5, r"effective history $n_i(w)\ \downarrow$",
         ha="center", fontsize=9.5)
for xbox, ybox, w, h, c, lines in (
    (0.9, 0.6, 3.6, 2.0, BLUE, "item level\nprior weight $1-\\lambda_i\\ \\uparrow$"),
    (5.5, 0.6, 3.9, 2.0, PINK, "group level\nresolution $\\hat z_g\\ \\downarrow$"),
):
    ax0.add_patch(FancyBboxPatch(
        (xbox, ybox), w, h, boxstyle="round,pad=0.15",
        linewidth=1.2, edgecolor=c, facecolor=c, alpha=0.999,
        fill=False, zorder=2))
    ax0.text(xbox + w / 2, ybox + h / 2, lines, ha="center", va="center",
             fontsize=8.5)
for xt in (2.7, 7.4):
    ax0.add_patch(FancyArrowPatch(
        (5.2, 4.1), (xt, 3.0), arrowstyle="->", color="grey",
        lw=1.0, shrinkA=2, shrinkB=2))

# ---------------------------------------------------------------- panel (b)
cal = pd.read_csv(ROOT / "outputs" / "synthetic_resolution" / "collapse_calibration.csv")
ax1.scatter(cal["z"], cal["p_empirical"], s=12, color="#8FCBEA", alpha=0.75,
            linewidths=0, label="simulated", zorder=2)
zg = np.linspace(0, 4, 300)
ax1.plot(zg, norm.cdf(-zg), color="black", lw=1.6, label=r"$\Phi(-z)$", zorder=3)
ax1.set_xlim(-0.05, 4)
ax1.set_ylim(0, 0.68)
ax1.set_xlabel(r"resolution $z=\rho_g\sqrt{(m_g-1)/2}$")
ax1.set_ylabel(r"collapse probability $\Pr[\hat\tau_g^2=0]$")
ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
ax1.text(-0.72, 0.705, r"$\mathbf{(b)}$", fontsize=11)
ax1.set_title("Ability to resolve groups", fontsize=10)

fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "fig01_mechanism_v2.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig01_mechanism_v2.png", dpi=200, bbox_inches="tight")
print("wrote fig01_mechanism_v2.pdf/.png")
