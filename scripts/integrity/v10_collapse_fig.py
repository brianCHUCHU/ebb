"""Appendix figure: collapse probability against the resolution coordinate.

Single-panel version of the former mechanism figure (panel (b) of
g6_mechanism_two_panel.py); its schematic panel (a) is now Figure 1 of
the main text. Data: outputs/synthetic_resolution/collapse_calibration.csv
(unchanged).

Output: paper_v2/v8_paper_runs/figs/fig_app_collapse.pdf

Usage: py scripts/integrity/v10_collapse_fig.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_v2" / "v8_paper_runs" / "figs"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
})

cal = pd.read_csv(ROOT / "outputs" / "synthetic_resolution" / "collapse_calibration.csv")

fig, ax = plt.subplots(figsize=(3.9, 2.7))
ax.scatter(cal["z"], cal["p_empirical"], s=12, color="#8FCBEA", alpha=0.75,
           linewidths=0, label="simulated", zorder=2)
zg = np.linspace(0, 4, 300)
ax.plot(zg, norm.cdf(-zg), color="black", lw=1.6, label=r"$\Phi(-z)$", zorder=3)
ax.set_xlim(-0.05, 4)
ax.set_ylim(0, 0.68)
ax.set_xlabel(r"resolution $z=\rho_g\sqrt{(m_g-1)/2}$")
ax.set_ylabel(r"collapse probability $\Pr[\hat\tau_g^2=0]$")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

fig.tight_layout()
fig.savefig(OUT / "fig_app_collapse.pdf", bbox_inches="tight")
print("wrote", OUT / "fig_app_collapse.pdf", "| cells:", len(cal))
