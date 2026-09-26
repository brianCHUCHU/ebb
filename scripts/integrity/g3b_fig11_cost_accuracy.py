"""Task 3b: cost-accuracy scatter for Carparts.

x = log-scaled seconds per series (from efficiency_carparts.csv, same
machine/session), y = mean SPL (from the unified rescoring). Table-2
methods are filled circles; external reference points that use subset or
proxy protocols (DeepState full-panel MXNet run, Chronos-Bolt 500-series
zero-shot) are open diamonds. No GPU was used by any method (all rows
CPU); the marker distinction is protocol, not hardware. Zero (no cost,
diagnostic) and TSB-tuned (point-only, no predictive distribution) are
omitted. No annotations beyond method names, per instructions.

Output: outputs/<date>/fig11_cost_accuracy_carparts.pdf (+ .png)

Usage: py scripts/integrity/g3b_fig11_cost_accuracy.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / date.today().isoformat()

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
})

REMIX_C, GREY = "#D55E00", "#555555"

SPL = {  # unified rescoring, Carparts mean SPL
    "EBB": 0.3229, "TweedieGP": 0.3214, "iETS": 0.4561,
    "AutoARIMA": 0.4084, "AutoTheta": 0.3925, "Tweedie-GLM": 0.4776,
    "CP-Croston": 0.4012, "CP-SBA": 0.3982, "CP-TSB": 0.3783,
    "CP-ADIDA": 0.3571, "CP-IMAPA": 0.3561,
}
REFERENCE_SPL = {"DeepState": 0.4608, "Chronos-Bolt-small": 0.3622}

LABEL_OFF = {
    "EBB": (6, 0, "left"), "TweedieGP": (-4, 7, "right"),
    "iETS": (6, -2, "left"), "AutoARIMA": (6, -2, "left"),
    "AutoTheta": (6, -2, "left"), "Tweedie-GLM": (6, -2, "left"),
    "CP-Croston": (6, 2, "left"), "CP-SBA": (6, -8, "left"),
    "CP-TSB": (6, -2, "left"), "CP-ADIDA": (6, 4, "left"),
    "CP-IMAPA": (6, -9, "left"), "DeepState": (-8, -3, "right"),
    "Chronos-Bolt-small": (6, -2, "left"),
}


def main() -> None:
    eff = pd.read_csv(OUT / "efficiency_carparts.csv").set_index("method")
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for name, spl in SPL.items():
        x = float(eff.loc[name, "s_per_series"])
        ax.scatter([x], [spl], s=34,
                   color=(REMIX_C if name == "EBB" else GREY),
                   marker="o", zorder=3)
        dx, dy, ha = LABEL_OFF.get(name, (6, -2, "left"))
        ax.annotate(name, (x, spl), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7.8,
                    color=(REMIX_C if name == "EBB" else "#333333"))
    for name, spl in REFERENCE_SPL.items():
        if name not in eff.index:
            continue
        x = float(eff.loc[name, "s_per_series"])
        ax.scatter([x], [spl], s=40, facecolors="none", edgecolors=GREY,
                   marker="D", zorder=3)
        dx, dy, ha = LABEL_OFF.get(name, (6, -2, "left"))
        ax.annotate(name, (x, spl), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7.8, color="#333333")
    ax.set_xscale("log")
    ax.set_xlim(3.2e-3, 2.2e-1)
    ax.set_ylim(0.305, 0.49)
    ax.set_xlabel("seconds per series (wall-clock, CPU)")
    ax.set_ylabel("mean SPL (Carparts)")
    ax.scatter([], [], s=34, color=GREY, marker="o", label="fixed-origin, full panel")
    ax.scatter([], [], s=40, facecolors="none", edgecolors=GREY, marker="D",
               label="reference protocol (subset/proxy)")
    ax.legend(frameon=False, fontsize=7.8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig11_cost_accuracy_carparts.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig11_cost_accuracy_carparts.png", dpi=200, bbox_inches="tight")
    print("wrote fig11_cost_accuracy_carparts.pdf/.png")


if __name__ == "__main__":
    main()
