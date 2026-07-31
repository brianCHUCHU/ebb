"""Regenerate figs/fig_resolution.pdf from the CORRECTED experiments only.

The first version of this figure drew panels A and C from the retracted
difficulty-confounded grid. This version uses:
  A  synthetic_sanity/grid.csv   (symmetric generator, oracle ladder)
  B  synthetic_resolution/collapse_calibration.csv  (unchanged, valid)
  C  synthetic_sanity/extreme.csv (structure fixed, series length swept)

Panel A: forgetting raises the leverage of a correct partition (NLL gain vs 1-w).
Panel B: collapse probability follows Phi(-z).
Panel C: the gain is governed by the item's own sample size, on every metric.
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
GRID = ROOT / "outputs" / "synthetic_sanity" / "grid.csv"
CAL = ROOT / "outputs" / "synthetic_resolution" / "collapse_calibration.csv"
EXTREME = ROOT / "outputs" / "synthetic_sanity" / "extreme.csv"
OUT = ROOT / "paper_v2" / "figs" / "fig_resolution.pdf"


def main() -> None:
    grid = pd.read_csv(GRID)
    cal = pd.read_csv(CAL)
    ext = pd.read_csv(EXTREME)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # ---- A: NLL gain of the true partition vs forgetting strength
    ax = axes[0]
    w = grid.pivot_table(index=["separation", "T", "discount", "rep"], columns="arm", values="nll")
    gain = 100.0 * (w["global-estimated"] - w["oracle-labels+estimated-hypers"]) / w["global-estimated"]
    for T, marker in ((30, "o"), (120, "s")):
        sub = gain.xs(T, level="T").groupby(level="discount")
        mean, se = sub.mean(), sub.std(ddof=1) / np.sqrt(sub.count())
        x = 1.0 - mean.index.to_numpy(dtype=float)
        order = np.argsort(x)
        ax.errorbar(x[order], mean.to_numpy()[order], yerr=se.to_numpy()[order],
                    marker=marker, capsize=3, label=f"$T={T}$")
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"forgetting strength $1-w$")
    ax.set_ylabel("NLL gain of true partition\nover single pool (%)")
    ax.set_title("A. Forgetting raises the leverage\nof a correct partition")
    ax.legend(fontsize=8, frameon=False)

    # ---- B: collapse calibration
    ax = axes[1]
    sub = cal.sort_values("z")
    ax.scatter(sub["z"], sub["p_empirical"], s=7, alpha=0.35, label="simulated")
    zz = np.linspace(0.0, 4.0, 200)
    ax.plot(zz, norm.cdf(-zz), color="k", lw=1.6, label=r"$\Phi(-z)$")
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"$z=\rho_g\sqrt{(m_g-1)/2}$")
    ax.set_ylabel(r"$\Pr[\hat\tau_g^2=0]$")
    ax.set_title("B. Collapse probability follows\nthe resolution statistic")
    ax.legend(fontsize=8, frameon=False)

    # ---- C: gain vs the item's own sample size (extreme sweep)
    ax = axes[2]
    piv = {m: ext.pivot_table(index="median_n_pos", columns="arm", values=m)
           for m in ("pinball", "nll", "brier_occurrence", "log_size_mse")}
    styles = {"nll": ("NLL", "o-"), "pinball": ("pinball", "s-"),
              "brier_occurrence": ("Brier", "^-"), "log_size_mse": ("log-size MSE", "d-")}
    for m, (label, style) in styles.items():
        p = piv[m]
        g = 100.0 * (p["global-estimated"] - p["oracle-labels+estimated-hypers"]) / p["global-estimated"]
        ax.plot(g.index.to_numpy(dtype=float), g.to_numpy(), style, label=label, ms=4)
    ax.set_xscale("log")
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"median positive observations per item $n_i^{+}$")
    ax.set_ylabel("gain of true partition\nover single pool (%)")
    ax.set_title("C. The gain is governed by the\nitem's own sample, on every metric")
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
