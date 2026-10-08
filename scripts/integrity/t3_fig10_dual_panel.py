"""Integrity task 3: refinement-leverage figure, corrected, one block per panel.

Replaces the deprecated fig10 (which plotted four panels on lambda^(+) but
moved Auto alone to lambda^(o) -- a per-point axis choice that is not
defensible). Two figures, each with a single consistent axis:

  fig10a_refinement_leverage_size.{pdf,png}
      x = median lambda^(+) (size block). Five real-panel trajectories
      (hollow at w=1 -> filled at the corrected w*) plus the simulated
      trajectories (T=30, T=120 forgetting families; length sweep at w=1),
      whose lambda^(+) column is recorded by run_synthetic_sanity.
  fig10b_refinement_leverage_occ.{pdf,png}
      x = median lambda^(o) (occurrence block). The same five real panels;
      no simulated cells (the sanity study records the size diagnostic only).

y on both: internal refinement gain, 100*(R(global,w) - min(R(taxonomy,w),
R(mixture,w)))/R(global,w) on the corrected leakage-safe surfaces; for the
simulated cells it is the paired external pinball advantage of the TRUE
partition over the single pool (a different, more favorable quantity --
stated in the caption).

Inputs: outputs/integrity_<date>/{selection_surfaces.csv,
        leverage_dual_lambda_corrected.csv}, outputs/synthetic_sanity/*.csv
Usage:  py scripts/integrity/t3_fig10_dual_panel.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
SANITY = ROOT / "outputs" / "synthetic_sanity"

plt.rcParams.update(
    {
        "font.size": 8.5,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 200,
    }
)
CAT = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7"]
NEUTRAL = "#8a8a8a"
PANEL_LABEL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto",
               "carparts": "Carparts", "raf": "RAF"}
COLOR = {"Online Retail": CAT[0], "M5": CAT[1], "Auto": CAT[2],
         "Carparts": CAT[3], "RAF": CAT[4]}


def panel_points() -> pd.DataFrame:
    surf = pd.read_csv(OUT / "selection_surfaces.csv")
    lev = pd.read_csv(OUT / "leverage_dual_lambda_corrected.csv").set_index(["panel", "w"])
    rows = []
    for panel, df in surf.groupby("panel"):
        s = df.set_index(["structure", "discount"])["validation_spl"]
        _, w_star = s.idxmin()
        w_star = float(lev.loc[panel].index[lev.loc[panel].index != 1.0][0])
        for w, stage in ((1.0, "w=1"), (w_star, "w=w*")):
            g = s[("global", w)]
            refined = min(s[("taxonomy", w)], s[("mixture", w)])
            rec = lev.loc[(panel, w)]
            rows.append({
                "label": PANEL_LABEL[panel], "stage": stage, "w": w,
                "lambda_size": float(rec["median_lambda_size"]),
                "lambda_occ": float(rec["median_lambda_occ"]),
                "refinement_gain_pct": float(100.0 * (g - refined) / g),
            })
    return pd.DataFrame(rows)


def sim_points() -> pd.DataFrame:
    rows = []
    grid = pd.read_csv(SANITY / "grid.csv")
    keys = ["separation", "discount", "T", "rep"]
    piv = grid.pivot_table(index=keys, columns="arm", values="pinball")
    lam = grid[grid["arm"] == "global-estimated"].set_index(keys)["median_lambda_size"]
    gain = 100.0 * (piv["global-estimated"] - piv["oracle-labels+estimated-hypers"]) / piv["global-estimated"]
    cell = pd.DataFrame({"gain": gain, "lam": lam}).reset_index()
    for (T, w), c in cell.groupby(["T", "discount"]):
        rows.append({"family": f"sim-T{T}", "lambda_size": float(c["lam"].median()),
                     "refinement_gain_pct": float(c["gain"].mean())})
    extreme = pd.read_csv(SANITY / "extreme.csv")
    ep = extreme.pivot_table(index="T", columns="arm", values="pinball")
    el = extreme[extreme["arm"] == "global-estimated"].set_index("T")["median_lambda_size"]
    for T in ep.index:
        rows.append({"family": "sim-length", "lambda_size": float(el.loc[T]),
                     "refinement_gain_pct": float(
                         100.0 * (ep.loc[T, "global-estimated"] - ep.loc[T, "oracle-labels+estimated-hypers"])
                         / ep.loc[T, "global-estimated"])})
    return pd.DataFrame(rows)


def draw_panels(ax, pts: pd.DataFrame, xcol: str, offsets: dict) -> None:
    for label, sub in pts.groupby("label"):
        sub = sub.set_index("stage")
        c = COLOR[label]
        x0, y0 = sub.loc["w=1", [xcol, "refinement_gain_pct"]]
        x1, y1 = sub.loc["w=w*", [xcol, "refinement_gain_pct"]]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.9, alpha=0.65), zorder=3)
        ax.scatter([x0], [y0], s=26, facecolors="white", edgecolors=c, linewidths=1.1, zorder=4)
        ax.scatter([x1], [y1], s=30, color=c, zorder=4)
        dx, dy, ha = offsets[label]
        ax.annotate(label, (x1, y1), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.0, color=c, ha=ha)


def main() -> None:
    t0 = time.time()
    pts = panel_points()
    sims = sim_points()
    pd.concat([pts.assign(kind="panel"),
               sims.rename(columns={"family": "label"}).assign(kind="sim")],
              ignore_index=True).to_csv(OUT / "refinement_leverage_corrected.csv", index=False)
    print(pts.to_string(index=False))
    print(sims.to_string(index=False))

    # ---------------- fig10a: size block ----------------
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axhline(0.0, color=NEUTRAL, lw=0.7, zorder=1)
    for fam, marker, ls in (("sim-T30", "D", (0, (2, 2))),
                            ("sim-T120", "^", (0, (4, 2))),
                            ("sim-length", "s", (0, (1, 1.5)))):
        s = sims[sims["family"] == fam].sort_values("lambda_size")
        ax.plot(s["lambda_size"], s["refinement_gain_pct"], color=NEUTRAL, lw=0.8, ls=ls, zorder=2)
        ax.scatter(s["lambda_size"], s["refinement_gain_pct"], marker=marker, s=14,
                   facecolors="none", edgecolors=NEUTRAL, linewidths=0.9, zorder=2)
    offsets_a = {"Online Retail": (-7, -11, "right"), "M5": (4, 2, "left"),
                 "Auto": (-8, -8, "right"), "Carparts": (2, 6, "left"),
                 "RAF": (-9, 6, "right")}
    draw_panels(ax, pts, "lambda_size", offsets_a)
    proxies = [
        Line2D([], [], marker="o", color=NEUTRAL, mfc="white", ls="none", ms=5, label="panel, $w{=}1$"),
        Line2D([], [], marker="o", color=NEUTRAL, ls="none", ms=5.5, label="panel, corrected $w^{*}$"),
        Line2D([], [], marker="D", color=NEUTRAL, mfc="none", ls=(0, (2, 2)), ms=3.5, lw=0.8, label="simulated, $T{=}30$"),
        Line2D([], [], marker="^", color=NEUTRAL, mfc="none", ls=(0, (4, 2)), ms=3.5, lw=0.8, label="simulated, $T{=}120$"),
        Line2D([], [], marker="s", color=NEUTRAL, mfc="none", ls=(0, (1, 1.5)), ms=3.5, lw=0.8, label="simulated, length sweep"),
    ]
    ax.legend(handles=proxies, loc="upper left", handletextpad=0.4, borderaxespad=0.2,
              labelspacing=0.35)
    ax.set_xlabel(r"median size-block credibility $\lambda^{(+)}$ at the fitting discount")
    ax.set_ylabel("internal refinement gain (%)")
    ax.set_xlim(0.15, 1.03)
    fig.savefig(OUT / "fig10a_refinement_leverage_size.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig10a_refinement_leverage_size.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- fig10b: occurrence block ----------------
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axhline(0.0, color=NEUTRAL, lw=0.7, zorder=1)
    offsets_b = {"Online Retail": (-7, -11, "right"), "M5": (4, 2, "left"),
                 "Auto": (4, -9, "left"), "Carparts": (4, 3, "left"),
                 "RAF": (4, -3, "left")}
    draw_panels(ax, pts, "lambda_occ", offsets_b)
    proxies = [
        Line2D([], [], marker="o", color=NEUTRAL, mfc="white", ls="none", ms=5, label="panel, $w{=}1$"),
        Line2D([], [], marker="o", color=NEUTRAL, ls="none", ms=5.5, label="panel, corrected $w^{*}$"),
    ]
    ax.legend(handles=proxies, loc="upper left", handletextpad=0.4, borderaxespad=0.2,
              labelspacing=0.35)
    ax.set_xlabel(r"median occurrence-block credibility $\lambda^{(o)}$")
    ax.set_ylabel("internal refinement gain (%)")
    ax.set_xlim(0.0, 1.03)
    fig.savefig(OUT / "fig10b_refinement_leverage_occ.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig10b_refinement_leverage_occ.png", bbox_inches="tight")
    plt.close(fig)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "t3_fig10_dual_panel",
        "commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "inputs": ["selection_surfaces.csv", "leverage_dual_lambda_corrected.csv",
                   "synthetic_sanity/grid.csv", "synthetic_sanity/extreme.csv"],
        "wall_clock_total_s": round(time.time() - t0, 1),
    }
    (OUT / "t3_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote fig10a/fig10b + refinement_leverage_corrected.csv + t3_run_meta.json")


if __name__ == "__main__":
    main()
