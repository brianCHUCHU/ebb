"""DEPRECATED (2026-08-06 integrity audit): this script reads the legacy
2026-07-19 selection surfaces, whose taxonomy/mixture labels were learned on
the full internal selection window (leakage). Its outputs are retained only
with the DEPRECATED_leaky_labels suffix. The corrected pipeline lives in
scripts/integrity/ and writes to outputs/integrity_<date>/. Do not rerun.

Internal refinement gain vs credibility leverage (fig + surfaces for v4 TODOs).

Three artifacts for the v4 draft's leverage section:

1. outputs/aistats2027/selection_surfaces.csv -- the full 8-discount x
   3-structure internal-validation SPL surface for all five panels. The four
   non-M5 panels come from the runs behind the paper's external numbers
   (outputs/aistats2027/{or,auto,carparts,raf}_prob_select); M5 comes from the
   corrected-split audit (outputs/aistats2027_rebuild/m5_prob_correct_split,
   seed 42, init 2/3) because the legacy m5_prob_select surface used the
   1/3-init protocol and is obsolete.

2. outputs/aistats2027/refinement_leverage.csv -- the figure's points in long
   format. Internal refinement gain at discount w:
       100 * (R(global, w) - min(R(taxonomy, w), R(mixture, w))) / R(global, w)
   Real panels contribute two points each (w=1 and the selected w*), with the
   median size-block credibility from leverage_dual_lambda.csv at the same
   discounts. Synthetic cells contribute the paired true-partition pinball
   gain (oracle labels + estimated hypers vs single pool) with the same
   diagnostic computed on the fitted single pool; the grid pairs 9 replicates
   per (T, w) cell, the length sweep is one deterministic run per T.

3. paper_v2/figs_v3/fig10_refinement_leverage.{pdf,png} -- the scatter that
   places real panels and simulated cells on the same leverage axis, drawing
   each real panel and each simulated family as a trajectory of increasing
   forgetting (or shrinking history) so the two-sided mechanism is visible:
   forgetting moves a panel left (more leverage) while gains only survive
   where the discounted data still resolve the structure.

Usage: py scripts/analysis/refinement_leverage.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "aistats2027"
REBUILD = ROOT / "outputs" / "aistats2027_rebuild"
SANITY = ROOT / "outputs" / "synthetic_sanity"
FIGS = ROOT / "paper_v2" / "figs_v3"

SURFACES = {
    "online_retail": OUT / "or_prob_select" / "remix_selection_diagnostics.csv",
    "m5": REBUILD / "m5_prob_correct_split" / "remix_selection_diagnostics.csv",
    "auto": OUT / "auto_prob_select" / "remix_selection_diagnostics.csv",
    "carparts": OUT / "carparts_prob_select" / "remix_selection_diagnostics.csv",
    "raf": OUT / "raf_prob_select" / "remix_selection_diagnostics.csv",
}
PANEL_LABEL = {
    "online_retail": "Online Retail", "m5": "M5", "auto": "Auto",
    "carparts": "Carparts", "raf": "RAF",
}

# Style block copied from make_figures_v3.py so the figure joins the suite.
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
DATASET_COLOR = {
    "Online Retail": CAT[0], "M5": CAT[1], "Auto": CAT[2],
    "Carparts": CAT[3], "RAF": CAT[4],
}


def refine_gain(surf: pd.Series, w) -> float:
    g = surf[("global", w)]
    return float(100.0 * (g - min(surf[("taxonomy", w)], surf[("mixture", w)])) / g)


def load_surfaces() -> pd.DataFrame:
    frames = []
    for panel, path in SURFACES.items():
        df = pd.read_csv(path)
        df["panel"] = panel
        frames.append(df[["panel", "structure", "discount", "scaled_pinball"]])
    return pd.concat(frames, ignore_index=True)


def real_points(surfaces: pd.DataFrame) -> pd.DataFrame:
    lev = pd.read_csv(OUT / "leverage_dual_lambda.csv").set_index(["panel", "w"])
    rows = []
    for panel, df in surfaces.groupby("panel"):
        surf = df.set_index(["structure", "discount"])["scaled_pinball"]
        s_star, w_star = surf.idxmin()
        for w, stage in ((1.0, "w=1"), (w_star, "w=w*")):
            rec = lev.loc[(panel, w)]
            rows.append({
                "kind": "panel",
                "label": PANEL_LABEL[panel],
                "stage": stage,
                "w": w,
                "argmin_structure": s_star,
                "median_lambda_size": float(rec["median_lambda_size"]),
                "median_lambda_occ": float(rec["median_lambda_occ"]),
                "refinement_gain_pct": refine_gain(surf, w),
            })
    return pd.DataFrame(rows)


def synthetic_points() -> pd.DataFrame:
    rows = []
    grid = pd.read_csv(SANITY / "grid.csv")
    keys = ["separation", "discount", "T", "rep"]
    piv = grid.pivot_table(index=keys, columns="arm", values="pinball")
    lam = grid[grid["arm"] == "global-estimated"].set_index(keys)["median_lambda_size"]
    gain = 100.0 * (piv["global-estimated"] - piv["oracle-labels+estimated-hypers"]) / piv["global-estimated"]
    cell = pd.DataFrame({"gain": gain, "lam": lam}).reset_index()
    for (T, w), c in cell.groupby(["T", "discount"]):
        rows.append({
            "kind": f"sim-grid-T{T}",
            "label": f"T={T}, w={w:g}",
            "stage": f"w={w:g}",
            "w": w,
            "argmin_structure": "true-partition",
            "median_lambda_size": float(c["lam"].median()),
            "median_lambda_occ": np.nan,
            "refinement_gain_pct": float(c["gain"].mean()),
        })
    extreme = pd.read_csv(SANITY / "extreme.csv")
    ep = extreme.pivot_table(index="T", columns="arm", values="pinball")
    el = extreme[extreme["arm"] == "global-estimated"].set_index("T")["median_lambda_size"]
    for T in ep.index:
        rows.append({
            "kind": "sim-length",
            "label": f"T={T}",
            "stage": f"T={T}",
            "w": 1.0,
            "argmin_structure": "true-partition",
            "median_lambda_size": float(el.loc[T]),
            "median_lambda_occ": np.nan,
            "refinement_gain_pct": float(
                100.0 * (ep.loc[T, "global-estimated"] - ep.loc[T, "oracle-labels+estimated-hypers"])
                / ep.loc[T, "global-estimated"]
            ),
        })
    return pd.DataFrame(rows)


def make_figure(points: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axhline(0.0, color=NEUTRAL, lw=0.7, zorder=1)

    # Simulated families: forgetting trajectories at T=30 and T=120, and the
    # shrinking-history sweep at w=1. All in recessive gray.
    for kind, marker, ls in (("sim-grid-T30", "D", (0, (2, 2))),
                             ("sim-grid-T120", "^", (0, (4, 2))),
                             ("sim-length", "s", (0, (1, 1.5)))):
        s = points[points["kind"] == kind].sort_values("median_lambda_size")
        ax.plot(s["median_lambda_size"], s["refinement_gain_pct"],
                color=NEUTRAL, lw=0.8, ls=ls, zorder=2)
        ax.scatter(s["median_lambda_size"], s["refinement_gain_pct"],
                   marker=marker, s=14, facecolors="none", edgecolors=NEUTRAL,
                   linewidths=0.9, zorder=2)

    # Real panels: hollow at w=1, filled at the selected discount, connected.
    pan = points[points["kind"] == "panel"]
    for label, sub in pan.groupby("label"):
        sub = sub.set_index("stage")
        c = DATASET_COLOR[label]
        x0, y0 = sub.loc["w=1", ["median_lambda_size", "refinement_gain_pct"]]
        x1, y1 = sub.loc["w=w*", ["median_lambda_size", "refinement_gain_pct"]]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.9, alpha=0.65),
                    zorder=3)
        ax.scatter([x0], [y0], s=26, facecolors="white", edgecolors=c,
                   linewidths=1.1, zorder=4)
        ax.scatter([x1], [y1], s=30, color=c, zorder=4)

    # Occurrence-block position for the two monthly panels whose low-leverage
    # block is occurrence, not size (the "opposite regimes" paragraph).
    for label in ("Auto", "RAF"):
        sub = pan[pan["label"] == label].set_index("stage")
        c = DATASET_COLOR[label]
        x1, y1 = sub.loc["w=w*", ["median_lambda_size", "refinement_gain_pct"]]
        xo = sub.loc["w=w*", "median_lambda_occ"]
        ax.plot([x1, xo], [y1, y1], color=c, lw=0.7, ls=(0, (1, 1.5)),
                alpha=0.8, zorder=3)
        ax.scatter([xo], [y1], marker="o", s=26, facecolors="white",
                   edgecolors=c, linewidths=1.1, zorder=4)
        ax.annotate(r"$\lambda^{(o)}$", (xo, y1), textcoords="offset points",
                    xytext=(2, 6), fontsize=6.5, color=c, ha="left")

    offsets = {
        "Online Retail": (-7, 5, "right"), "M5": (5, -2, "left"),
        "Auto": (4, 4, "left"), "Carparts": (-2, 6, "left"),
        "RAF": (1, -11, "center"),
    }
    for label, sub in pan.groupby("label"):
        sub = sub.set_index("stage")
        dx, dy, ha = offsets[label]
        ax.annotate(label,
                    (sub.loc["w=w*", "median_lambda_size"],
                     sub.loc["w=w*", "refinement_gain_pct"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.0,
                    color=DATASET_COLOR[label], ha=ha)

    # Legend proxies.
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([], [], marker="o", color=NEUTRAL, mfc="white", ls="none",
               ms=5, label="panel, $w{=}1$"),
        Line2D([], [], marker="o", color=NEUTRAL, ls="none", ms=5.5,
               label="panel, selected $w^{*}$"),
        Line2D([], [], marker="D", color=NEUTRAL, mfc="none", ls=(0, (2, 2)),
               ms=3.5, lw=0.8, label="simulated, $T{=}30$"),
        Line2D([], [], marker="^", color=NEUTRAL, mfc="none", ls=(0, (4, 2)),
               ms=3.5, lw=0.8, label="simulated, $T{=}120$"),
        Line2D([], [], marker="s", color=NEUTRAL, mfc="none", ls=(0, (1, 1.5)),
               ms=3.5, lw=0.8, label="simulated, length sweep"),
    ]
    ax.legend(handles=proxies, loc="upper left", handletextpad=0.4,
              borderaxespad=0.2, labelspacing=0.35)

    ax.set_xlabel(r"median size-block credibility $\lambda^{(+)}$ at the fitting discount")
    ax.set_ylabel("internal refinement gain (%)")
    ax.set_xlim(0.05, 1.03)
    fig.savefig(FIGS / "fig10_refinement_leverage.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig10_refinement_leverage.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    surfaces = load_surfaces()
    surfaces.to_csv(OUT / "selection_surfaces.csv", index=False)

    points = pd.concat([real_points(surfaces), synthetic_points()], ignore_index=True)
    points.to_csv(OUT / "refinement_leverage.csv", index=False)
    print(points.to_string(index=False))

    make_figure(points)
    print("\nwrote selection_surfaces.csv, refinement_leverage.csv, "
          "fig10_refinement_leverage.{pdf,png}")


if __name__ == "__main__":
    main()
