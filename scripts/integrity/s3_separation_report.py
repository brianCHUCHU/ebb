"""S3: place the five real panels on the (leverage, separation) surface and
report the implied room, per docs/DESIGN_separation.md.

Inputs: outputs/<date>/sep_surface_cells.csv (S1), real_separation.csv (S2),
        outputs/integrity_2026-08-06/refinement_leverage_corrected.csv (realized
        internal refinement gain at w*, the fig10 points).
Outputs: outputs/<date>/separation_placement.csv, tab_separation.tex,
         paper_v2/figs_v3/fig14_separation_surface.{pdf,png} (+ copies to figs/)
Usage: py scripts/integrity/s3_separation_report.py [YYYY-MM-DD]
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
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
FIGS = ROOT / "paper_v2" / "figs_v3"
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}
INK, MUTED, ACCENT = "#222222", "#9a9a9a", "#D55E00"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def main():
    cells_name = sys.argv[2] if len(sys.argv) > 2 else "sep_surface_cells.csv"   # e.g. sep_surface_v2_cells.csv
    cells = pd.read_csv(OUT / cells_name)
    print("surface:", cells_name)
    real = pd.read_csv(OUT / "real_separation.csv").set_index("panel")
    fig10 = pd.read_csv(ROOT / "outputs" / "integrity_2026-08-06" / "refinement_leverage_corrected.csv")
    realized = fig10[(fig10["kind"] == "panel") & (fig10["stage"] == "w=w*")].set_index("label")["refinement_gain_pct"]

    pts = cells[["lambda_size", "r2_true"]].to_numpy(float)
    z_true = cells["gain_true"].to_numpy(float)
    z_learn = cells["gain_learned"].to_numpy(float)

    def room(lam, r2, z):
        v = griddata(pts, z, (lam, r2), method="linear")
        if np.isnan(v):
            v = griddata(pts, z, (lam, r2), method="nearest")
        d = np.hypot(pts[:, 0] - lam, pts[:, 1] - r2)
        i = int(np.argmin(d))
        return float(v), i

    # Room read within the panel's own forgetting regime: fix w at the nearest simulated
    # discount, then interpolate gain along lambda (monotone in separation within a (w,T)
    # family) at the panel's lambda, separately for T=30 and T=120. The family's R2 at that
    # lambda is reported next to the panel's own R2 so the separation match is visible.
    W_SLICE = {"online_retail": 0.95, "m5": 0.95, "auto": 1.0, "carparts": 0.9, "raf": 1.0}

    def along_lambda(fam, lam, col):
        f = fam.sort_values("lambda_size")
        x, y = f["lambda_size"].to_numpy(float), f[col].to_numpy(float)
        return float(np.interp(lam, x, y, left=y[0], right=y[-1]))

    rows = []
    for p in PNL:
        r = real.loc[p]
        lam = float(r["median_lambda_size"]); w = W_SLICE[p]
        rec = {"panel": p, "w_slice": w, "lambda_size": lam, "r2_learned": float(r["r2_learned"]),
               "r2_taxonomy": float(r["r2_taxonomy"]), "k_learned": int(r["k_learned"]),
               "realized_gain_pct": float(realized.get(PNL[p], np.nan))}
        for T in (30, 120):
            fam = cells[(cells["discount"] == w) & (cells["T"] == T)]
            rec[f"room_T{T}"] = along_lambda(fam, lam, "gain_true")
            rec[f"sim_learned_T{T}"] = along_lambda(fam, lam, "gain_learned")
            rec[f"family_r2_T{T}"] = along_lambda(fam, lam, "r2_true")
        rec["room_lo"] = min(rec["room_T30"], rec["room_T120"]); rec["room_hi"] = max(rec["room_T30"], rec["room_T120"])
        # secondary: 2-D interpolation over all cells at (lambda, R2)
        for tag, r2 in (("learned", rec["r2_learned"]), ("taxonomy", rec["r2_taxonomy"])):
            v, i = room(lam, r2, z_true)
            rec[f"room2d_{tag}"] = v
        rows.append(rec)
    place = pd.DataFrame(rows)
    place.round(4).to_csv(OUT / "separation_placement.csv", index=False)
    pd.set_option("display.width", 250)
    print(place.round(3).to_string(index=False))

    # LaTeX rows: Panel & w & lambda & R2 learned & R2 tax & family R2 (T=120) & room T=30 & room T=120 & sim learned (T=120) & realized
    lines = []
    for _, r in place.iterrows():
        lines.append(f"{PNL[r['panel']]} & {r['w_slice']:g} & {r['lambda_size']:.3f} & {r['r2_learned']:.2f} & {r['r2_taxonomy']:.2f} & "
                     f"{r['family_r2_T120']:.2f} & {r['room_T30']:+.2f} & {r['room_T120']:+.2f} & "
                     f"{r['sim_learned_T120']:+.2f} & {r['realized_gain_pct']:+.2f}\\\\")
    (OUT / "tab_separation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # surface summary rows (per separation, averaged over w,T) for the appendix
    aggs = dict(r2_true=("r2_true", "mean"), gain_true=("gain_true", "mean"),
                gain_learned=("gain_learned", "mean"), k=("k_learned", "median"))
    has_fix = "gain_learned_split_cred" in cells.columns
    if has_fix:
        aggs["gain_fix"] = ("gain_learned_split_cred", "mean")
    s = cells.groupby("separation").agg(**aggs).reset_index()
    lines_sim = []
    for r in s.itertuples():
        fix = f" & {r.gain_fix:+.2f}" if has_fix else ""
        lines_sim.append(f"{r.separation:g} & {r.r2_true:.2f} & {r.gain_true:+.2f} & {r.gain_learned:+.2f}{fix} & {r.k:.0f}\\\\")
    (OUT / "tab_separation_sim.tex").write_text("\n".join(lines_sim) + "\n", encoding="utf-8")
    print(s.round(3).to_string(index=False))

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    vmax = float(np.nanmax(np.abs(z_true)))
    sc = ax.scatter(cells["lambda_size"], cells["r2_true"], c=z_true, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=60, marker="s", edgecolor="white", linewidth=0.5, zorder=2)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("true-partition gain (% SPL)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    OFF = {"online_retail": (7, 0, "left"), "carparts": (7, 0, "left"), "m5": (-8, -9, "right"),
           "auto": (7, -2, "left"), "raf": (-8, 8, "right")}
    for _, r in place.iterrows():
        ax.plot([r["lambda_size"], r["lambda_size"]], [r["r2_taxonomy"], r["r2_learned"]], color=ACCENT, lw=0.9, zorder=3)
        ax.scatter([r["lambda_size"]], [r["r2_learned"]], color=ACCENT, s=42, zorder=4, edgecolor="white", linewidth=0.8)
        ax.scatter([r["lambda_size"]], [r["r2_taxonomy"]], facecolor="white", edgecolor=ACCENT, s=42, zorder=4, linewidth=1.2)
        dx, dy, ha = OFF[r["panel"]]
        ax.annotate(f"{PNL[r['panel']]} ({r['realized_gain_pct']:+.2f}%)", (r["lambda_size"], r["r2_learned"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.6, color=ACCENT, fontweight="bold")
    ax.set_xlabel(r"median size-block credibility $\lambda^{(+)}$")
    ax.set_ylabel(r"size-block separation $R^2$")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.06)
    ax.set_title("Squares: simulated cells. Circles: real panels\n(filled: learned partition, open: taxonomy)", fontsize=7.4, loc="left")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax = axes[1]
    xs = {0.0: 0, 0.25: 1, 0.5: 2, 1.0: 3, 2.0: 4, 3.0: 5}
    agg = cells.groupby("separation").agg(gt=("gain_true", "mean"), gl=("gain_learned", "mean"),
                                          gt_lo=("gain_true", "min"), gt_hi=("gain_true", "max"),
                                          gl_lo=("gain_learned", "min"), gl_hi=("gain_learned", "max")).reset_index()
    x = agg["separation"].map(xs)
    ax.fill_between(x, agg["gt_lo"], agg["gt_hi"], color=MUTED, alpha=0.18, lw=0)
    ax.fill_between(x, agg["gl_lo"], agg["gl_hi"], color=ACCENT, alpha=0.15, lw=0)
    ax.plot(x, agg["gt"], color=MUTED, lw=1.4, marker="o", ms=3.5, label="true partition (mean over w, T)")
    ax.plot(x, agg["gl"], color=ACCENT, lw=1.4, marker="s", ms=3.5, ls=(0, (3, 2)), label="learned mixture (mean over w, T)")
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.set_xticks(list(xs.values())); ax.set_xticklabels([f"{k:g}" for k in xs])
    ax.set_xlabel("group-center separation (log-size units)")
    ax.set_ylabel("gain over single pool (% SPL)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Bands: range over the six (w, T) cells", fontsize=7.6, loc="left")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig14_separation_surface.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig14_separation_surface.png", bbox_inches="tight", dpi=200)
    for d in ("v5", "v5_paper"):
        shutil.copy(FIGS / "fig14_separation_surface.pdf", ROOT / "paper_v2" / d / "figs" / "fig14_separation_surface.pdf")
    print("wrote separation_placement.csv, tab_separation.tex, tab_separation_sim.tex, fig14_separation_surface")


if __name__ == "__main__":
    main()
