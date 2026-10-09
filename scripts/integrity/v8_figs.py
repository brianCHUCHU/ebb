"""v8 figures (2026-09-30), written to paper_v2/v8_paper_runs/figs/.

fig01_pooling_pays.{pdf,png}
    Main-text figure. (a) within-model gain from the shared prior against retained history
    (EBB vs EBB with the prior off, both B=0; outputs/2026-09-27/coldstart_nopool_wide.csv).
    (b) controlled surface with the five real panels (outputs/2026-09-13). Same as
    v7_fig01.py except that panel (a) no longer carries the dashed TweedieGP lines.
fig_app_prior_vs_tg.pdf
    Appendix. One small panel per dataset: the within-model prior gain (solid) next to EBB's
    advantage over TweedieGP (dashed), i.e. the lines removed from the main figure.
fig_app_deepar_credibility.pdf
    Appendix. EBB's walk-forward advantage over DeepAR against the median occurrence credibility
    of each panel at its audited configuration (outputs/deepar/deepar_results.csv,
    outputs/2026-09-09/wf_all_panels_wide.csv, outputs/2026-09-13/real_separation.csv).

Usage: py scripts/integrity/v8_figs.py
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
FIGS = ROOT / "paper_v2" / "v8_paper_runs" / "figs"
O = ROOT / "outputs"
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}
COLORS = {"online_retail": "#0072B2", "m5": "#009E73", "auto": "#D55E00", "carparts": "#CC79A7", "raf": "#E69F00"}
MARK = {"online_retail": "o", "m5": "s", "auto": "^", "carparts": "D", "raf": "v"}
INK, MUTED = "#222222", "#9a9a9a"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK})


def despine(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def fig_main() -> None:
    cells = pd.read_csv(O / "2026-09-13" / "sep_surface_v2_cells.csv")
    place = pd.read_csv(O / "2026-09-13" / "separation_placement.csv")
    nopool = pd.read_csv(O / "2026-09-27" / "coldstart_nopool_wide.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1, 1.18]})
    ax = axes[0]
    for p in PANELS:
        sub = nopool[nopool["panel"] == p].sort_values("median_len")
        ax.plot(sub["median_len"], sub["pool_gain_mean_pct"], marker=MARK[p], ms=4.2, lw=1.3,
                color=COLORS[p], label=PNL[p], markeredgecolor="white", markeredgewidth=0.6)
    # RAF at L=12: the prior-off arm is refit on twelve observations and the comparison is
    # unstable (text and tab:coldstart-nopool); drawn as an open marker
    r12 = nopool[(nopool["panel"] == "raf") & (nopool["L"].astype(str) == "12")]
    ax.scatter(r12["median_len"], r12["pool_gain_mean_pct"], marker=MARK["raf"], s=34, facecolor="white",
               edgecolor=COLORS["raf"], linewidth=1.1, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("history retained per series (periods, log)")
    ax.set_ylabel("gain from the shared prior (% SPL)")
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.5)
    ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=1.6, columnspacing=0.9)
    ax.set_title("(a) Pooling gains as credibility falls", fontsize=8, loc="left")

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
    off = {"online_retail": (7, 0, "left"), "carparts": (0, 10, "center"), "m5": (-7, -8, "right"),
           "auto": (-4, -28, "right"), "raf": (-7, 8, "right")}
    for _, r in place.iterrows():
        p = r["panel"]
        ax.scatter([r["lambda_size"]], [r["r2_learned"]], color=COLORS[p], marker=MARK[p], s=48,
                   zorder=5, edgecolor="white", linewidth=0.8)
        dx, dy, ha = off[p]
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
        despine(a)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(FIGS / "fig01_pooling_pays.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig01_pooling_pays.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig01_pooling_pays")


def fig_prior_vs_tg() -> None:
    nopool = pd.read_csv(O / "2026-09-27" / "coldstart_nopool_wide.csv")
    fig, axes = plt.subplots(1, 5, figsize=(6.9, 1.75), sharey=False)
    for ax, p in zip(axes, PANELS):
        sub = nopool[nopool["panel"] == p].sort_values("median_len")
        ax.plot(sub["median_len"], sub["pool_gain_mean_pct"], marker=MARK[p], ms=3.6, lw=1.2, color=COLORS[p],
                markeredgecolor="white", markeredgewidth=0.5, label="prior on vs. off")
        ax.plot(sub["median_len"], sub["adv_vs_TweedieGP"], marker=MARK[p], ms=3.0, lw=0.9, ls=(0, (3, 2)),
                color=INK, alpha=0.65, markerfacecolor="white", markeredgewidth=0.6, label="vs. TweedieGP")
        ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 3)))
        ax.set_xscale("log")
        ticks = sub["median_len"].to_numpy(float)
        if len(ticks) > 4:                       # keep the ends and every other level
            ticks = np.unique(np.r_[ticks[::2], ticks[-1]])
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(v)}" for v in ticks])
        ax.minorticks_off()
        ax.set_title(PNL[p], fontsize=7.5)
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.5)
        ax.tick_params(labelsize=6.5)
        despine(ax)
    axes[0].set_ylabel("gain (% SPL)", fontsize=7.5)
    axes[2].set_xlabel("history retained per series (periods, log scale)", fontsize=7.5)
    handles = [plt.Line2D([], [], color=INK, lw=1.2, marker="o", ms=3.6, markeredgecolor="white"),
               plt.Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), alpha=0.65, marker="o", ms=3.0,
                          markerfacecolor="white")]
    fig.legend(handles, ["shared prior on vs. off (within EBB)", "EBB vs. TweedieGP"], frameon=False,
               fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08), handlelength=2.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIGS / "fig_app_prior_vs_tg.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_app_prior_vs_tg")


def gap(deepar: float, ours: float) -> float:
    """Loss of the worse method relative to the better (the convention of the text and of
    tab:deepar), signed so that positive means EBB is the better."""
    return (deepar / ours - 1) * 100 if ours <= deepar else -(ours / deepar - 1) * 100


def fig_deepar() -> pd.DataFrame:
    res = pd.read_csv(O / "deepar" / "deepar_results.csv")
    res = res[res["protocol"] == "wf"].set_index("panel")
    wf = pd.read_csv(O / "2026-09-09" / "wf_all_panels_wide.csv").set_index(["panel", "model"])["mean"]
    lam = pd.read_csv(O / "2026-09-13" / "real_separation.csv").set_index("panel")["median_lambda_occ"]
    rows = []
    for p in PANELS:
        d = float(res.loc[p, "DeepAR"])
        e = float(wf.loc[(p, "EBB")])
        r = float(wf.loc[(p, "EBB-rule")])
        rows.append({"panel": p, "median_lambda_occ": float(lam.loc[p]), "DeepAR": d, "EBB": e, "EBB_rule": r,
                     "adv_EBB_pct": gap(d, e), "adv_rule_pct": gap(d, r)})
    t = pd.DataFrame(rows).sort_values("median_lambda_occ")
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(t["median_lambda_occ"], t["adv_EBB_pct"], color=MUTED, lw=0.8, zorder=1)
    for r in t.itertuples():
        ax.scatter([r.median_lambda_occ], [r.adv_EBB_pct], color=COLORS[r.panel], marker=MARK[r.panel], s=44,
                   edgecolor="white", linewidth=0.8, zorder=3)
        if abs(r.adv_rule_pct - r.adv_EBB_pct) > 0.2:
            ax.scatter([r.median_lambda_occ], [r.adv_rule_pct], facecolor="white", edgecolor=COLORS[r.panel],
                       marker=MARK[r.panel], s=36, linewidth=1.0, zorder=3)
        dx, ha = (7, "left") if r.median_lambda_occ < 0.8 else (-7, "right")
        ax.annotate(PNL[r.panel], (r.median_lambda_occ, r.adv_EBB_pct), xytext=(dx, 0),
                    textcoords="offset points", ha=ha, va="center", fontsize=6.8, color=COLORS[r.panel],
                    fontweight="bold")
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"median occurrence credibility $\lambda^{(o)}$")
    ax.set_ylabel("EBB advantage over DeepAR (%)")
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.5)
    despine(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_app_deepar_credibility.pdf", bbox_inches="tight")
    plt.close(fig)
    t.to_csv(O / "deepar" / "deepar_vs_credibility.csv", index=False)
    print(t.round(4).to_string(index=False))
    return t


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_main()
    fig_prior_vs_tg()
    fig_deepar()
