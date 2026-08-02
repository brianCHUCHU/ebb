"""Unified-style figure suite for the AISTATS draft (candidate set, no triage).

Writes every candidate figure to paper_v2/figs_v3/ as PDF (paper) + PNG
(preview). Existing paper_v2/figs/ is untouched. Style: serif/STIX, thin
recessive grid, no top/right spines, Okabe-Ito-derived categorical palette
validated with the dataviz six-checks script (order fixed, REMIX always
vermilion, Zero always black dashed, neutral gray reserved for reference
lines only).

Usage:
  py scripts/analysis/make_figures_v3.py            # everything
  py scripts/analysis/make_figures_v3.py fig04 fig06  # subset
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "aistats2027"
FIGS = ROOT / "paper_v2" / "figs_v3"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
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

# Validated categorical order (dataviz six checks, all PASS):
ACCENT = "#D55E00"  # REMIX, always
CAT = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7"]
NEUTRAL = "#8a8a8a"  # reference lines / de-emphasized series only
INK = "#222222"
ZERO_STYLE = dict(color="black", ls=(0, (4, 2)), lw=1.3)

DATASET_COLOR = {
    "Online Retail": CAT[0],
    "M5": CAT[1],
    "Auto": CAT[2],
    "Carparts": CAT[3],
    "RAF": CAT[4],
}


def save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png", flush=True)


def panel_label(ax, s: str) -> None:
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------- fig01
def fig01_overview() -> None:
    """Redrawn concept figure: (a) the model plane, (b) the pipeline."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.6),
                                 gridspec_kw={"width_ratios": [1.0, 1.25],
                                              "wspace": 0.18})

    # (a) the plane
    ax.grid(False)
    ax.set_xlim(-0.08, 1.10)
    ax.set_ylim(-0.12, 1.10)
    ax.set_xlabel("cross-series pooling (structure + EB shrinkage)")
    ax.set_ylabel(r"temporal forgetting $1-w$")
    ax.set_xticks([])
    ax.set_yticks([])
    # candidate grid: three structures x discount levels
    ys = np.array([0.0, 0.02, 0.05, 0.10, 0.18, 0.30, 0.55, 0.9])
    for xpos in (0.34, 0.61, 0.88):
        ax.scatter([xpos] * len(ys), ys, s=8, color=CAT[0], alpha=0.45, zorder=3)
    for xpos, lab in ((0.34, "global"), (0.61, "taxonomy"), (0.88, "mixture")):
        ax.annotate(lab, (xpos, 0.97), ha="center", fontsize=6.2, color=INK)
    # corners
    ax.scatter([0.02], [0.22], marker="s", s=34, color=INK, zorder=4)
    ax.annotate("classical TSB\n(no pooling, $w<1$)", (0.055, 0.235),
                fontsize=6.2, va="center", color=INK)
    ax.plot([0.26, 0.98], [0.0, 0.0], color=NEUTRAL, lw=2.6, alpha=0.55,
            solid_capstyle="round", zorder=2)
    ax.annotate("stationary hierarchical EB ($w{=}1$)", (0.62, -0.085),
                ha="center", fontsize=6.2, color=INK)
    # selected configuration
    ax.scatter([0.88], [0.02], marker="*", s=150, color=ACCENT, zorder=5,
               edgecolor="white", linewidth=0.5)
    ax.annotate("selected $(g,w)$", (0.88, 0.09), ha="center", fontsize=7,
                color=ACCENT, fontweight="bold")
    ax.annotate("", xy=(0.07, 0.185), xytext=(0.32, 0.035),
                arrowprops=dict(arrowstyle="->", color=NEUTRAL, lw=0.9))
    ax.annotate("Prop. 1: noninformative\ncorner reproduces TSB",
                (0.075, 0.56), fontsize=6.0, color=INK)
    panel_label(ax, "(a)")

    # (b) the pipeline
    bx.grid(False)
    bx.axis("off")
    steps = [
        "discounted\nstatistics\n$n_i(w),s_i(w)$",
        "candidate\npartitions\n(g / t / mix)",
        "joint scoring\n24 $(g,w)$ pairs\non val. block",
        "refit winner\nfull window\n$O(N)$",
        "ZILN\npredictive\nlaw",
    ]
    n = len(steps)
    w_box, gap = 0.155, (1 - 5 * 0.155) / 4
    for i, s in enumerate(steps):
        x0 = i * (w_box + gap)
        face = "#fdf0e7" if i in (2, 3) else "#f2f2f0"
        edge = ACCENT if i in (2, 3) else NEUTRAL
        bx.add_patch(FancyBboxPatch((x0, 0.30), w_box, 0.42,
                                    boxstyle="round,pad=0.010",
                                    facecolor=face, edgecolor=edge, lw=0.9,
                                    transform=bx.transAxes))
        bx.text(x0 + w_box / 2, 0.51, s, transform=bx.transAxes, fontsize=6.0,
                ha="center", va="center", color=INK)
        if i < n - 1:
            bx.add_patch(FancyArrowPatch((x0 + w_box + 0.004, 0.51),
                                         (x0 + w_box + gap - 0.004, 0.51),
                                         transform=bx.transAxes,
                                         arrowstyle="-|>", mutation_scale=8,
                                         color=INK, lw=0.9))
    bx.text(0.5, 0.13, "deterministic end to end; $w{=}1$ reproduces the "
                       "stationary model exactly", transform=bx.transAxes,
            fontsize=6.5, ha="center", color=INK, style="italic")
    panel_label(bx, "(b)")
    save(fig, "fig01_overview")


# ---------------------------------------------------------------- fig02
def _draw_drift(ax) -> None:
    b = np.linspace(1e-4, 0.5, 400)
    for i, delta in enumerate([0.0005, 0.002, 0.008]):
        bias2 = (delta * (1 - b) / b) ** 2
        var = 0.25 * b / (2 - b)
        mse = bias2 + var
        ax.plot(b, mse, color=CAT[i], lw=1.3, label=rf"$\delta={delta}$")
        bstar = b[np.argmin(mse)]
        ax.plot([bstar], [mse.min()], marker="*", ms=10, color=CAT[i],
                zorder=5, markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"forgetting strength $1-w$")
    ax.set_ylabel("worst-case MSE bound")
    ax.legend(handlelength=1.4, loc="upper center")


def _draw_discount_curves(ax) -> None:
    runs = {
        "Online Retail": "or_point_fixed", "M5": "m5_point_fixed",
        "Auto": "auto_point_fixed", "Carparts": "carparts_point_fixed",
        "RAF": "raf_point_fixed",
    }
    for name, d in runs.items():
        p = OUT / d / "tsbhb_variant_diagnostics.csv"
        if not p.exists():
            continue
        diag = pd.read_csv(p)
        row = diag[diag["variant"] == "TSB-HB-Discount"]
        if row.empty or pd.isna(row.iloc[0].get("discount_grid")):
            continue
        grid = ast.literal_eval(row.iloc[0]["discount_grid"])
        g = pd.DataFrame(grid).sort_values("discount")
        rel = g["scaled_pinball"] / g.loc[g["discount"] == 1.0, "scaled_pinball"].iloc[0]
        c = DATASET_COLOR[name]
        ax.plot(1.0 - g["discount"], rel, marker="o", ms=2.4, lw=1.1, color=c,
                label=name)
        w_star = g.loc[g["scaled_pinball"].idxmin(), "discount"]
        rel_star = g["scaled_pinball"].min() / g.loc[g["discount"] == 1.0, "scaled_pinball"].iloc[0]
        ax.plot([1.0 - w_star], [rel_star], marker="*", ms=10, color=c,
                zorder=5, markeredgecolor="white", markeredgewidth=0.4)
    ax.axhline(1.0, color=NEUTRAL, lw=0.7, ls=":")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(-2e-4, 0.13)
    ax.set_xlabel(r"forgetting strength $1-w$")
    ax.set_ylabel("validation SPL (rel. to $w{=}1$)")
    ax.legend(handlelength=1.3, loc="lower left")


def fig02_theory_selection() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    _draw_drift(axes[0])
    panel_label(axes[0], "(a)")
    _draw_discount_curves(axes[1])
    panel_label(axes[1], "(b)")
    save(fig, "fig02_theory_selection")
    # singles for layout flexibility
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    _draw_drift(ax)
    save(fig, "fig02a_drift_tradeoff")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    _draw_discount_curves(ax)
    save(fig, "fig02b_discount_curves")


# ---------------------------------------------------------------- fig03
_PROFILE_ORDER = [
    ("TSB-HB", "REMIX (ours)", ACCENT, "-", 1.9),
    ("AutoARIMA", "AutoARIMA", CAT[0], "--", 1.1),
    ("AutoTheta", "AutoTheta", CAT[1], "--", 1.1),
    ("CP-TSB", "CP-TSB", CAT[2], ":", 1.1),
    ("CP-ADIDA", "CP-ADIDA", CAT[3], ":", 1.1),
    ("CP-IMAPA", "CP-IMAPA", CAT[4], ":", 1.1),
    ("CP-CrostonSBA", "CP-SBA", NEUTRAL, ":", 1.0),
]


def _draw_profile(ax, spl: pd.DataFrame, order=None) -> None:
    order = order or _PROFILE_ORDER
    spl = spl[spl["quantile"].astype(str) != "mean"].copy()
    spl["q"] = spl["quantile"].astype(float)
    for model, label, color, ls, lw in order:
        sub = spl[spl["model"] == model].sort_values("q")
        if sub.empty:
            continue
        ax.plot(sub["q"], sub["scaled_pinball"], ls, color=color, lw=lw,
                marker="o", ms=2.4, label=label)
    ax.set_xlabel("quantile level $q$")
    ax.set_ylabel("scaled pinball loss")
    ax.set_xticks([0.1, 0.25, 0.5, 0.75, 0.9])


def fig03_spl_profiles() -> None:
    base = pd.read_csv(OUT / "or_prob_fixed_paper" / "prob_pinball_scaled.csv")
    remix = pd.read_csv(OUT / "or_prob_select" / "prob_pinball_scaled.csv")
    fixed = pd.concat(
        [base[base["model"] != "TSB-HB"], remix[remix["model"] == "TSB-HB"]],
        ignore_index=True)
    wf = pd.read_csv(OUT / "or_prob_wf_remix" / "prob_pinball_scaled.csv")
    wf_order = [
        ("TSB-HB", "REMIX (online)", ACCENT, "-", 1.9),
        ("CP-TSB", "CP-TSB", CAT[2], ":", 1.1),
        ("CP-ADIDA", "CP-ADIDA", CAT[3], ":", 1.1),
        ("CP-IMAPA", "CP-IMAPA", CAT[4], ":", 1.1),
        ("CP-CrostonSBA", "CP-SBA", NEUTRAL, ":", 1.0),
        ("CP-CrostonClassic", "CP-Croston", CAT[0], ":", 1.0),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
    _draw_profile(axes[0], fixed)
    axes[0].legend(ncol=2, handlelength=1.5, loc="upper left")
    axes[0].set_title("fixed origin")
    panel_label(axes[0], "(a)")
    _draw_profile(axes[1], wf, wf_order)
    axes[1].legend(ncol=2, handlelength=1.5, loc="upper left")
    axes[1].set_title("strict walk-forward (7-day blocks)")
    panel_label(axes[1], "(b)")
    save(fig, "fig03_spl_profiles")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    _draw_profile(ax, fixed)
    ax.legend(ncol=2, handlelength=1.5, loc="upper left")
    save(fig, "fig03a_spl_fixed")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    _draw_profile(ax, wf, wf_order)
    ax.legend(ncol=2, handlelength=1.5, loc="upper left")
    save(fig, "fig03b_spl_walkforward")


# ---------------------------------------------------------------- fig04
def fig04_fartail() -> None:
    zero = pd.read_csv(OUT / "zero_fartail.csv").set_index("dataset")
    qs = np.array([0.5, 0.75, 0.9, 0.95, 0.975, 0.99])

    def tx(q):  # logit stretch so the far tail is readable
        q = np.asarray(q, dtype=float)
        return np.log(q / (1.0 - q))

    series = [
        ("TSB-HB", "REMIX (ours)", ACCENT, "-", 1.9),
        ("AutoTheta", "AutoTheta", CAT[1], "--", 1.1),
        ("AutoARIMA", "AutoARIMA", CAT[0], "--", 1.1),
        ("CP-TSB", "CP-TSB", CAT[2], ":", 1.1),
        ("CP-IMAPA", "CP-IMAPA", CAT[4], ":", 1.1),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    handles = labels = None
    for ax, ds, title in ((axes[0], "carparts", "Carparts"),
                          (axes[1], "raf", "RAF")):
        df = pd.read_csv(OUT / f"fartail_{ds}_prob" / "prob_pinball_scaled.csv")
        df = df[df["quantile"].astype(str) != "mean"].copy()
        df["q"] = df["quantile"].astype(float)
        ax.axvspan(tx(0.925), tx(0.995), color="0.92", zorder=0)
        ax.text(tx(0.966), 0.965, "far tail", transform=ax.get_xaxis_transform(),
                fontsize=6.5, ha="center", va="top", color=INK)
        # Zero drawn first (under everything): on RAF it coincides with REMIX
        # at the central quantiles -- that overlap IS the saturation.
        zq = [zero.loc[ds, f"q{q}"] for q in qs]
        ax.plot(tx(qs), zq, label="Zero forecast", zorder=2, **ZERO_STYLE)
        for model, label, color, ls, lw in series:
            sub = df[df["model"] == model].sort_values("q")
            if sub.empty:
                continue
            z = 5 if model == "TSB-HB" else 3
            ax.plot(tx(sub["q"]), sub["scaled_pinball"], ls, color=color,
                    lw=lw, marker="o", ms=2.4, label=label, zorder=z)
        ax.set_xlabel("quantile level $q$ (logit scale)")
        ax.set_title(title)
        ax.set_xticks(tx([0.5, 0.75, 0.9, 0.95, 0.975, 0.99]))
        ax.set_xticklabels(["0.5", "0.75", "0.9", "0.95", "0.975", "0.99"],
                           fontsize=7)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("scaled pinball loss")
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.06), handlelength=1.6, fontsize=7)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    save(fig, "fig04_fartail")


# ---------------------------------------------------------------- fig05
def fig05_resolution() -> None:
    from scipy.stats import norm
    grid = pd.read_csv(ROOT / "outputs" / "synthetic_sanity" / "grid.csv")
    cal = pd.read_csv(ROOT / "outputs" / "synthetic_resolution" / "collapse_calibration.csv")
    ext = pd.read_csv(ROOT / "outputs" / "synthetic_sanity" / "extreme.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35),
                             gridspec_kw={"wspace": 0.42})

    ax = axes[0]
    w = grid.pivot_table(index=["separation", "T", "discount", "rep"],
                         columns="arm", values="nll")
    gain = 100.0 * (w["global-estimated"] - w["oracle-labels+estimated-hypers"]) / w["global-estimated"]
    for (T, marker, color) in ((30, "o", CAT[0]), (120, "s", CAT[2])):
        sub = gain.xs(T, level="T").groupby(level="discount")
        mean, se = sub.mean(), sub.std(ddof=1) / np.sqrt(sub.count())
        x = 1.0 - mean.index.to_numpy(dtype=float)
        order = np.argsort(x)
        ax.errorbar(x[order], mean.to_numpy()[order], yerr=se.to_numpy()[order],
                    marker=marker, ms=3.2, capsize=2.5, lw=1.1, color=color,
                    label=f"$T={T}$")
    ax.axhline(0.0, color=NEUTRAL, lw=0.7, ls="--")
    ax.set_xlabel(r"forgetting strength $1-w$")
    ax.set_ylabel("NLL gain of true partition (%)")
    ax.legend()
    panel_label(ax, "(a)")

    ax = axes[1]
    sub = cal.sort_values("z")
    ax.scatter(sub["z"], sub["p_empirical"], s=6, alpha=0.35, color=CAT[3],
               label="simulated", edgecolor="none")
    zz = np.linspace(0.0, 4.0, 200)
    ax.plot(zz, norm.cdf(-zz), color=INK, lw=1.4, label=r"$\Phi(-z)$")
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"$z=\rho_g\sqrt{(m_g-1)/2}$")
    ax.set_ylabel(r"$\Pr[\hat\tau_g^2=0]$")
    ax.legend()
    panel_label(ax, "(b)")

    ax = axes[2]
    styles = {"nll": ("NLL", "o-", CAT[0]), "pinball": ("pinball", "s-", CAT[1]),
              "brier_occurrence": ("Brier", "^-", CAT[2]),
              "log_size_mse": ("log-size MSE", "d-", CAT[4])}
    for m, (label, style, color) in styles.items():
        p = ext.pivot_table(index="median_n_pos", columns="arm", values=m)
        g = 100.0 * (p["global-estimated"] - p["oracle-labels+estimated-hypers"]) / p["global-estimated"]
        ax.plot(g.index.to_numpy(dtype=float), g.to_numpy(), style, color=color,
                label=label, ms=3.0, lw=1.1)
    ax.set_xscale("log")
    ax.axhline(0.0, color=NEUTRAL, lw=0.7, ls="--")
    ax.set_xlabel(r"median positive obs. per item $n_i^{+}$")
    ax.set_ylabel("gain of true partition (%)")
    ax.legend()
    panel_label(ax, "(c)")

    save(fig, "fig05_resolution")


# ---------------------------------------------------------------- fig06
def fig06_leverage() -> None:
    pts = [  # (panel, median lambda at w*, learned-vs-single-pool MAE %, note)
        ("Carparts", 0.224, +2.7, ""),
        ("M5", 0.712, -2.3, ""),
        ("Online Retail", 0.749, -2.9, ""),
        ("RAF", 0.917, -5.0, " (reg.)"),
        ("Auto", 0.942, -3.6, ""),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axvspan(0.0, 0.5, color="#fdf0e7", zorder=0)
    ax.text(0.03, -5.9, "median item\nprior-dominated", fontsize=6.5,
            ha="left", va="bottom", color=INK)
    ax.axhline(0.0, color=NEUTRAL, lw=0.8)
    ax.axvline(0.5, color=NEUTRAL, lw=0.7, ls=":")
    offsets = {"Carparts": (7, -3), "M5": (-8, 7), "Online Retail": (-52, -13),
               "RAF": (-42, -3), "Auto": (-10, 8)}
    for name, lam, eff, note in pts:
        color = ACCENT if eff > 0 else CAT[0]
        ax.scatter([lam], [eff], s=42, color=color, zorder=5,
                   edgecolor="white", linewidth=0.6)
        ax.annotate(f"{name}{note}", (lam, eff), xytext=offsets[name],
                    textcoords="offset points", fontsize=6.8, color=INK)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-6.8, 4.6)
    ax.set_xlabel(r"median credibility $\lambda_i$ at selected $w^{*}$")
    ax.set_ylabel("learned partition vs. single pool\n(MAE gain, %)")
    save(fig, "fig06_leverage")


# ---------------------------------------------------------------- fig07
def fig07_horizon() -> None:
    base = pd.read_csv(OUT / "or_point_fixed" / "point_predictions.csv.gz")
    remix = pd.read_csv(OUT / "or_point_remix" / "point_predictions.csv.gz")
    remix = remix[["unique_id", "ds", "TSB-HB"]].rename(columns={"TSB-HB": "REMIX"})
    joint = base.merge(remix, on=["unique_id", "ds"], how="inner")
    joint = joint.sort_values(["unique_id", "ds"]).copy()
    joint["h"] = joint.groupby("unique_id").cumcount() + 1
    joint["bucket"] = pd.cut(joint["h"], bins=[0, 7, 14, 28, 56, 112, 10 ** 9],
                             labels=["1-7", "8-14", "15-28", "29-56", "57-112", "113+"])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    show = [("REMIX", "REMIX (ours)", ACCENT, "-", 1.9),
            ("TSB", "TSB", CAT[0], "--", 1.1),
            ("ADIDA", "ADIDA", CAT[2], "--", 1.1),
            ("IMAPA", "IMAPA", CAT[4], ":", 1.1),
            ("AutoARIMA", "AutoARIMA", CAT[1], ":", 1.1)]
    for colname, label, color, ls, lw in show:
        g = joint.groupby("bucket", observed=True).apply(
            lambda d: float(np.sqrt(((d[colname] - d.y) ** 2).mean())),
            include_groups=False)
        ax.plot(range(len(g)), g.values, ls, marker="o", ms=2.4, lw=lw,
                color=color, label=label)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["1-7", "8-14", "15-28", "29-56", "57-112", "113+"],
                       fontsize=7)
    ax.set_xlabel("forecast horizon bucket (days ahead)")
    ax.set_ylabel("RMSE")
    ax.legend(ncol=2, handlelength=1.5)
    save(fig, "fig07_horizon")


# ---------------------------------------------------------------- fig08
def fig08_cd() -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    import horizon_and_cd as hcd

    rank_blocks = []
    models = hcd.MODELS + ["REMIX"]
    n_datasets = 0
    for ds in hcd.PRED_SOURCES:
        joint = hcd.load_joint(ds)
        if joint is None:
            print(f"{ds}: predictions missing, skipped")
            continue
        denom = hcd.load_init_naive_mse(ds)
        if denom.empty:
            continue
        ps = hcd.per_series_scaled_sqerr(
            joint, [m for m in models if m in joint.columns], denom)
        rank_blocks.append(ps.rank(axis=1))
        n_datasets += 1
        print(f"{ds}: ranked over {len(ps)} series", flush=True)
    if not rank_blocks:
        return
    from scipy import stats
    ranks_all = pd.concat(rank_blocks, ignore_index=True).dropna()
    avg_rank = ranks_all.mean(axis=0).sort_values()
    k = ranks_all.shape[1]
    n_blocks = len(ranks_all)
    _, fried_p = stats.friedmanchisquare(
        *[ranks_all[c].values for c in ranks_all.columns])
    q_alpha = {8: 3.031}[k]
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks))

    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", visible=False)
    y = 0
    for m, r in avg_rank.items():
        color = ACCENT if m == "REMIX" else INK
        ax.plot([r], [y], "o", ms=4.5, color=color)
        ax.annotate(f" {m} ({r:.2f})", (r, y), fontsize=7, va="center",
                    color=color)
        y -= 1
    ax.errorbar([avg_rank.min()], [y], xerr=[[0], [cd]], fmt="none",
                capsize=3, color=INK, lw=0.9)
    ax.annotate(f" CD={cd:.2f} (Nemenyi, $\\alpha$=0.05)",
                (avg_rank.min(), y), fontsize=7, va="center")
    ax.set_yticks([])
    p_txt = f"p={fried_p:.2g}" if fried_p > 0 else "p<$10^{-300}$"
    ax.set_xlabel(
        f"avg. rank of per-series scaled sq. error\n"
        f"({n_blocks} series, {n_datasets} datasets; Friedman {p_txt}; "
        f"descriptive)", fontsize=7)
    save(fig, "fig08_cd")


# ---------------------------------------------------------------- fig09
def fig09_reselection() -> None:
    tr = pd.read_csv(OUT / "wf_reselection_trajectory.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    e1 = tr[tr["arm"] == "every1"].sort_values("block")
    ax.step(e1["block"], e1["w"], where="post", color=CAT[0], lw=1.4,
            label="re-selected every block")
    e4 = tr[tr["arm"] == "every4"].sort_values("block")
    ax.plot(e4["block"], e4["w"], "s", ms=4, color=CAT[1], zorder=5,
            label="re-selected every 4 blocks")
    ax.axhline(0.98, color=ACCENT, lw=1.2, ls=(0, (4, 2)),
               label="origin selection ($w{=}0.98$)")
    non_mix = e1[e1["structure"] != "mixture"]
    for _, row in non_mix.iterrows():
        ax.annotate("global", (row["block"], row["w"]), xytext=(0, -11),
                    textcoords="offset points", fontsize=6, ha="center",
                    color=INK)
    ax.set_yticks([0.90, 0.95, 0.98])
    ax.set_ylim(0.885, 0.995)
    ax.set_xlabel("walk-forward block (7 days each)")
    ax.set_ylabel("selected discount $w$")
    ax.legend(loc="lower left", handlelength=1.6)
    save(fig, "fig09_reselection")


# ---------------------------------------------------------------- main
ALL = {
    "fig01": fig01_overview,
    "fig02": fig02_theory_selection,
    "fig03": fig03_spl_profiles,
    "fig04": fig04_fartail,
    "fig05": fig05_resolution,
    "fig06": fig06_leverage,
    "fig07": fig07_horizon,
    "fig08": fig08_cd,
    "fig09": fig09_reselection,
}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(ALL)
    for key in todo:
        ALL[key]()
