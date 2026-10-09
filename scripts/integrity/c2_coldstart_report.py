"""C2: assemble the cold-start scaling results (C1) into one table, LaTeX rows,
and a figure: EBB's relative advantage over each per-series comparator as a
function of the median size-block credibility weight lambda^(+) (x), one line
per panel, plus the truncation length as point labels.

Outputs (outputs/<date>/): coldstart_all.csv, tab_coldstart.tex,
  paper_v2/figs_v3/fig13_coldstart.{pdf,png} (+ copies into both figs/ dirs)
Usage: py scripts/integrity/c2_coldstart_report.py [YYYY-MM-DD]
"""
from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
FIGS = ROOT / "paper_v2" / "figs_v3"
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}
COLORS = {"online_retail": "#0072B2", "m5": "#009E73", "auto": "#D55E00", "carparts": "#CC79A7", "raf": "#E69F00"}
MARK = {"online_retail": "o", "m5": "s", "auto": "^", "carparts": "D", "raf": "v"}
INK, MUTED = "#222222", "#9a9a9a"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def main():
    frames = []
    for p in PANELS:
        for d in sorted((ROOT / "outputs").glob("2026-09-*"), reverse=True):
            f = d / f"coldstart_{p}.csv"
            if f.exists():
                frames.append(pd.read_csv(f)); break
    df = pd.concat(frames, ignore_index=True)
    df["L"] = df["L"].astype(str)
    df.to_csv(OUT / "coldstart_all.csv", index=False)
    wide = df.pivot_table(index=["panel", "L", "median_len", "median_lambda_occ", "median_lambda_size"],
                          columns="model", values="spl_mean").reset_index()
    for c in ("TweedieGP", "CP-ADIDA", "CP-Croston"):
        if c in wide.columns:
            wide[f"adv_vs_{c}"] = 100 * (wide[c] - wide["EBB"]) / wide[c]
    wide["Lnum"] = wide["L"].map(lambda s: 10**9 if s == "full" else int(s))
    wide = wide.sort_values(["panel", "Lnum"])
    wide.round(4).to_csv(OUT / "coldstart_wide.csv", index=False)
    print(wide.round(4).to_string())
    # LaTeX rows: Panel & L & median len & lambda_o & lambda_+ & EBB & TweedieGP & CP-ADIDA & adv vs TG
    lines = []
    for p in PANELS:
        sub = wide[wide["panel"] == p]
        if sub.empty:
            continue
        for i, (_, r) in enumerate(sub.iterrows()):
            name = PNL[p] if i == 0 else ""
            tg = "---" if pd.isna(r.get("TweedieGP", float("nan"))) else f"{r['TweedieGP']:.4f}"
            cp = "---" if pd.isna(r.get("CP-ADIDA", float("nan"))) else f"{r['CP-ADIDA']:.4f}"
            adv = r.get("adv_vs_TweedieGP", float("nan"))
            advs = "---" if pd.isna(adv) else f"{adv:+.1f}"
            lines.append(f"{name} & {r['L']} & {int(r['median_len'])} & {r['median_lambda_occ']:.3f} & "
                         f"{r['median_lambda_size']:.3f} & {r['EBB']:.4f} & {tg} & {cp} & {advs}\\\\")
        lines.append(r"\addlinespace")
    (OUT / "tab_coldstart.tex").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    # figure: x = median history length (log), one line per panel
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for p in PANELS:
        sub = wide[(wide["panel"] == p) & wide.get("adv_vs_TweedieGP", pd.Series(dtype=float)).notna()]
        if sub.empty:
            continue
        sub = sub.sort_values("median_len")
        ax.plot(sub["median_len"], sub["adv_vs_TweedieGP"], marker=MARK[p], ms=4.5, lw=1.4,
                color=COLORS[p], label=PNL[p], markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("history retained per series (periods, log)")
    ax.set_ylabel("EBB advantage over TweedieGP (%)")
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.subplots_adjust(left=0.17, bottom=0.17, right=0.98, top=0.97)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig13_coldstart.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig13_coldstart.png", bbox_inches="tight", dpi=200)
    for d in ("v5", "v5_paper"):
        shutil.copy(FIGS / "fig13_coldstart.pdf", ROOT / "paper_v2" / d / "figs" / "fig13_coldstart.pdf")
    print("wrote coldstart_all.csv, coldstart_wide.csv, tab_coldstart.tex, fig13_coldstart")


if __name__ == "__main__":
    main()
