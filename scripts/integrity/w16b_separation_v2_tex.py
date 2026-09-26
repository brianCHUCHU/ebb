"""W16b: move the manuscript from the version-1 surface to version 2
(occurrence separation scaled with size separation; remedy arms), in both
directories: swap the two table bodies (tab:sepsim gains a column), update
the numeric phrases in the prose, add the remedy sentences.
Usage: py scripts/integrity/w16b_separation_v2_tex.py [YYYY-MM-DD]
"""
from __future__ import annotations
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
DIRS = [ROOT / "paper_v2" / "v5", ROOT / "paper_v2" / "v5_aistats"]
ROWS_PLACE = (OUT / "tab_separation.tex").read_text(encoding="utf-8")
ROWS_SIM = (OUT / "tab_separation_sim.tex").read_text(encoding="utf-8")
place = pd.read_csv(OUT / "separation_placement.csv").set_index("panel")
cells = pd.read_csv(OUT / "sep_surface_v2_cells.csv")

four = place.loc[["online_retail", "m5", "auto", "raf"], ["room_lo", "room_hi"]]
lo4, hi4 = float(four["room_lo"].min()), float(four["room_hi"].max())
cp_lo, cp_hi = float(place.loc["carparts", "room_lo"]), float(place.loc["carparts", "room_hi"])
cp_sim = place.loc["carparts", ["sim_learned_T30", "sim_learned_T120"]].astype(float)
or_hi = float(place.loc["online_retail", "room_hi"])
means = cells[["gain_learned", "gain_learned_cred", "gain_learned_split", "gain_learned_split_cred"]].mean()


def sub(s, old, new, path, count=None):
    n = s.count(old)
    if n == 0 and new in s:      # already applied (idempotent re-run)
        return s
    assert n >= 1, (path.name, old[:60])
    if count is not None:
        assert n == count, (path.name, n, old[:60])
    return s.replace(old, new)


def swap_body(s, label, body):
    i = s.index("\\label{" + label + "}")
    j = s.index("\\midrule\n", i) + len("\\midrule\n")
    k = s.index("\\bottomrule", j)
    return s[:j] + body + s[k:]


rng4 = f"between ${lo4:+.1f}\\%$ and ${hi4:+.1f}\\%$".replace("+", "+")
cp_rng = f"${cp_lo:.1f}$ to ${cp_hi:.1f}\\%$"
cp_sim_rng = f"(${cp_sim.max():+.1f}\\%$ to ${cp_sim.min():+.1f}\\%$)"

for d in DIRS:
    # ---- experiments.tex ----
    p = d / "sections" / "experiments.tex"; s = p.read_text(encoding="utf-8")
    s = sub(s, "supplied true partition would earn between $-0.9\\%$ and $+0.3\\%$ on\nOnline Retail, M5, Auto, and RAF, and $1.8$ to $2.4\\%$ on Carparts.",
            f"supplied true partition would earn {rng4} on\nOnline Retail, M5, Auto, and RAF, and {cp_rng} on Carparts.", p, 1)
    s = sub(s, "coordinates the simulated learned mixture also loses ($-0.9\\%$ to\n$-1.0\\%$)", f"coordinates the simulated learned mixture also loses {cp_sim_rng}", p, 1)
    s = sub(s, "resolved, its room is\nat most $0.3\\%$.", f"resolved, its room is\nat most ${or_hi:.1f}\\%$.", p, 1)
    s = sub(s, "with $w=0.90$ a true partition earns\n$1$--$2.7\\%$ across most separations, with $w=0.95$ it exceeds $1\\%$ only\nbelow $\\lambda^{(+)}\\approx0.4$, and with $w=1$ it never does. The learned\nmixture, run on the same simulated panels, is negative at every moderate\nseparation and matches the true partition only when centers are two or\nmore log-units apart and histories are long.",
            "with no separation in either block the return is within noise of\nzero; with $w=0.90$ a true partition earns $1$--$2.7\\%$ from separation\n$0.5$ upward, with $w=0.95$ it exceeds $1\\%$ only at separation $0.5$\nwith long histories ($\\lambda^{(+)}\\approx0.4$), and with $w=1$ it never\ndoes. The learned mixture, run on the same simulated panels, is negative\nat every moderate separation, and the two remedies already in the code\nbase---label sample-splitting and credibility shrinkage of the group\nvariance---do not recover the loss (grid means "
            + f"${means['gain_learned']:+.2f}\\%$, ${means['gain_learned_split']:+.2f}\\%$, ${means['gain_learned_cred']:+.2f}\\%$, and ${means['gain_learned_split_cred']:+.2f}\\%$ combined).", p, 1)
    s = sub(s, "each square is one cell of the\ncontrolled grid (separation $\\times$ discount $\\times$ length, nine paired\nreplicates)",
            "each square is one cell of the\ncontrolled grid (separation $\\times$ discount $\\times$ length, nine paired\nreplicates; occurrence separation scaled with size separation)", p, 1)
    p.write_text(s, encoding="utf-8")
    # ---- main.tex ----
    p = d / "main.tex"; s = p.read_text(encoding="utf-8")
    s = sub(s, "$1.8$ to $2.4\\%$", cp_rng, p)
    s = sub(s, "in\nsimulation the mixture objective is negative at every moderate\nseparation---identifies a target rather than hitting it.",
            "in\nsimulation the mixture objective is negative at every moderate\nseparation and neither label sample-splitting nor credibility shrinkage\nof the group variance recovers it (Appendix~\\ref{app:simulation-details})---identifies a target rather than hitting it.", p, 1)
    p.write_text(s, encoding="utf-8")
    # ---- appendix ----
    p = d / "sections" / "experiment_appendix.tex"; s = p.read_text(encoding="utf-8")
    s = sub(s, "nine paired replicates per cell (seed 20260913), four groups of sixty\nitems, $\\tau^2=0.15$, $\\sigma^2=1$, occurrence spread $0.8$ in logit held\nfixed.",
            "nine paired replicates per cell (seed 20260913), four groups of sixty\nitems, $\\tau^2=0.15$, $\\sigma^2=1$, occurrence spread in logit scaled\nwith the size separation as $0.8\\min(\\mathrm{sep},1)$ so that zero\nseparation removes structure from both blocks (a first pass held it fixed\nat $0.8$; it is archived and superseded).", p, 1)
    s = sub(s, "Because the occurrence-rate separation is held fixed, even zero\nsize separation yields about $1\\%$ at low leverage; that part of the\nreturn belongs to the occurrence block. ",
            "Two remedies for the learned mixture that already exist in the released\ncode are run as extra arms on the same panels: label sample-splitting\n(labels from one interleaved half, hyperparameters from the other) and\ncredibility shrinkage of the group variance component, alone and\ncombined. ", p, 1)
    s = sub(s, "$(w,T)$ cells at each separation: true-label $R^2$, the true partition's\ngain over a single pool, the learned mixture's gain, and the median\nnumber of components the mixture selects.}",
            "$(w,T)$ cells at each separation: true-label $R^2$, the true partition's\ngain over a single pool, the learned mixture's gain, the same with label\nsample-splitting and credibility shrinkage combined, and the median\nnumber of components the mixture selects.}", p, 1)
    s = sub(s, "Separation & $R^2$ & True (\\%) & Learned (\\%) & $K$\\\\", "Separation & $R^2$ & True (\\%) & Learned (\\%) & Learned+fix (\\%) & $K$\\\\", p, 1)
    s = sub(s, "\\begin{tabular}{lrrrr}\n\\toprule\nSeparation", "\\begin{tabular}{lrrrrr}\n\\toprule\nSeparation", p, 1)
    s = swap_body(s, "tab:sepsim", ROWS_SIM)
    s = swap_body(s, "tab:separation", ROWS_PLACE)
    p.write_text(s, encoding="utf-8")
    print("patched", d.name, "| four-panel room", round(lo4, 2), round(hi4, 2), "| Carparts", round(cp_lo, 2), round(cp_hi, 2), "| OR max", round(or_hi, 2))
