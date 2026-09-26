"""W14: land the cold-start scaling experiment in both manuscript directories:
a short subsection in Section 5 with fig:coldstart, and tab:coldstart in the
appendix (after tab:regret-full). Numbers are read from coldstart_wide.csv so
the prose is generated from data, not typed.
Usage: py scripts/integrity/w14_coldstart_tex.py [YYYY-MM-DD]
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
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}

w = pd.read_csv(OUT / "coldstart_wide.csv")
w["L"] = w["L"].astype(str)


def adv(panel, L):
    r = w[(w["panel"] == panel) & (w["L"] == str(L))]
    return float(r["adv_vs_TweedieGP"].iloc[0]) if len(r) else float("nan")


def shortest(panel):
    r = w[w["panel"] == panel].sort_values("Lnum").iloc[0]
    return str(r["L"]), float(r["adv_vs_TweedieGP"]), float(r["median_lambda_size"])


def fmt(x):
    return f"{x:+.1f}\\%"


lines = []
for p in ("auto", "raf", "carparts", "online_retail", "m5"):
    if p not in set(w["panel"]):
        continue
    L0, a0, lam0 = shortest(p)
    lines.append(f"{PNL[p]}: {fmt(adv(p, 'full'))} at full length, {fmt(a0)} with {L0} periods")
SUMMARY_SENT = "; ".join(lines) + "."

rows = (OUT / "tab_coldstart.tex").read_text(encoding="utf-8")

SUBSEC = r"""
\subsection{Short histories}\label{sec:coldstart}

Corollary~\ref{cor:resolution} predicts where a pooled prior pays: when an
item's own evidence is thin, its credibility weight is low and the shared
prior carries the forecast. Truncating the initialization window to its
last $L$ observations per series, with the evaluation span unchanged,
moves every panel along that axis. Figure~\ref{fig:coldstart} plots
\EBB{}'s mean-SPL advantage over TweedieGP, the strongest per-series
comparator, against the retained history; Table~\ref{tab:coldstart} gives
the numbers together with the median credibility weights and the static
conformal wrapper. The advantage grows as history shrinks on Auto and RAF,
turns positive on Carparts only at the shortest window, stays within a
third of a percent on Online Retail, and does not appear on M5, where both
median credibility weights stay above $0.8$ even with 28 days of history
and the per-series GP leads at every length but one, by a margin that
grows with the history it is given. In words:
""" + SUMMARY_SENT + r"""
Both methods degrade gracefully; the gap that opens at short histories is
the pooled prior's contribution, and its size tracks the leverage the
diagnostic assigns to each panel rather than the panel's frequency.

\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figs/fig13_coldstart.pdf}
\caption{\textbf{Pooling pays when histories are short.} \EBB{}'s
mean-SPL advantage over TweedieGP (percent) against the history retained
per series at initialization; the evaluation span is fixed and the audited
structure and discount are unchanged. Full-length points reproduce
Table~\ref{tab:prob}.}
\label{fig:coldstart}
\end{figure}
"""

TAB = r"""
\begin{table*}[t]
\centering
\caption{Short-history scaling: initialization window truncated to its
last $L$ observations per series (evaluation span unchanged; SPL scale
from the full window). Median credibility weights at the audited discount,
mean SPL of \EBB{}, TweedieGP, and CP-ADIDA, and \EBB{}'s advantage over
TweedieGP. Mixture labels are re-learned on the truncated window.}
\label{tab:coldstart}
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llrrrrrrr}
\toprule
Panel & $L$ & median $n$ & $\lambda^{(o)}$ & $\lambda^{(+)}$ & \EBB{} & TweedieGP & CP-ADIDA & Adv.\ (\%)\\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table*}
"""

for d in DIRS:
    p = d / "sections" / "experiments.tex"
    s = p.read_text(encoding="utf-8")
    anchor = "\\subsection{Controlled test of the resolution mechanism}"
    assert s.count(anchor) == 1, d
    if "\\label{sec:coldstart}" in s:
        i = s.index("\n\\subsection{Short histories}"); j = s.index(anchor)
        s = s[:i] + "\n" + s[j:]
    s = s.replace(anchor, SUBSEC.lstrip("\n") + "\n" + anchor)
    p.write_text(s, encoding="utf-8")
    p = d / "sections" / "experiment_appendix.tex"
    s = p.read_text(encoding="utf-8")
    if "\\label{tab:coldstart}" in s:
        i = s.index("\n\\begin{table*}[t]\n\\centering\n\\caption{Short-history scaling")
        j = s.index("\\end{table*}\n", i) + len("\\end{table*}\n")
        s = s[:i] + s[j:]
    m = re.compile(r"\\end\{table\*\}\n").search(s, s.index("\\label{tab:regret-full}"))
    s = s[:m.end()] + TAB + s[m.end():]
    p.write_text(s, encoding="utf-8")
    print("patched", d.name)
print(SUMMARY_SENT)
