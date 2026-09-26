"""W13: add the two-rate forgetting negative result (tab:occdisc) to
app:negative and one clause to Limitations (1), in both manuscript dirs.
Usage: py scripts/integrity/w13_occdisc_tex.py [YYYY-MM-DD]
"""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
DIRS = [ROOT / "paper_v2" / "v5", ROOT / "paper_v2" / "v5_aistats"]
ROWS = (OUT / "tab_occdisc.tex").read_text(encoding="utf-8")

PARA = r"""
\textbf{Separate occurrence forgetting.} The M5 shortfall sits in the
occurrence block (Section~\ref{sec:main-results}), and TSB itself smooths
occurrence and size with two constants, so a third remedy gives the
occurrence counts their own discount $w^{(o)}$ while the size statistics
keep the selected $w$. $w^{(o)}$ is chosen on the same 80/20 chronological
split and criterion as $w$, from a grid that contains the tied value.
Table~\ref{tab:occdisc} reports the result. The selector keeps the tied
rate on Online Retail, Auto, and RAF, moves M5 by $0.14\%$, and moves
Carparts by $0.77\%$ (to $0.3204$, marginally ahead of TweedieGP); the
pre-registered criterion of a $1\%$ gain on M5 is not met, so the single
rate is retained. A faster occurrence rate ($w^{(o)}{=}0.8$) removes every
zero q90 forecast on M5 and lowers q90 SPL from $2.476$ to $2.452$ without
changing the mean, which locates the M5 tail problem in occurrence
persistence rather than in the size law, but the selector does not choose
it and we do not override the selector.

\begin{table*}[t]
\centering
\caption{Separate occurrence discount $w^{(o)}$ with the size discount
fixed at the selected $w$; $w^{(o)\star}$ is chosen on the initialization
window's validation tail. External mean SPL ($B{=}20$) for the tied and
selected rates, the share of evaluation rows whose q90 forecast is exactly
zero, and TweedieGP for reference.}
\label{tab:occdisc}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lccccccc}
\toprule
Panel & $w$ & $w^{(o)\star}$ & Tied & Selected & $\Delta$ (\%) & q90 $=0$ share & TweedieGP\\
\midrule
""" + ROWS + r"""\bottomrule
\end{tabular}
\end{table*}
"""

for d in DIRS:
    p = d / "sections" / "experiment_appendix.tex"
    s = p.read_text(encoding="utf-8")
    anchor = "\\label{tab:peritem}"
    i = s.index("\\end{table*}\n", s.index(anchor)) + len("\\end{table*}\n")
    if "tab:occdisc" not in s:
        s = s[:i] + PARA + s[i:]
    s = s.replace("\\subsection{Two model-side remedies that do not close the gap}",
                  "\\subsection{Three model-side remedies that do not close the gap}")
    s = s.replace("TweedieGP on Online Retail, Carparts, and M5. Two remedies within the\n\\EBB{} family are the obvious candidates: a heavier-tailed positive-size\nlaw, and an item-specific forgetting rate.",
                  "TweedieGP on Online Retail, Carparts, and M5. Three remedies within the\n\\EBB{} family are the obvious candidates: a heavier-tailed positive-size\nlaw, an item-specific forgetting rate, and a separate forgetting rate for\nthe occurrence block.")
    p.write_text(s, encoding="utf-8")
    m = d / "main.tex"
    s = m.read_text(encoding="utf-8")
    old = ("item-specific forgetting rates with empirical-Bayes shrinkage improve M5\n"
           "by $1.2\\%$ but leave it $1.8\\%$ behind TweedieGP and change no other\n"
           "panel by more than $0.1\\%$. The remaining gap therefore lies in the\n"
           "posterior---per-series temporal adaptation and a single latent process\n"
           "for occurrence and size---rather than in the shape of the predictive law.")
    new = ("item-specific forgetting rates with empirical-Bayes shrinkage improve M5\n"
           "by $1.2\\%$ but leave it $1.8\\%$ behind TweedieGP and change no other\n"
           "panel by more than $0.1\\%$, and a separate occurrence discount, chosen\n"
           "by the same validation rule, moves M5 by $0.14\\%$. The remaining gap\n"
           "therefore lies in the posterior---per-series temporal adaptation and a\n"
           "single latent process for occurrence and size---rather than in the\n"
           "shape of the predictive law or the forgetting schedule.")
    assert s.count(old) == 1, d
    m.write_text(s.replace(old, new), encoding="utf-8")
    print("patched", d.name)
