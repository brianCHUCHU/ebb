"""W12: land the ACI-EBB results, walk-forward significance, and the frontier
figure in both manuscript directories. Replaced passages are kept as
`% ORIG|` comment lines (header `% [ORIG 2026-09-07]`).

Rows generated from outputs/<date>/spl_significance_wf.csv.
Usage: py scripts/integrity/w12_aci_ebb_tex.py [YYYY-MM-DD]
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
TAG = "% [ORIG 2026-09-07]"
PANELS = ["online_retail", "carparts", "auto", "raf", "m5"]
PNL = {"online_retail": "Online Retail", "carparts": "Carparts", "auto": "Auto", "raf": "RAF", "m5": "M5"}


def orig_block(old):
    return TAG + "\n" + "".join("% ORIG| " + ln + "\n" for ln in old.rstrip("\n").split("\n"))


def rep(s, old, new, path, keep=True):
    assert s.count(old) == 1, (path.name, s.count(old), old[:70])
    return s.replace(old, (orig_block(old) if keep else "") + new)


def rep_re(s, pat, new, path):
    m = list(re.finditer(pat, s, re.S))
    assert len(m) == 1, (path.name, len(m), pat[:70])
    return s[:m[0].start()] + orig_block(m[0].group(0)) + new + s[m[0].end():]


def sig_rows():
    d = pd.read_csv(OUT / "spl_significance_wf.csv")
    lines = []
    for ref in ("EBB", "ACI-EBB"):
        for p in PANELS:
            g = d[(d["panel"] == p) & (d["reference"] == ref)]
            if g.empty:
                continue
            better = int((g["ref_better"] & g["significant_5pct"]).sum())
            worse = g[(~g["ref_better"]) & g["significant_5pct"]]["comparator"].tolist()
            und = g[~g["significant_5pct"]]["comparator"].tolist()
            fmt = lambda xs: ", ".join(x.replace("EBB", r"\EBB{}") for x in xs) if xs else "---"
            worse_s = f"{len(worse)} ({fmt(worse)})" if worse else "0"
            lines.append(f"{PNL[p]} & {ref.replace('EBB', chr(92) + 'EBB{}')} & {len(g)} & {better} & {worse_s} & {fmt(und)}\\\\")
    return "\n".join(lines) + "\n"


TAB_WFSIG = r"""
\begin{table*}[t]
\centering
\caption{Walk-forward per-series paired tests, Holm-adjusted within each
(panel, reference) family; a comparison is significant when both the
paired $t$ and the Wilcoxon adjusted $p$ are below $0.05$. Families
contain every walk-forward method with stored per-series quantiles;
TweedieGP is not in the M5 family and the static conformal wrappers are
not in the Online Retail family.}
\label{tab:wf-significance}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llcccl}
\toprule
Panel & Reference & Family & Sig.\ better & Sig.\ worse & Undecided\\
\midrule
""" + sig_rows() + r"""\bottomrule
\end{tabular}
\end{table*}
"""

FIG_FRONTIER = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figs/fig12_frontier.pdf}
\caption{\textbf{Largest shortfall against cost.} Each point is one
method; $y$ is its largest mean-SPL shortfall to the per-panel best
(Table~\ref{tab:regret}), $x$ is measured wall-clock on a log scale.
Left: fixed origin over five panels against Carparts cost per series
(Table~\ref{tab:efficiency}). Right: walk-forward over the three monthly
panels, on which every method has a measured wall-clock, against the total
walk-forward wall-clock on those panels (Table~\ref{tab:wfcost}; TweedieGP
uses twelve workers, every other method one process). The dashed line
marks $5\%$.}
\label{fig:frontier}
\end{figure*}
"""


def patch_main(path):
    s = path.read_text(encoding="utf-8")
    # abstract
    s = rep(s, r"""fixed-origin and a strict walk-forward protocol, \EBB{} and an official
Tweedie--Gaussian-process specialist are the only two methods within five
percent of the best method on every panel. \EBB{} leads on Auto and RAF
under both protocols; the specialist leads on the remaining panels by
under one percent in four of its five leads and by $2.9\%$ on M5 at fixed
origin, and it costs one to two orders of magnitude more and must be refit
at every update, where \EBB{} updates sufficient statistics in closed
form. Under walk-forward updating adaptive conformal inference leads on M5
by $4.4\%$ and \EBB{} improves on it by $6.2\%$ and $8.0\%$ on the two
panels where both are strongest. """,
            r"""fixed-origin and a strict walk-forward protocol, \EBB{} and an official
Tweedie--Gaussian-process specialist are the only two methods within five
percent of the best competing method on every panel. \EBB{} leads on Auto
and RAF under both protocols; the specialist leads on the remaining panels
by under one percent in four of its five leads and by $2.9\%$ on M5 at
fixed origin, and it costs one to two orders of magnitude more and must be
refit at every update, where \EBB{} updates sufficient statistics in
closed form. Adding the same adaptive conformal layer to \EBB{} that the
strongest conformal baseline uses makes it the walk-forward leader on M5
and Auto and closes Carparts to $0.2\%$, while hurting on Online Retail and
RAF; under equal calibration \EBB{} beats the ADIDA base on all five
panels. """, path)
    # intro
    s = rep(s, r"""walk-forward updating) and the specialist ($3.1\%$, $4.5\%$) are alone
within five percent everywhere, and every other method is more than
$15\%$ behind on at least one panel. """,
            r"""walk-forward updating) and the specialist ($3.1\%$, $4.5\%$) are alone
within five percent of the best competing method everywhere, and every
other method is more than $15\%$ behind on at least one panel. The
adaptive conformal layer that makes ADIDA competitive can be applied to
\EBB{} as well: it takes M5 and Auto and closes Carparts to $0.2\%$, but
degrades Online Retail and RAF, so we report it as a separate row rather
than folding it into the method. """, path)
    # contributions (3)
    s = rep(s, "Comparators include the official TweedieGP implementation, adaptive\nconformal inference, a published hierarchical state-space model, and a\nzero-shot foundation model.",
            "Comparators include the official TweedieGP implementation, adaptive\nconformal inference applied to the baselines and to \\EBB{} alike, a\npublished hierarchical state-space model, and a zero-shot foundation\nmodel.", path, keep=False)
    # conclusion
    s = rep(s, r"""protocols it is within $3\%$ (fixed origin) and $5\%$ (walk-forward) of
the best method everywhere, a property shared only by an official
Tweedie--Gaussian-process specialist that costs one to two orders of
magnitude more and must be refit at every update. \EBB{} leads that
specialist on Auto and RAF and trails it by under one percent on Online
Retail and Carparts and by $2.9\%$ on M5 at fixed origin, where adaptive
conformal inference leads under walk-forward updating. """,
            r"""protocols it is within $3\%$ (fixed origin) and $5\%$ (walk-forward) of
the best competing method everywhere, a property shared only by an
official Tweedie--Gaussian-process specialist that costs one to two orders
of magnitude more and must be refit at every update. \EBB{} leads that
specialist on Auto and RAF and trails it by under one percent on Online
Retail and Carparts and by $2.9\%$ on M5 at fixed origin. Under
walk-forward updating the adaptive conformal layer decides M5: applied to
\EBB{} it leads, applied to ADIDA it is second, and uncalibrated \EBB{} is
third. """, path)
    path.write_text(s, encoding="utf-8")


def patch_experiments(path):
    s = path.read_text(encoding="utf-8")
    # setup comparators
    s = rep(s, "walk-forward studies additionally include adaptive conformal inference\n(ACI) applied to ADIDA on every panel and to TSB on Online Retail.",
            "walk-forward studies additionally include adaptive conformal inference\n(ACI) applied to ADIDA and to \\EBB{} on every panel and to TSB on Online\nRetail; ACI-\\EBB{} reads \\EBB{}'s analytic predictive quantile at the\nadaptive level, with the same $\\gamma$ and update rule as ACI-ADIDA.",
            path, keep=False)
    # tab:wfprob caption
    s = rep(s, r"""sufficient statistics each block. Bold and underline rank non-degenerate
methods by SPL; on RAF the all-zero forecast is marginally lower than every
method through q90. Adaptive conformal inference leads on M5, the longest
daily panel.}""",
            r"""sufficient statistics each block; ACI-\EBB{} applies the ACI-ADIDA
calibration layer to \EBB{}'s predictive quantiles. Bold and underline
rank non-degenerate methods by SPL; on RAF the all-zero forecast is
marginally lower than every method through q90. On M5, the longest daily
panel, the adaptive conformal layer decides the ranking: ACI-\EBB{} leads,
ACI-ADIDA is second, and uncalibrated \EBB{} is third.}""", path, keep=False)
    # 5.3 paragraph
    s = rep_re(s, r"Table~\\ref\{tab:wfprob\} is the deployment-facing test, and it reproduces\n.*?core \(Table~\\ref\{tab:wfcost\}\)\.\n",
               r"""Table~\ref{tab:wfprob} is the deployment-facing test, and it reproduces
the fixed-origin picture for the uncalibrated methods. \EBB{} leads on
Auto and RAF, by $0.5\%$ and $4.3\%$ over TweedieGP; TweedieGP leads on
Online Retail and Carparts by $0.2\%$ and $0.8\%$. Per-series paired tests
(Table~\ref{tab:wf-significance}) cannot separate the two on Online
Retail, Carparts, or Auto, and favor \EBB{} on RAF. Over adaptive
conformal inference on ADIDA, \EBB{} gains $6.2\%$ on Online Retail and
$8.0\%$ on Carparts, beats every static conformal wrapper on all five
panels, and loses M5 by $4.4\%$.

The M5 loss is a calibration effect, and the same layer is available to
\EBB{}. ACI-\EBB{} reads \EBB{}'s predictive quantile at the adaptive
level instead of the nominal one. On M5 it lowers mean SPL from $1.4851$
to $1.3785$, $2.9\%$ below ACI-ADIDA and significant; it also leads Auto
($0.2901$) and brings Carparts to within $0.2\%$ of TweedieGP. On Online
Retail and RAF the same layer hurts, by $6.3\%$ and $2.3\%$, because the
per-series coverage signal is too noisy on those panels to improve an
already calibrated forecaster. We therefore report ACI-\EBB{} as a row,
not as the method. Two facts survive the mixed result: under equal
calibration \EBB{} beats the ADIDA base on all five panels (undecided only
on Online Retail), and the adaptive layer's benefit is a property of the
panel, not of the forecaster it wraps.

Table~\ref{tab:regret} and Figure~\ref{fig:frontier} summarize both
protocols by the largest shortfall to the per-panel best. Against
competing methods, \EBB{} and TweedieGP are the only ones within five
percent everywhere (\EBB{} at most $3.0\%$ at fixed origin and $4.6\%$
under walk-forward updating; TweedieGP $3.1\%$ and $4.5\%$), and every
other method is at least $15\%$ behind on some panel. Counting our own
wrapper as a competitor, uncalibrated \EBB{} trails ACI-\EBB{} by $7.7\%$
on M5. The two leaders differ in cost. TweedieGP refits per-series
variational GPs at every update, so its walk-forward runs took between
$36$ minutes (Auto) and $2.4$ hours (RAF) with twelve workers and project
to about $51$ hours on M5 at the Online Retail cadence; \EBB{} updates
sufficient statistics in closed form and completes each of the five
walk-forward runs in under $40$ seconds on one core, and ACI-\EBB{} adds
a few seconds (Table~\ref{tab:wfcost}).
""", path)
    # tab:regret caption
    s = rep(s, r"""$5\%$ of the best. A superscript gives the number of panels evaluated
when fewer than five. Per-panel regrets are in
Table~\ref{tab:regret-full}.}""",
            r"""$5\%$ of the best. A superscript gives the number of panels evaluated
when fewer than five. ACI-\EBB{} exists only under walk-forward updating.
Per-panel regrets are in Table~\ref{tab:regret-full}.}""", path, keep=False)
    # figure after the regret table
    anchor = "\\label{tab:regret}"
    i = s.index("\\end{table}\n", s.index(anchor)) + len("\\end{table}\n")
    if "fig:frontier" not in s:
        s = s[:i] + FIG_FRONTIER + s[i:]
    path.write_text(s, encoding="utf-8")


def patch_appendix(path):
    s = path.read_text(encoding="utf-8")
    s = rep(s, r"""quantile, coinciding with the zero forecast through q75; on M5 ACI-ADIDA
leads q25 through q90 and the mean, and \EBB{} leads q10.""",
            r"""quantile, coinciding with the zero forecast through q75; on M5 ACI-\EBB{}
leads every quantile from q25 up and the mean, ACI-ADIDA is second, and
uncalibrated \EBB{} leads q10. ACI-\EBB{} improves on \EBB{} on M5, Auto,
and Carparts and degrades it on Online Retail and RAF
(Table~\ref{tab:wf-significance}).""", path)
    anchor = "\\label{tab:spl-significance}"
    m = re.compile(r"\\end\{table\*?\}\n").search(s, s.index(anchor))
    i = m.end()
    if "tab:wf-significance" not in s:
        s = s[:i] + TAB_WFSIG + s[i:]
    path.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    for d in DIRS:
        patch_main(d / "main.tex")
        patch_experiments(d / "sections" / "experiments.tex")
        patch_appendix(d / "sections" / "experiment_appendix.tex")
        print("patched", d)
