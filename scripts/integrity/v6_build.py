"""V6 builder: assemble paper_v2/v6_aistats/ from paper_v2/v5_aistats/ by moving
blocks (no numbers change). Main text: theory-first story; everything not used
in the main text is demoted verbatim into a new appendix section
(app:demoted). Re-runnable: overwrites v6_aistats/main.tex and
sections/experiments_v6.tex; other files are copied once.

Usage: py scripts/integrity/v6_build.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper_v2" / "v5_aistats"
DST = ROOT / "paper_v2" / "v6_aistats"
FIX = ROOT / "outputs" / "integrity_2026-09-05" / "rescored_tab_prob.csv"
WF = ROOT / "outputs" / "2026-09-09" / "wf_all_panels_wide.csv"
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]
PN = {"online_retail": "OR", "m5": "M5", "auto": "Auto", "carparts": "Carp.", "raf": "RAF"}
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}


def between(s: str, start: str, end: str | None = None, start_inclusive: bool = True) -> str:
    i = s.index(start)
    j = s.index(end, i + len(start)) if end else len(s)
    return s[i if start_inclusive else i + len(start):j]


def cut(s: str, block: str) -> str:
    assert s.count(block) == 1, block[:60]
    return s.replace(block, "")


def strip_orig(s: str) -> str:
    """drop the ORIG/STALE bookkeeping comments (kept in v5, not needed in v6)."""
    return "\n".join(l for l in s.split("\n") if not (l.startswith("% ORIG|") or l.startswith("% [ORIG") or "STALE]" in l)) + "\n"


# --------------------------------------------------------------------------
# main table
# --------------------------------------------------------------------------
def main_table() -> str:
    fx = pd.read_csv(FIX); fx["model"] = fx["model"].replace({"TSB-HB": "EBB"})
    fxp = fx.pivot_table(index="model", columns="panel", values="spl_mean")
    wf = pd.read_csv(WF).pivot_table(index="model", columns="panel", values="mean")
    excl = {"EBB", "ACI-EBB", "EBB-rule", "TweedieGP", "ACI-ADIDA", "ACI-TSB", "Zero"}
    rows, best_names = [], {"fixed": {}, "wf": {}}
    vals = {}
    for tag, piv in (("fixed", fxp), ("wf", wf)):
        base = piv.loc[[m for m in piv.index if m not in excl]]
        vals[(tag, "base")] = base.min(axis=0)
        for p in PANELS:
            best_names[tag][p] = base[p].idxmin()
        for m in ("ACI-ADIDA", "TweedieGP", "EBB", "EBB-rule"):
            vals[(tag, m)] = piv.loc[m] if m in piv.index else pd.Series({p: float("nan") for p in PANELS})
    labels = [("base", "Best classical or static-conformal baseline"), ("ACI-ADIDA", "ACI-ADIDA"),
              ("TweedieGP", "TweedieGP (per-series GP)"), ("EBB", r"\EBB{}"), ("EBB-rule", r"\EBB{} (calibration by rule)")]
    # column-wise best among rows except the rule row
    col_best = {}
    for tag in ("fixed", "wf"):
        for p in PANELS:
            cand = {k: vals[(tag, k)][p] for k, _ in labels if k != "EBB-rule" and pd.notna(vals[(tag, k)][p])}
            col_best[(tag, p)] = min(cand, key=cand.get)
    lines = []
    for k, lab in labels:
        cells = [lab]
        for tag in ("fixed", "wf"):
            for p in PANELS:
                v = vals[(tag, k)][p]
                if pd.isna(v):
                    cells.append("---")
                else:
                    s = f"{v:.3f}"
                    cells.append(rf"\best{{{s}}}" if col_best[(tag, p)] == k else s)
        lines.append(" & ".join(cells) + r"\\")
        if k == "TweedieGP":
            lines.append(r"\midrule")
    names = "; ".join(f"{PNL[p]}: {best_names['fixed'][p]} / {best_names['wf'][p]}" for p in PANELS)
    return "\n".join(lines) + "\n", names


TABLE_ROWS, BEST_NAMES = main_table()

TAB_MAIN = r"""
\begin{table*}[t]
\centering
\caption{\textbf{Mean scaled pinball loss on five panels under two
protocols.} Fixed origin fits once and scores the full evaluation span;
walk-forward reveals one block at a time (seven days on the daily panels,
one month on the monthly ones), with every baseline refit per block and
\EBB{} updating sufficient statistics. The first row is the best of ten
classical, automatic, and static split-conformal baselines in each cell
(""" + BEST_NAMES.replace("CP-CrostonClassic", "CP-Croston").replace("CP-CrostonSBA", "CP-SBA") + r""").
ACI-ADIDA is adaptive conformal inference on ADIDA; \EBB{} (rule) applies
the same adaptive layer to \EBB{} only where a validation-tail rule inside
the initialization window selects it, and is not ranked. Bold marks the
column's best ranked method. Full tables with all quantiles, coverage,
significance, and cost are in Appendix~\ref{app:demoted} and
Appendix~\ref{app:extra}.}
\label{tab:main}
\small
\setlength{\tabcolsep}{3.6pt}
\begin{tabular}{lrrrrrrrrrr}
\toprule
 & \multicolumn{5}{c}{Fixed origin} & \multicolumn{5}{c}{Walk-forward}\\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
Method & OR & M5 & Auto & Carp. & RAF & OR & M5 & Auto & Carp. & RAF\\
\midrule
""" + TABLE_ROWS + r"""\bottomrule
\end{tabular}
\end{table*}
"""

# --------------------------------------------------------------------------
# new prose
# --------------------------------------------------------------------------
ABSTRACT = r"""\begin{abstract}
Forecasting many sparse series forces two coupled choices: how much of
each series' own past to keep, and how much to borrow from other series.
We show that in a hierarchical empirical-Bayes hurdle model with one
exponential recency operator applied to item- and group-level statistics
alike, the two are one decision. Forgetting keeps a shared prior's
leverage alive while eroding the information that resolves fine group
structure; a two-sided credibility bound makes this precise, and its
resolution corollary turns it into a diagnostic computable from the
fitting window, before any pooling structure is fitted. On five public
intermittent-demand panels (more than 19{,}000 series), read against a
controlled surface over leverage and group separation, the diagnostic is
right in both directions. Where credibility is low the shared prior pays:
the advantage over a per-series Gaussian-process specialist grows from
two to three percent at full length to seven to eleven percent as
histories shrink. Where credibility is high, refining the partition is
worth nothing and nothing is found; on the one panel with room the
mixture objective fails to collect it, in data and in simulation alike.
The forecaster that carries the diagnostic, \EBB{}, selects pooling
structure, forgetting rate, and calibration layer on one validation tail,
and is, together with that specialist, the only method within five
percent of the best competitor on every panel under both fixed-origin and
strict walk-forward protocols, at one to three orders of magnitude lower
cost and with closed-form online updates. Two evaluation conventions
obscure such results: absolute-error metrics reward the all-zero forecast
on three panels, and static split-conformal wrappers understate what
conformal prediction achieves by about twenty percent.
\end{abstract}
"""

INTRO = r"""\section{Introduction}\label{sec:intro}

Whether to fit each series on its own or to pool evidence across series
is the oldest question in forecasting many series at once, and the
working answer is still empirical: global models win on some collections
and lose on others, and one finds out by running both
\citep{monteromanso2021,januschowski2020,hewamalage2022}. The question is
sharpest for intermittent demand---long runs of zeros interrupted by
sporadic positive quantities, pervasive in spare parts, long-tail
commerce, and supply chains---because each series carries the least
evidence of its own. There, a second choice compounds the first. Old
observations can be discounted to track drift and obsolescence, but
forgetting also removes information. Local methods such as Croston, SBA,
and TSB \citep{croston1972,syntetos2005accuracy,teunter2011} adapt in
time while estimating each item separately; hierarchical probabilistic
models \citep{chapados2014,seeger2016,pitkin2024} shrink across items,
usually under a pooling structure and temporal model fixed in advance.
Pooling and forgetting are consequently designed as separate decisions.

They are not separate statistically. With exponential forgetting, an
item's effective history remains bounded, so a shared prior can retain
substantial influence even after a long observation sequence. The same
discounting, however, reduces the effective information used to estimate
each group-level prior. Stronger forgetting can therefore make pooling
more consequential while making fine pooling structure harder to resolve.
This tension is the paper's organizing principle: the forgetting rate
sets not only a temporal horizon but also the statistical resolution at
which cross-series structure can be learned, and both sides of the
trade-off are functions of one quantity---the credibility weight an item
retains after discounting---that can be computed from the fitting window
before any partition is fitted.

We develop this principle inside
\EBB{},\footnote{Empirical Bayes with Bounded memory: under the recency
operator an item's effective history stays bounded
(Section~\ref{sec:theory}), so evidence ebbs rather than accumulates.} a
recency-weighted hierarchical empirical-Bayes forecaster with
Beta--Binomial occurrence and log-normal size blocks represented by
per-item sufficient statistics. The same exponential weights enter the
item posterior and the group-level hyperparameters. A small candidate
family crosses discount factors with a global pool, the ADI/CV$^2$
taxonomy, and BIC-selected mixture partitions; one chronological
validation tail selects the discount, the structure, and, under
deployment, whether an adaptive conformal layer is applied. Estimation is
closed form apart from the mixture search, deterministic, and linear in
the number of observations.

The theory predicts an asymmetric pattern, and the experiments confirm
both halves of it. Where credibility is low the shared prior should
carry the forecast: truncating histories on five public panels, \EBB{}'s
advantage over the strongest per-series comparator, an official
Tweedie--Gaussian-process implementation, grows from two to three percent
at full length to seven to eleven percent with six months of history.
Where credibility is high, refining the pooling structure should be
worth little: a controlled surface that varies leverage and group
separation together places four panels where a supplied true partition
would earn nothing, and nothing is found; it places Carparts where one
would earn $1.7$ to $2.0\%$, and the learned partition returns $-0.08\%$
there, as the same objective does at those coordinates in simulation.
Forgetting itself is a first-order decision: all five panels select it,
and where both effects are measurable it is worth an order of magnitude
more than refinement.

The forecaster is competitive on the terms that matter for deployment.
Across the five panels and both a fixed-origin and a strict walk-forward
protocol, \EBB{} and the Gaussian-process specialist are the only methods
within five percent of the best competitor everywhere; \EBB{} leads on
two panels, trails by under one percent on two, and by $2.9\%$ on M5 at
fixed origin, at one to three orders of magnitude lower cost and with
online updates that need no refitting.

\paragraph{Contributions.}
\textbf{(1) A two-sided resolution theory and an ex ante diagnostic.}
Proposition~\ref{prop:twosided} shows that one recency operator keeps
prior leverage active but raises the sampling-variance floor against
which group heterogeneity must be resolved; Corollary~\ref{cor:resolution}
converts this into a statistic computable from the fitting window that
bounds what any partition could be worth. A controlled surface over
leverage and separation calibrates the bound, and five real panels
confirm it in both directions (Section~\ref{sec:pays}).
\textbf{(2) A forecaster that carries the diagnostic.} \EBB{} selects
pooling structure, forgetting rate, and calibration layer on one
leakage-safe validation tail; its predictive law is closed form,
bit-reproducible, costs milliseconds per series, and updates online. It
matches a per-series Gaussian-process specialist to within a percent on
most panels and is the only other method within five percent of the best
competitor on all of them (Section~\ref{sec:accuracy}).
\textbf{(3) Two evaluation cautions.} Absolute-error metrics reward the
all-zero forecast on three of five panels and reverse the ranking of a
known-correct partition in simulation, and static split-conformal
wrappers understate what conformal prediction achieves by about twenty
percent relative to an adaptive construction
(Section~\ref{sec:cautions}). Comparators include the official TweedieGP
implementation, adaptive conformal inference applied to baselines and to
\EBB{} alike, a published hierarchical state-space model, and a zero-shot
foundation model.
"""

MIXTURE_SHORT = r"""\subsection{Candidate pooling structures}\label{sec:mixture}

Let $\mathcal R$ contain three ways to construct $g(\cdot)$ from the
fitting data, ordered coarse to fine: a \emph{global} pool with one prior
for the panel; the \emph{taxonomy} of four ADI/CV$^2$ demand-pattern
classes at the classical thresholds \citep{syntetos2005categorization};
and a \emph{learned} partition, hard assignments from a mixture of
occurrence--size prior components with $K\in\{1,\dots,6\}$ selected by
BIC, fit by EM with an analytic responsibility update and a
deterministic initialization (Appendix~\ref{app:demoted},
Table~\ref{tab:pool-candidates}). The question this paper asks is not
which rung is best in general but whether the discounted data support
moving up the ladder at all; Section~\ref{sec:resolution} gives the
condition and Section~\ref{sec:pays} reports what happens when it fails.
Every data-dependent label is computed from the selector's fitting head,
never its validation tail.

"""

LIMITATIONS = r"""\section{Limitations}\label{sec:limitations}

\textbf{(1) The remaining gap to a per-series specialist.} The
log-normal size block keeps item posteriors analytic, and official
TweedieGP is better on Online Retail, Carparts, and M5 at fixed origin
and on Online Retail and Carparts under walk-forward updating, by under
one percent except on M5 ($2.9\%$). Three model-side remedies were tested
and none closes it (Appendix~\ref{app:negative}): heavier-tailed size
laws move mean SPL by at most $0.003$, item-specific forgetting rates
improve M5 by $1.2\%$ but leave it $1.8\%$ behind, and a separate
occurrence discount moves M5 by $0.14\%$. The gap lies in the posterior,
where the specialist adapts each series in time with one latent process
for occurrence and size, not in the predictive law's shape or the
forgetting schedule.

\textbf{(2) What the resolution condition is, and what it is not.}
Corollary~\ref{cor:resolution} uses a Gaussian working model and a
$\chi^2$ approximation, accurate except for small groups with strongly
unequal counts, where it understates the collapse probability by up to
$0.21$; it is an estimator-specific estimability statement, not an
information-theoretic bound. Its calibration on the controlled surface
is simulation-based, and the room it assigns to a real panel is read
within the panel's forgetting regime with an interpolation across two
series lengths. On the one panel with room, Carparts, neither the real
nor the simulated learned partition collects it, and neither label
sample-splitting nor credibility shrinkage of the group variance recovers
the loss in simulation (Appendix~\ref{app:simulation-details}); a
partition objective that maximizes resolvable separation rather than
marginal likelihood is the implicated next step.

\textbf{(3) Scope.} We do not run the external hierarchical Bayesian
implementations we cite; a same-model Gibbs sampler matches plug-in EB
within $0.1\%$ (Appendix~\ref{app:mcmc}), so the inference half of that
gap is closed and the model-class half is not. The adaptive conformal
comparison is one ACI design; given how much it moves the baseline, the
remaining designs deserve more attention. Selection uses one pinball
score; the discount selected at fixed origin is reused under walk-forward
updating without re-selection; DeepState, Chronos-Bolt, and Deep Renewal
Process comparisons use subset or proxy protocols
(Appendix~\ref{app:neural}); CRPS and a cost-based criterion remain
unrun.
"""

CONCLUSION = r"""\section{Conclusion}\label{sec:conclusion}

Forgetting and pooling are one decision. In a hierarchical
empirical-Bayes hurdle model with a single recency operator, the
credibility weight an item retains after discounting governs both how
much a shared prior can move its forecast and how well the data can
resolve the structure of that prior; a two-sided bound makes the
trade-off explicit and its corollary makes it computable before a
partition is fitted. On five public panels the diagnostic is right in
both directions. Where credibility is low the shared prior pays, and it
pays more the shorter the histories become. Where it is high, refining
the pooling structure is worth nothing and nothing is found; on the one
panel with room, the mixture objective fails to collect it in data and in
simulation alike, which locates the open problem in the objective rather
than in the bound. The forecaster that carries the diagnostic selects
structure, forgetting, and calibration on one validation tail and is,
with a per-series Gaussian-process specialist, the only method within
five percent of the best competitor on every panel under both protocols,
at a small fraction of the cost. Two conventions of the field obscure
results of this kind: absolute-error metrics that reward the all-zero
forecast, and static conformal wrappers that understate what conformal
prediction can do. The next steps follow from the gap the diagnostic
exposes: partition objectives that optimize resolvable separation, and
decision-aware sequential selection.
"""

NEW_BIB = r"""
\bibitem[Januschowski et~al.(2020)]{januschowski2020}
T.~Januschowski, J.~Gasthaus, Y.~Wang, D.~Salinas, V.~Flunkert,
M.~Bohlke-Schneider, and L.~Callot.
\newblock Criteria for classifying forecasting methods.
\newblock \emph{International Journal of Forecasting}, 36(1):167--177, 2020.

\bibitem[Hewamalage et~al.(2022)]{hewamalage2022}
H.~Hewamalage, C.~Bergmeir, and K.~Bandara.
\newblock Global models for time series forecasting: A simulation study.
\newblock \emph{Pattern Recognition}, 124:108441, 2022.

"""

# --------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------
def main():
    if not DST.exists():
        shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("*.log", "*.aux", "*.out", "*.txt", "*.pdf", "main.pdf"))
        for f in ("main.pdf",):
            pass
    (DST / "sections").mkdir(exist_ok=True)
    M = (SRC / "main.tex").read_text(encoding="utf-8")
    E = (SRC / "sections" / "experiments.tex").read_text(encoding="utf-8")

    # ---- main.tex pieces ----
    pre = M[: M.index("\\begin{abstract}")]
    pre = pre.replace("\\runningtitle{EBB: Pooling under Forgetting for Probabilistic Intermittent-Demand Forecasting}",
                      "\\runningtitle{When Does Pooling Pay? Forgetting, Credibility, and Intermittent Demand}")
    pre = pre.replace("\\aistatstitle{EBB: Pooling under Forgetting for Probabilistic\\\\\nIntermittent-Demand Forecasting}",
                      "\\aistatstitle{When Does Pooling Pay? A Credibility Diagnostic for\\\\\nForgetting and Pooling in Intermittent-Demand Forecasting}")
    related = strip_orig(between(M, "\\section{Related Work}\\label{sec:related}", "\\section{\\EBB{}: Recency-Weighted"))
    method_a = strip_orig(between(M, "\\section{\\EBB{}: Recency-Weighted", "\\subsection{Candidate pooling structures}"))
    mixture_full = strip_orig(between(M, "\\subsection{Candidate pooling structures}", "\\subsection{Leakage-safe joint selection}"))
    select = strip_orig(between(M, "\\subsection{Leakage-safe joint selection}", "\\subsection{Forecasts and computation}"))
    resdiag = between(select, "\\paragraph{Resolution diagnostics.}", "Appendix~\\ref{app:implementation} gives split edge cases")
    select = select.replace(resdiag, (
        "\\paragraph{Resolution diagnostics.} Fitting each candidate already\n"
        "produces everything the resolution analysis of Section~\\ref{sec:resolution}\n"
        "needs: for every $(r,w)$ and group $g$ the selector records, from the fitting\n"
        "head only, the median credibility weights $\\lambda_i^{(o)}$ and\n"
        "$\\lambda_i^{(+)}$, the share of items with $\\lambda_i^{(+)}<0.1$, the group\n"
        "sizes, and the resolution statistic $\\hat z_g$ whose nonpositive values\n"
        "coincide with the collapse event of Proposition~\\ref{prop:twosided}\n"
        "(Appendix~\\ref{app:demoted}). They are computed before any external\n"
        "observation is seen, and Section~\\ref{sec:pays} reads them.\n\n"))
    complex_full = strip_orig(between(M, "\\subsection{Forecasts and computation}", "\\section{Resolution under Forgetting}"))
    complex_full = complex_full[: complex_full.rfind("% ====")] if "% ====" in complex_full else complex_full
    cost_para = between(complex_full, "Two properties of this cost profile", "\n\n")
    complex_short = complex_full.replace(cost_para + "\n\n", "").replace(cost_para, "")
    theory = strip_orig(between(M, "\\section{Resolution under Forgetting}", "\\input{sections/experiments}"))
    theory = theory[: theory.rfind("% Experiments are maintained")] if "% Experiments are maintained" in theory else theory
    prop1 = between(theory, "\\subsection{Persistent prior leverage}", "\\subsection{Variance-component resolution}")
    PROP1_SHORT = (
        "\\subsection{Persistent prior leverage}\n\n"
        "Fix an item and suppress its subscripts. Because\n"
        "$n(w)=\\sum_{t=1}^{T}w^{T-t}\\le(1-w)^{-1}$ for $w<1$, the occurrence\n"
        "credibility weight cannot converge to one: $\\lambda(w)\\le\\{1+(1-w)\\phi\\}^{-1}<1$\n"
        "for every horizon, whereas $\\lambda(1)\\to1$ as $T\\to\\infty$\n"
        "(Proposition~\\ref{prop:active}, Appendix~\\ref{app:demoted}). Forgetting\n"
        "keeps the pooled prior relevant indefinitely; the statement is about\n"
        "leverage, not benefit.\n\n")
    theory = theory.replace(prop1, PROP1_SHORT)
    problem_fig = between(M, "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{fig01_problem.png}", "\\end{figure*}\n") + "\\end{figure*}\n"
    bib = between(M, "\\begin{thebibliography}", "\\end{thebibliography}") + "\\end{thebibliography}\n"
    bib = bib.replace("\\end{thebibliography}", NEW_BIB + "\\end{thebibliography}")
    appendix = M[M.index("\\appendix"):M.index("\\begin{thebibliography}")] if M.index("\\appendix") < M.index("\\begin{thebibliography}") else M[M.index("\\appendix"):]
    # the v5 file has bibliography BEFORE \appendix: handle both orders
    if M.index("\\appendix") > M.index("\\begin{thebibliography}"):
        appendix = M[M.index("\\appendix"):]
        appendix = appendix[: appendix.index("\\end{document}")]
    else:
        appendix = appendix[: appendix.index("\\end{document}")] if "\\end{document}" in appendix else appendix

    # ---- experiments pieces ----
    setup = strip_orig(between(E, "\\subsection{Experimental design}", "\\subsection{Probabilistic forecasting across panels}"))
    fixed = strip_orig(between(E, "\\subsection{Probabilistic forecasting across panels}", "\\subsection{Strict walk-forward forecasting}"))
    wf = strip_orig(between(E, "\\subsection{Strict walk-forward forecasting}", "\\subsection{Short histories}"))
    cold = strip_orig(between(E, "\\subsection{Short histories}", "\\subsection{Controlled test"))
    sim = strip_orig(between(E, "\\subsection{Controlled test", "\\subsection{When can pooling pay?}"))
    lev = strip_orig(between(E, "\\subsection{When can pooling pay?}"))
    frontier_fig = between(wf, "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.96\\textwidth]{figs/fig12_frontier.pdf}", "\\end{figure*}\n") + "\\end{figure*}\n"
    three_para = between(wf, "Three features of this table deserve more attention than the margins.", "\n\n")
    wf_app = cut(wf, three_para).replace("\\label{fig:frontier}", "\\label{fig:frontier-full}")
    sep_fig = between(lev, "\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{figs/fig14_separation_surface.pdf}", "\\end{figure*}\n") + "\\end{figure*}\n"
    room_paras = between(lev, "\\paragraph{Where the room is, and whether it is collected.}", "\\paragraph{High leverage is an amplifier")
    forget_para = between(lev, "\\paragraph{Forgetting is an order of magnitude larger.}", "\n\n")
    lev_app = cut(cut(cut(lev, sep_fig), room_paras), forget_para + "\n\n")
    cold_fig = between(cold, "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.98\\linewidth]{figs/fig13_coldstart.pdf}", "\\end{figure}\n") + "\\end{figure}\n"
    cold_prose = cut(cold, cold_fig).replace("\\subsection{Short histories}\\label{sec:coldstart}\n\n", "")
    iw = cold_prose.find(" In words:")
    if iw > 0:   # drop the generated numeric sentence; the figure carries it
        je = cold_prose.find("\nBoth methods degrade gracefully", iw)
        cold_prose = cold_prose[:iw] + "\n" + cold_prose[je + 1:]
    k2 = three_para.find(" Second, \\emph{coverage")
    three_para = (three_para[:k2] if k2 > 0 else three_para).replace(
        "Three features of this table deserve more attention than the margins.\nFirst, ",
        "The first caution concerns the probabilistic baseline: ")
    # rename the used figure to single-column width in main text (frontier stays figure*)

    experiments_v6 = r"""% =========================================================================
% EBB v6 — sections/experiments_v6.tex (assembled by scripts/integrity/v6_build.py)
% =========================================================================
\section{Experiments}\label{sec:experiments}

""" + setup + r"""
% =========================================================================
\subsection{Where pooling pays}\label{sec:pays}
% =========================================================================

The diagnostic of Section~\ref{sec:resolution} makes two predictions:
that a shared prior's return grows as an item's own credibility falls, and
that refining the pooling structure returns something only where the
discounted data both leave room and resolve the structure. We test each
side on the five panels, using the credibility weights at the audited
configurations (Appendix~\ref{app:demoted}, Table~\ref{tab:leverage})
and a controlled study that varies leverage and group separation together
(Appendix~\ref{app:demoted}, Table~\ref{tab:sanity};
Appendix~\ref{app:simulation-details}).

\paragraph{Short histories: the shared prior pays.}
""" + cold_prose.rstrip("\n") + "\n\n" + cold_fig + "\n" + sep_fig + "\n" + room_paras.rstrip("\n") + "\n\n" + forget_para.rstrip("\n") + r"""

% =========================================================================
\subsection{Accuracy and cost}\label{sec:accuracy}
% =========================================================================
""" + TAB_MAIN + r"""
Table~\ref{tab:main} reports the comparison that a practitioner would
make. At fixed origin, \EBB{} attains the lowest mean SPL on Auto and RAF,
by $1.6\%$ and $3.0\%$ over the official TweedieGP implementation, both
significant; TweedieGP attains it on Online Retail, Carparts, and M5, by
$0.05\%$, $0.5\%$, and $2.9\%$, of which only the M5 margin is
significant. Against every other comparator \EBB{} is best on all five
panels, by $10.1\%$ over iETS on Online Retail and $1.2\%$ over
CP-Croston on M5 (Appendix~\ref{app:demoted}, Table~\ref{tab:prob};
significance in Appendix~\ref{app:extra}, Table~\ref{tab:spl-significance}).
Under strict walk-forward updating the picture repeats for the
uncalibrated methods: \EBB{} leads on Auto and RAF, TweedieGP on Online
Retail and Carparts by $0.2\%$ and $0.8\%$, and per-series paired tests
cannot separate the two on three of the four panels where both run.
Adaptive conformal inference on ADIDA leads M5 by $4.4\%$; the same layer
applied to \EBB{} lowers its M5 loss from $1.485$ to $1.379$, $2.9\%$
below ACI-ADIDA, but degrades Online Retail and RAF, so whether to apply
it is itself selected on the validation tail. The rule applies the layer
on M5 and Auto, withholds it elsewhere, matches the ex-post better variant
on four of five panels, and leaves \EBB{} (rule) within $0.8\%$ of the
best competing method on every panel (Appendix~\ref{app:demoted},
Tables~\ref{tab:wfprob}--\ref{tab:regret}; Appendix~\ref{app:extra},
Table~\ref{tab:aci-rule}).

""" + r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figs/fig12b_frontier_wf.pdf}
\caption{\textbf{Largest shortfall against cost under walk-forward
updating.} Each point is one method on the three monthly panels, on which
every method has a measured wall-clock: $y$ is its largest mean-SPL
shortfall to the per-panel best, $x$ the total walk-forward wall-clock
(TweedieGP uses twelve workers, every other method one process). The
fixed-origin counterpart and the full regret tables are in
Appendix~\ref{app:demoted}.}
\label{fig:frontier}
\end{figure}
""" + r"""
Figure~\ref{fig:frontier} summarizes both protocols by the largest
shortfall to the per-panel best against measured cost. \EBB{} and
TweedieGP are the only methods within five percent of the best competitor
everywhere (\EBB{} at most $3.0\%$ at fixed origin and $4.6\%$ under
walk-forward updating; TweedieGP $3.1\%$ and $4.5\%$); every other method
is at least $15\%$ behind on some panel. The two differ in cost by one to
three orders of magnitude: TweedieGP refits per-series variational GPs at
every update, so its walk-forward runs took between $36$ minutes and
$2.4$ hours with twelve workers and project to about $51$ hours on M5,
whereas \EBB{} completes each of the five walk-forward runs in under $40$
seconds on one core, updating sufficient statistics in closed form
(Appendix~\ref{app:efficiency}). Point forecasts are competitive but not
uniformly best (Appendix~\ref{app:point}).

% =========================================================================
\subsection{Two evaluation cautions}\label{sec:cautions}
% =========================================================================

""" + three_para.rstrip("\n") + r"""
The second caution concerns point metrics. The all-zero forecast attains
the lowest MAE on three of five panels and, in the controlled study, a
known-correct partition is reported as harmful by MAE while improving
every proper scoring rule (Appendix~\ref{app:demoted},
Table~\ref{tab:sanity}); absolute-error criteria should not be the sole
basis for comparing intermittent-demand forecasters, and we use them
only as diagnostics.
"""

    demoted = r"""
\section{Main-text material in full}\label{app:demoted}

This appendix keeps, verbatim, the tables, figures, and paragraphs of the
long-form manuscript that the main text summarizes.

\subsection{The two coupled choices in real data}
""" + problem_fig + r"""
\subsection{Persistent prior leverage}
""" + prop1.replace("\\subsection{Persistent prior leverage}\n", "", 1) + r"""
\subsection{Resolution diagnostics recorded by the selector}
""" + resdiag.replace("\\paragraph{Resolution diagnostics.}\n", "", 1) + r"""
\subsection{Pooling candidates and computational profile}
""" + mixture_full.replace("\\subsection{Candidate pooling structures}\\label{sec:mixture}", "").lstrip("\n") + "\n" + cost_para + r"""

\subsection{Fixed-origin results in full}
""" + fixed.replace("\\subsection{Probabilistic forecasting across panels}\n\\label{sec:main-results}", "\\label{sec:main-results}") + r"""
\subsection{Walk-forward results in full}
""" + wf_app.replace("\\subsection{Strict walk-forward forecasting}\\label{sec:walkforward}", "\\label{sec:walkforward}") + r"""
\subsection{Controlled test of the resolution mechanism}
""" + sim.replace("\\subsection{Controlled test of the resolution mechanism}", "", 1) + r"""
\subsection{Leverage, selection, and ablation}
""" + lev_app.replace("\\subsection{When can pooling pay?}\\label{sec:leverage}", "\\label{sec:leverage}") + "\n\\input{sections/room_app}\n"

    # write experiments_v6 and the demoted appendix as section files
    (DST / "sections" / "experiments_v6.tex").write_text(experiments_v6, encoding="utf-8")
    (DST / "sections" / "demoted.tex").write_text(demoted, encoding="utf-8")

    # ---- appendix: insert the demoted section before "Additional results" ----
    app = appendix.replace("\\section{Additional results}\\label{app:extra}", "\\input{sections/demoted}\n\n\\section{Additional results}\\label{app:extra}", 1)
    app = strip_orig(app)

    main_v6 = (pre + ABSTRACT + "\n" + INTRO + "\n" + related + method_a + MIXTURE_SHORT + select + complex_short
               + theory + "\\input{sections/experiments_v6}\n\n" + LIMITATIONS + "\n" + CONCLUSION + "\n"
               + bib + "\n" + app + "\n\\end{document}\n")
    (DST / "main.tex").write_text(main_v6, encoding="utf-8")
    print("wrote", DST / "main.tex", "| sections/experiments_v6.tex, sections/demoted.tex")
    print("best baseline names:", BEST_NAMES)


if __name__ == "__main__":
    main()
