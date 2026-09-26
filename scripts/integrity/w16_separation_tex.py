"""W16: land the separation-resolved controlled study (S1-S3) in both
manuscript directories: 5.5 sentence, 5.6 two paragraphs, fig10 caption,
fig:separation figure, abstract / intro / contributions / conclusion /
limitation (4) sentences, appendix paragraph + tab:sepsim + tab:separation.
Replaced passages are kept as `% ORIG|` lines under `% [ORIG 2026-09-13]`.
Usage: py scripts/integrity/w16_separation_tex.py [YYYY-MM-DD]
"""
from __future__ import annotations
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
DIRS = [ROOT / "paper_v2" / "v5", ROOT / "paper_v2" / "v5_aistats"]
TAG = "% [ORIG 2026-09-13]"
ROWS_PLACE = (OUT / "tab_separation.tex").read_text(encoding="utf-8")
ROWS_SIM = (OUT / "tab_separation_sim.tex").read_text(encoding="utf-8")


def orig_block(old):
    return TAG + "\n" + "".join("% ORIG| " + ln + "\n" for ln in old.rstrip("\n").split("\n"))


def rep(s, old, new, path, keep=True):
    assert s.count(old) == 1, (path.name, s.count(old), old[:70])
    return s.replace(old, (orig_block(old) if keep else "") + new)


FIG = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{figs/fig14_separation_surface.pdf}
\caption{\textbf{Room depends on separation as well as leverage, and the
real panels have little of it.} Left: each square is one cell of the
controlled grid (separation $\times$ discount $\times$ length, nine paired
replicates), placed at its median size-block credibility and at the share
of item log-size variance its true partition explains ($R^2$), colored by
the true partition's SPL gain over a single pool. Real panels are placed
at the same two coordinates, with $R^2$ measured under the learned
partition (filled) and under the objective-free ADI/CV$^2$ taxonomy (open);
labels give the learned partition's realized refinement gain. Right: the
true partition's gain and the learned mixture's gain against separation,
averaged over the six $(w,T)$ cells, with the range across cells shaded.
Room read within each panel's own forgetting regime is in
Table~\ref{tab:separation}.}
\label{fig:separation}
\end{figure*}
"""

SEC56_A_OLD = r"""The remaining three do have
room. Carparts is extreme---median $0.224$, with $22.5\%$ of items
essentially fully shrunk---and Online Retail, at $0.553$ under the
selected discount, sits squarely in the regime where
Section~\ref{sec:synthetic} shows a supplied true partition earning more
than two percent."""
SEC56_A_NEW = r"""The remaining three retain
leverage: Carparts is extreme---median $0.224$, with $22.5\%$ of items
essentially fully shrunk---and Online Retail sits at $0.553$ under the
selected discount. Leverage alone does not fix the prize, however. The
controlled study shows that the true partition's return at a given
$\lambda^{(+)}$ depends on the forgetting regime that produced it and on
how far the groups are apart, so the panels are placed on the full
surface below."""

SEC56_B_OLD_START = "\\paragraph{What binds is learnability, not leverage.}"
SEC56_B_OLD_END = "Corollary~\\ref{cor:resolution} bounds the prize rather than the shortfall.\n"
SEC56_B_NEW = r"""\paragraph{Where the room is, and whether it is collected.}
Figure~\ref{fig:separation} places the five panels on a surface that
varies leverage and group separation together. Each panel's size-block
separation is measured as the share of item log-size variance explained
by a partition, $R^2$, under its learned partition ($0.44$ to $0.95$) and
under the objective-free ADI/CV$^2$ taxonomy ($0.01$ to $0.30$); the
simulated cells carry the same statistic under their true labels. Read
within each panel's own forgetting regime (Table~\ref{tab:separation}), a
supplied true partition would earn between $-0.9\%$ and $+0.3\%$ on
Online Retail, M5, Auto, and RAF, and $1.8$ to $2.4\%$ on Carparts. The
learned partitions return $-0.05\%$, $+0.58\%$, $-0.31\%$, $+0.03\%$, and
$-0.08\%$: within the surface's range on the four panels without room,
and about two points short on the one panel with it. At Carparts's
coordinates the simulated learned mixture also loses ($-0.9\%$ to
$-1.0\%$), and across the grid it is negative at every moderate
separation (Figure~\ref{fig:separation}, right), recovering the true
partition's return only when centers are two or more log-units apart and
histories are long. The earlier reading of Online Retail as the decisive
case rested on an envelope that pooled separations; resolved, its room is
at most $0.3\%$.

This separates the two explanations that aggregate results cannot. On
four panels the credibility weights and the separation together say there
is little to exploit, and the estimator's near-zero refinement gains are
the predicted outcome, not a failure. On Carparts the structure would pay
and the objective does not find it; that the same objective fails on
simulated panels at the same coordinates makes learnability, not the
bound, the binding constraint there. Corollary~\ref{cor:resolution}
bounds the prize; the surface says where the prize is nonzero.
"""

FIG10_OLD = r"""Every real panel lies far below the envelope at
its own leverage---most visibly Online Retail, which the selected
discount moves into the high-return region of the size block while its
learned structure returns $-0.05\%$.}"""
FIG10_NEW = r"""The grey curves pool the three separations of the
controlled grid; Figure~\ref{fig:separation} resolves them. Read against
the pooled envelope, every real panel lies below it at its own leverage.}"""

SEC55_OLD = r"""$w=0.90$ rows are bought against a noisier $\hat\tau_g^2$ and do not
continue to grow without limit."""
SEC55_NEW = r"""$w=0.90$ rows are bought against a noisier $\hat\tau_g^2$ and do not
continue to grow without limit. Extending the grid with a separation axis
(Appendix~\ref{app:simulation-details}; group centers $0$ to $3$
log-units apart) shows that the return is governed by the forgetting
regime as much as by leverage: with $w=0.90$ a true partition earns
$1$--$2.7\%$ across most separations, with $w=0.95$ it exceeds $1\%$ only
below $\lambda^{(+)}\approx0.4$, and with $w=1$ it never does. The learned
mixture, run on the same simulated panels, is negative at every moderate
separation and matches the true partition only when centers are two or
more log-units apart and histories are long."""

ABS_OLD = r"""That bound is not what binds in practice. On the panel with the
most room---a median size-block weight of $0.553$, where a supplied true
partition earns over two percent in controlled study---the learned
partition returns $-0.05\%$. Across all five panels, refining the pooling
structure improves external accuracy on none, and the internal gain
exceeds selector noise on only one. The constraint is therefore not that
cross-series structure is unavailable to exploit, but that current mixture
objectives do not recover a partition worth shrinking toward; we give the
diagnostic that separates the two cases and show that all five public
panels fall on the second side."""
ABS_NEW = r"""Placing each panel on a controlled surface that varies leverage
and group separation together, the study assigns room to one panel:
Carparts, where a supplied true partition earns $1.8$ to $2.4\%$ and the
learned partition returns $-0.08\%$, a failure the same objective
reproduces at the same coordinates in simulation. On the other four the
surface predicts no room and none is found; across all five, refining the
pooling structure improves external accuracy on none. The diagnostic thus
separates panels without exploitable structure from the one panel where
the structure exists and current mixture objectives do not recover it."""

INTRO_OLD = r"""and the diagnostic shows this is not because the panels lack room. Three
of them retain substantial credibility leverage, and one---Online
Retail---sits at exactly the level where a supplied true partition earns
more than two percent in controlled study, while its learned partition
returns $-0.05\%$. What binds is therefore not the resolution limit but
learnability: the structure would pay if it were correct, and the mixture
objective does not find it. Separating those two explanations is what the
diagnostic is for, and it is the distinction between a resolution limit
and an absence of useful cross-series structure."""
INTRO_NEW = r"""and a controlled surface over leverage and group separation says why.
Four panels sit where a supplied true partition would earn nothing; one,
Carparts, sits where it would earn $1.8$ to $2.4\%$, and its learned
partition returns $-0.08\%$. The same mixture objective loses at those
coordinates in simulation, so on the one panel with room what binds is
learnability, and on the other four it is the absence of room. Separating
those two explanations is what the diagnostic is for."""

CONTRIB_OLD = r"""Learned pooling improves external accuracy on
none of the five panels, and we show this is not attributable to a lack of
leverage: on the panel with the most room the learned partition still
loses."""
CONTRIB_NEW = r"""Learned pooling improves external accuracy on
none of the five panels; a controlled surface over leverage and group
separation shows that four panels have no room and that on the fifth,
Carparts, both the real objective and its simulated counterpart fail to
collect it."""

CONCL_OLD = r"""That bound is real but not binding. Three panels retain substantial room,
and one of them, Online Retail, sits at the leverage where a supplied true
partition earns more than two percent in controlled study; its learned
partition returns $-0.05\%$. The constraint on cross-series pooling in
these data is therefore learnability rather than leverage: the structure
would pay if it were correct, and current mixture objectives do not find
it. Separating those two explanations required the diagnostic, and the
diagnostic costs nothing beyond the fit that produces it."""
CONCL_NEW = r"""Once group separation is placed on the same surface as leverage, that
bound binds on four of five panels: a supplied true partition would earn
nothing there, and nothing is found. On Carparts it would earn $1.8$ to
$2.4\%$, the learned partition returns $-0.08\%$, and the same objective
loses at those coordinates in simulation. The constraint on cross-series
pooling in these data is therefore the absence of room on most panels and
learnability on the one panel that has it. Separating those two
explanations required the diagnostic, and the diagnostic costs nothing
beyond the fit that produces it."""

LIM4_OLD = r"""Our
strongest negative result---leverage exists and the learned partition does
not exploit it---identifies a target rather than hitting it."""
LIM4_NEW = r"""Our
strongest negative result---on the one panel with room, Carparts, neither
the real nor the simulated learned partition collects it, and in
simulation the mixture objective is negative at every moderate
separation---identifies a target rather than hitting it."""

APP_NEW = r"""
\paragraph{Separation axis.} The grid above pools the three separations
$\{0.5,1,2\}$ of the original sweep, and the length sweep uses a single
separation of three; the envelope of Figure~\ref{fig:refinement-leverage}
therefore mixes separations. To resolve them we rerun the same generator
on a pre-registered grid: separation $\in\{0,0.25,0.5,1,2,3\}$ log-units
between adjacent group centers, $w\in\{1,0.95,0.90\}$, $T\in\{30,120\}$,
nine paired replicates per cell (seed 20260913), four groups of sixty
items, $\tau^2=0.15$, $\sigma^2=1$, occurrence spread $0.8$ in logit held
fixed. Each cell records the true partition's paired pinball gain over a
single pool, the gain of the paper's mixture objective run on the same
panel, the median size-block credibility under the single pool, and the
share of item log-size variance explained by the true labels ($R^2$,
computed from the discounted item statistics so it includes sampling
noise). Because the occurrence-rate separation is held fixed, even zero
size separation yields about $1\%$ at low leverage; that part of the
return belongs to the occurrence block. Table~\ref{tab:sepsim} summarizes
the surface by separation and Table~\ref{tab:separation} reads it at the
real panels' coordinates. Real-panel $R^2$ uses the same statistic under
the learned mixture labels at the selected discount and under the
ADI/CV$^2$ taxonomy; we do not use the learned $\hat\tau_g^2$ as a scale
because it collapses to its floor under learned partitions
(Section~\ref{sec:leverage}). The simulation was run and summarized
before the real panels were placed.

\begin{table}[t]
\centering
\caption{Separation-resolved controlled study, averaged over the six
$(w,T)$ cells at each separation: true-label $R^2$, the true partition's
gain over a single pool, the learned mixture's gain, and the median
number of components the mixture selects.}
\label{tab:sepsim}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrr}
\toprule
Separation & $R^2$ & True (\%) & Learned (\%) & $K$\\
\midrule
""" + ROWS_SIM + r"""\bottomrule
\end{tabular}
\end{table}

\begin{table*}[t]
\centering
\caption{Real panels placed on the separation-resolved surface. $w$ is the
simulated discount matching the panel's selected one; $R^2$ is the
size-block separation under the learned partition and under the taxonomy,
and ``family $R^2$'' the true-label $R^2$ of the simulated family at the
panel's $\lambda^{(+)}$ ($T{=}120$). ``Room'' is the true partition's gain
interpolated along $\lambda^{(+)}$ within the matching $w$ at each length;
``sim.\ learned'' the learned mixture's gain at the same point
($T{=}120$); ``realized'' the panel's internal refinement gain
(Figure~\ref{fig:refinement-leverage}).}
\label{tab:separation}
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lrrrrrrrrr}
\toprule
Panel & $w$ & $\lambda^{(+)}$ & $R^2$ learned & $R^2$ tax. & family $R^2$ & Room $T{=}30$ & Room $T{=}120$ & Sim.\ learned & Realized\\
\midrule
""" + ROWS_PLACE + r"""\bottomrule
\end{tabular}
\end{table*}
"""


def patch_experiments(path):
    s = path.read_text(encoding="utf-8")
    s = rep(s, SEC56_A_OLD, SEC56_A_NEW, path)
    i = s.index(SEC56_B_OLD_START); j = s.index(SEC56_B_OLD_END, i) + len(SEC56_B_OLD_END)
    old = s[i:j]
    s = s[:i] + orig_block(old) + SEC56_B_NEW + s[j:]
    s = rep(s, FIG10_OLD, FIG10_NEW, path)
    s = rep(s, SEC55_OLD, SEC55_NEW, path)
    if "\\label{fig:separation}" not in s:
        k = s.index("\\end{figure}\n", s.index("\\label{fig:refinement-leverage}")) + len("\\end{figure}\n")
        s = s[:k] + FIG + s[k:]
    path.write_text(s, encoding="utf-8")


def patch_main(path):
    s = path.read_text(encoding="utf-8")
    s = rep(s, ABS_OLD, ABS_NEW, path)
    s = rep(s, INTRO_OLD, INTRO_NEW, path)
    s = rep(s, CONTRIB_OLD, CONTRIB_NEW, path)
    s = rep(s, CONCL_OLD, CONCL_NEW, path)
    s = rep(s, LIM4_OLD, LIM4_NEW, path)
    path.write_text(s, encoding="utf-8")


def patch_appendix(path):
    s = path.read_text(encoding="utf-8")
    if "\\label{tab:sepsim}" not in s:
        anchor = "best on every metric in every cell."
        i = s.index(anchor, s.index("\\label{app:simulation-details}")) + len(anchor)
        s = s[:i] + "\n" + APP_NEW + s[i:]
    path.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    for d in DIRS:
        patch_experiments(d / "sections" / "experiments.tex")
        patch_main(d / "main.tex")
        patch_appendix(d / "sections" / "experiment_appendix.tex")
        print("patched", d.name)
