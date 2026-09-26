"""W17b: compress the pre-fit screen to one main-text paragraph (no figure),
move fig:room to the appendix, drop the abstract sentence, shorten the intro
clause. Edits w17_room_tex.py in place (so re-runs stay consistent) and then
re-applies it.
Usage: py scripts/integrity/w17b_compress.py
"""
from __future__ import annotations
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W17 = ROOT / "scripts" / "integrity" / "w17_room_tex.py"
s = W17.read_text(encoding="utf-8")

NEW_MAIN = r'''ROOM_MAIN = r"""
\paragraph{A pre-fit screen.}
The surface answers the question ex post, with true labels; the
practical question is ex ante. On an expanded controlled grid
(separation, discount, length, occurrence rate, group size, imbalance;
$2{,}700$ panels), a closed-form score computed from the single-pool fit
alone, $\mathrm{med}_i(1-\lambda_i^{(+)})^2\,\hat\tau^2$, ranks the room a
true partition could earn with held-out AUC """ + f"{auc_r1b['T']:.2f}" + r"""--""" + f"{auc_r1b['random']:.2f}" + r""" for
the event ``gain above one percent''; a logistic screen on the same
fitting-window quantities reaches """ + f"{auc['sep']:.2f}" + r"""--""" + f"{auc['random']:.2f}" + r""" when separation
levels, lengths, or random folds are held out and """ + f"{auc['w']:.2f}" + r""" when a
whole discount is held out, so the screen interpolates across regimes it
has seen and does not extrapolate across the discount
(Appendix~\ref{app:room}). Two facts bound the claim. """ + f"{100*share_collapse:.0f}" + r"""\% of the
simulated single-pool fits collapse ($\hat\tau^2=0$) and half of the
large oracle gains occur there: escapes from a collapsed global fit,
which the resolution statistic flags, not returns to structure. And on
real series with a known partition injected---group offsets of up to two
log-units on Online Retail, Carparts, and M5, at the selected discount and
at $w=0.90$---the oracle partition never earns more than """ + f"{semi_max:.1f}" + r"""\%,
because a single pool absorbs the injected heterogeneity into
$\hat\tau^2$ and the items' own evidence takes over; the screen assigns
every such panel a probability below """ + f"{semi_pmax:.2f}" + r""" of material room,
correctly, but the test checks rejections, not detections. The screen
therefore does what Corollary~\ref{cor:resolution} licenses and no more:
it identifies, from the fitting window, regimes where refinement cannot
pay materially.
"""

ROOM_FIG = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figs/fig15_room_diagnostic.pdf}
\caption{\textbf{Screening for pooling room before any partition is
fitted.} Left: the closed-form score computed from the single-pool fit
against the oracle partition's realized gain on $2{,}700$ simulated
panels, colored by discount; open markers are panels whose single-pool
variance component collapsed. Right: held-out AUC for the event that the
oracle gain exceeds one percent (or half a percent), when whole
separation levels, discounts, or lengths are held out, or under random
folds; $R_1'$ is the closed-form score and $R_2$ a logistic screen on the
same fitting-window quantities.}
\label{fig:room}
\end{figure*}
"""
'''
i = s.index('ROOM_MAIN = r"""'); j = s.index('\nROOM_APP = r"""')
s = s[:i] + NEW_MAIN + s[j:]
# figure goes to the appendix section, right after its heading paragraph
s = s.replace('ROOM_APP = r"""\n\\subsection{Pre-fit screen: protocol, metrics, and real-series transfer}\n\\label{app:room}\n',
              'ROOM_APP = r"""\n\\subsection{Pre-fit screen: protocol, metrics, and real-series transfer}\n\\label{app:room}\n""" + ROOM_FIG + r"""\n')
# abstract: drop the added sentence; intro: shorten
s = s.replace('''        ("mixture objective fails to collect it, in data and in simulation alike.\\nThe forecaster that carries the diagnostic,",
         "mixture objective fails to collect it, in data and in simulation alike.\\nA screen computed from the single-pool fit alone ranks that room with\\nheld-out AUC near $0.8$ across simulated regimes, and on real series with\\ninjected structure correctly finds none, because a single pool absorbs\\nthe heterogeneity and leaves no leverage.\\nThe forecaster that carries the diagnostic,"),''',
              '''        ("mixture objective fails to collect it, in data and in simulation alike.\\nA screen computed from the single-pool fit alone ranks that room with\\nheld-out AUC near $0.8$ across simulated regimes, and on real series with\\ninjected structure correctly finds none, because a single pool absorbs\\nthe heterogeneity and leaves no leverage.\\nThe forecaster that carries the diagnostic,",
         "mixture objective fails to collect it, in data and in simulation alike.\\nThe forecaster that carries the diagnostic,"),''')
s = s.replace('''         "there, as the same objective does at those coordinates in simulation. A\\nscreen built from the single-pool fit alone identifies the regimes where\\nrefinement cannot pay materially, with held-out AUC near $0.8$ when\\nregimes it has seen are interpolated and about $0.7$ when a forgetting\\nrate is extrapolated. Forgetting itself is a first-order decision:"),''',
              '''         "there, as the same objective does at those coordinates in simulation; a\\nscreen built from the single-pool fit alone identifies, in simulation,\\nthe regimes where refinement cannot pay materially. Forgetting itself is\\na first-order decision:"),''')
# make the intro replacement two-way: the long form (already in v6) must be replaced by the short form
s = s.replace('''    for old, new in reps:
        if new in s:
            continue
        assert s.count(old) == 1, old[:60]
        s = s.replace(old, new)''', '''    LONG_INTRO = "there, as the same objective does at those coordinates in simulation. A\\nscreen built from the single-pool fit alone identifies the regimes where\\nrefinement cannot pay materially, with held-out AUC near $0.8$ when\\nregimes it has seen are interpolated and about $0.7$ when a forgetting\\nrate is extrapolated. Forgetting itself is a first-order decision:"
    if LONG_INTRO in s:
        s = s.replace(LONG_INTRO, "there, as the same objective does at those coordinates in simulation.\\nForgetting itself is a first-order decision:")
    for old, new in reps:
        if new in s:
            continue
        assert s.count(old) == 1, old[:60]
        s = s.replace(old, new)''')
W17.write_text(s, encoding="utf-8")
print("w17 rewritten; re-applying")
runpy.run_path(str(W17), run_name="__main__")
