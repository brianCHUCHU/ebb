"""W17: land the pre-fit pooling-room screen in v6 (and keep it through builder
re-runs): writes sections/room_main.tex (paragraph + fig:room) and
sections/room_app.tex (protocol, tab:room, fig:room-transfer), inserts
\\input lines into experiments_v6.tex and demoted.tex, and patches the
abstract / intro / contributions / limitations sentences in main.tex.
Usage: py scripts/integrity/w17_room_tex.py [YYYY-MM-DD]
"""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
V6 = ROOT / "paper_v2" / "v6_aistats"
ROWS = (OUT / "tab_room.tex").read_text(encoding="utf-8")
met = pd.read_csv(OUT / "room_eval_metrics.csv")
auc = {f: met[met["family"] == f]["auc_R2_1.0"].mean() for f in ("sep", "w", "T", "random")}
auc_r1b = {f: met[met["family"] == f]["auc_R1b_1.0"].mean() for f in ("sep", "w", "T", "random")}
syn = pd.read_csv(OUT / "room_synthetic.csv")
share_collapse = float((syn["z_size"] <= 0).mean()); share_collapse_pos = float((syn[syn["delta_oracle"] > 1]["z_size"] <= 0).mean())
tr = pd.read_csv(OUT / "room_transfer_summary.csv")
semi_max = float(tr["delta_max"].max()); semi_pmax = float(tr["p_R2_1.0_max"].max())

ROOM_MAIN = r"""
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

ROOM_APP = r"""
\subsection{Pre-fit screen: protocol, metrics, and real-series transfer}
\label{app:room}
""" + ROOM_FIG + r"""

\paragraph{Protocol.} The grid crosses separation $\{0,.25,.5,1,2\}$,
discount $\{1,.95,.90\}$, length $\{30,60,120\}$, occurrence mean
$\{.1,.25,.5\}$, items per group $\{20,60\}$, and balanced versus
$70/10/10/10$ group sizes, with four groups, $\tau^2=0.15$, $\sigma^2=1$,
occurrence separation scaled as $0.8\min(\mathrm{sep},1)$, five
replicates per cell, and seed 20260915. Per panel, only the single-pool
fit is used for features: median retained leverage on the size and
occurrence blocks, $\hat\tau^2$, the Beta concentration, the resolution
statistic, effective counts, and panel size. The target is the oracle
partition's paired pinball gain over the single pool (estimated
hyperparameters, true labels), which can be negative. The pre-registered
closed-form score $R_1$ used an unweighted mean of sampling variances
that is dominated by items whose discounted positive count approaches
zero; it has no screening power (AUC $0.45$--$0.64$) and is retained in
the released artifact. $R_1'$ replaces that term by the pool's own
weighted estimate $\hat\tau^2$; $R_2$ is a logistic regression on
$\log(1+\cdot)$ of the features, fitted on the training folds only. The
design, the amendments, and their order are recorded in
\texttt{docs/DESIGN\_room\_diagnostic.md}.

\begin{table}[t]
\centering
\caption{Held-out screening performance on the simulated grid for the
event that the oracle partition's gain exceeds one percent (base rate
$22\%$). Columns: AUC of the pre-registered $R_1$, of $R_1'$, and of the
logistic screen $R_2$ on all panels; the same for $R_1'$ and $R_2$ on the
panels whose single-pool variance component did not collapse; precision
and recall of $R_2$ at its best-$F_1$ threshold.}
\label{tab:room}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{All panels} & \multicolumn{2}{c}{Resolved} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Held out & $R_1$ & $R_1'$ & $R_2$ & $R_1'$ & $R_2$ & $R_2$ P / R\\
\midrule
""" + ROWS + r"""\bottomrule
\end{tabular}
\end{table}

\paragraph{Real-series transfer.} Online Retail, Carparts, and the
5{,}000-series M5 sample keep their zeros, noise, lengths, and scales; a
known partition is injected by multiplying every positive value of item
$i$ by $\exp(\delta_{g(i)})$ with symmetric offsets
$\{\pm0.5,\pm1.5\}\times\delta$, $\delta\in\{0,.25,.5,1,2\}$, over four
randomly assigned groups (three assignments), at the selected discount and
again at $w=0.90$. The oracle gain of the injected partition never exceeds
""" + f"{semi_max:.1f}" + r"""\% on any panel: the single pool absorbs the injected
heterogeneity ($\hat\tau^2$ rises from $0.04$ to $5.1$ on Carparts as
$\delta$ goes from $0$ to $2$) and the median size-block credibility
rises toward one, so the items' own evidence, not the prior, carries the
forecast. $R_2$, fitted on the simulated grid and applied unchanged,
assigns every injected panel a probability below """ + f"{semi_pmax:.2f}" + r""" of material
room (Figure~\ref{fig:room-transfer}). The transfer test therefore checks
the screen's rejections, not its detections; a real-series test of
detections would require panels whose items lack positive observations,
which is the short-history regime of Section~\ref{sec:pays}.

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{figs/fig16_room_transfer.pdf}
\caption{Real-series transfer of the screen: predicted probability of
material room (from the simulated grid) against the realized oracle gain
of the injected partition, by panel and discount.}
\label{fig:room-transfer}
\end{figure}
"""


def main():
    (V6 / "sections" / "room_main.tex").write_text(ROOM_MAIN, encoding="utf-8")
    (V6 / "sections" / "room_app.tex").write_text(ROOM_APP, encoding="utf-8")
    # experiments_v6: input after the "Forgetting is an order of magnitude larger" paragraph
    p = V6 / "sections" / "experiments_v6.tex"; s = p.read_text(encoding="utf-8")
    if "\\input{sections/room_main}" not in s:
        i = s.index("\\paragraph{Forgetting is an order of magnitude larger.}")
        j = s.index("\n\n", i) + 2
        s = s[:j] + "\\input{sections/room_main}\n\n" + s[j:]
        p.write_text(s, encoding="utf-8")
    p = V6 / "sections" / "demoted.tex"; s = p.read_text(encoding="utf-8")
    if "\\input{sections/room_app}" not in s:
        s = s.rstrip("\n") + "\n\n\\input{sections/room_app}\n"
        p.write_text(s, encoding="utf-8")
    # main.tex prose
    p = V6 / "main.tex"; s = p.read_text(encoding="utf-8")
    reps = [
        ("mixture objective fails to collect it, in data and in simulation alike.\nA screen computed from the single-pool fit alone ranks that room with\nheld-out AUC near $0.8$ across simulated regimes, and on real series with\ninjected structure correctly finds none, because a single pool absorbs\nthe heterogeneity and leaves no leverage.\nThe forecaster that carries the diagnostic,",
         "mixture objective fails to collect it, in data and in simulation alike.\nThe forecaster that carries the diagnostic,"),
        ("there, as the same objective does at those coordinates in simulation.\nForgetting itself is a first-order decision:",
         "there, as the same objective does at those coordinates in simulation; a\nscreen built from the single-pool fit alone identifies, in simulation,\nthe regimes where refinement cannot pay materially. Forgetting itself is\na first-order decision:"),
        ("A controlled surface over\nleverage and separation calibrates the bound, and five real panels\nconfirm it in both directions (Section~\\ref{sec:pays}).",
         "A controlled surface over\nleverage and separation calibrates the bound, a pre-fit screen built from\nthe single-pool fit identifies regimes without material room, and five\nreal panels confirm the diagnostic in both directions\n(Section~\\ref{sec:pays})."),
        ("Its calibration on the controlled surface\nis simulation-based, and the room it assigns to a real panel is read\nwithin the panel's forgetting regime with an interpolation across two\nseries lengths.",
         "Its calibration on the controlled surface\nis simulation-based, and the room it assigns to a real panel is read\nwithin the panel's forgetting regime with an interpolation across two\nseries lengths. The pre-fit screen interpolates across regimes it has\nseen and does not extrapolate across the discount, and its real-series\ntransfer test could check only rejections, since an injected partition\nnever earns material room once a single pool has absorbed the\nheterogeneity."),
    ]
    LONG_INTRO = "there, as the same objective does at those coordinates in simulation. A\nscreen built from the single-pool fit alone identifies the regimes where\nrefinement cannot pay materially, with held-out AUC near $0.8$ when\nregimes it has seen are interpolated and about $0.7$ when a forgetting\nrate is extrapolated. Forgetting itself is a first-order decision:"
    if LONG_INTRO in s:
        s = s.replace(LONG_INTRO, "there, as the same objective does at those coordinates in simulation.\nForgetting itself is a first-order decision:")
    for old, new in reps:
        if new in s:
            continue
        assert s.count(old) == 1, old[:60]
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    # keep through builder re-runs
    b = ROOT / "scripts" / "integrity" / "v6_build.py"; bs = b.read_text(encoding="utf-8")
    if "room_main" not in bs:
        bs = bs.replace('""" + forget_para.rstrip("\\n") + r"""\n\n% =========================================================================\n\\subsection{Accuracy and cost}',
                        '""" + forget_para.rstrip("\\n") + r"""\n\n\\input{sections/room_main}\n\n% =========================================================================\n\\subsection{Accuracy and cost}')
        bs = bs.replace('lev_app.replace("\\\\subsection{When can pooling pay?}\\\\label{sec:leverage}", "\\\\label{sec:leverage}") + "\\n"',
                        'lev_app.replace("\\\\subsection{When can pooling pay?}\\\\label{sec:leverage}", "\\\\label{sec:leverage}") + "\\n\\\\input{sections/room_app}\\n"')
        b.write_text(bs, encoding="utf-8")
    print("patched v6 | AUC R2", {k: round(v, 3) for k, v in auc.items()}, "| R1'", {k: round(v, 3) for k, v in auc_r1b.items()},
          "| collapse share", round(share_collapse, 3), round(share_collapse_pos, 3), "| semi max", round(semi_max, 2), round(semi_pmax, 3))


if __name__ == "__main__":
    main()
