"""W5: patch tab:wfprob (five panels) and the appendix full walk-forward tables
in both manuscript directories from the W4 LaTeX rows. Prose is not rewritten;
stale sentences get a `% [2026-09-06 STALE]` comment with the new numbers.
Usage: py scripts/integrity/w5_patch_wf_tex.py [YYYY-MM-DD]
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
PANELS = ["online_retail", "carparts", "auto", "raf", "m5"]
PNAME = {"online_retail": "Online Retail", "carparts": "Carparts", "auto": "Auto", "raf": "RAF", "m5": "M5"}
PLABEL = {"online_retail": "tab:wfprob-full", "carparts": "tab:wf-carparts-full",
          "auto": "tab:wf-auto-full", "raf": "tab:wf-raf-full", "m5": "tab:wf-m5-full"}
REFIT = {
    "online_retail": r"CP and ACI wrappers refit each of the 36 seven-day blocks; AutoARIMA, AutoTheta, iETS, and TweedieGP refit every four blocks; \EBB{} updates sufficient statistics each block.",
    "carparts": r"Every baseline refits each of the six monthly blocks; \EBB{} updates sufficient statistics.",
    "auto": r"Every baseline refits each of the six monthly blocks; \EBB{} updates sufficient statistics.",
    "raf": r"Every baseline refits each of the twelve monthly blocks; \EBB{} updates sufficient statistics.",
    "m5": r"Static CP wrappers and ACI-ADIDA refit each of the 93 seven-day blocks on the 5{,}000-series sample; AutoARIMA, AutoTheta, iETS, and TweedieGP were not run under this protocol on cost grounds; \EBB{} updates sufficient statistics each block.",
}

MAIN_HEAD = r"""\begin{table*}[t]
\centering
\caption{\textbf{Strict probabilistic walk-forward evaluation on all five
panels.} Online Retail and M5 use seven-day blocks; Auto, Carparts, and
RAF use monthly blocks. Mean and q90 are scaled pinball losses; all five
quantiles, coverage, and interval width are reported in
Appendix~\ref{app:walkforward}. Classical wrappers and ACI refit each
block. AutoARIMA, AutoTheta, iETS, and TweedieGP refit every four blocks
on Online Retail and every block on the monthly panels; on M5 these four
were not run under this protocol on cost grounds (dashes). \EBB{} updates
sufficient statistics each block. Bold and underline rank non-degenerate
methods by SPL; on RAF the all-zero forecast is marginally lower than every
method through q90. Adaptive conformal inference leads on M5, the longest
daily panel.}
\label{tab:wfprob}
\footnotesize
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{lrrrrrrrrrr}
\toprule
 & \multicolumn{2}{c}{Online Retail} & \multicolumn{2}{c}{Carparts} & \multicolumn{2}{c}{Auto} & \multicolumn{2}{c}{RAF} & \multicolumn{2}{c}{M5}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Model & Mean & q90 & Mean & q90 & Mean & q90 & Mean & q90 & Mean & q90\\
\midrule
"""
MAIN_TAIL = "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"

STALE_MAIN = r"""% [2026-09-06 STALE] tab:wfprob is now five panels and includes TweedieGP. Walk-forward
% mean SPL (non-degenerate ranking): OR TweedieGP 0.8526 < EBB 0.8540 (-0.16%), ACI-ADIDA 0.9108
% (EBB +6.2% over ACI still holds; q90 TweedieGP 1.3791 < EBB 1.4002, so "q90 by 5.6% over AutoTheta"
% is no longer against the strongest comparator); Carparts TweedieGP 0.3121 < EBB 0.3145 (-0.8%),
% ACI-ADIDA 0.3417 (EBB +8.0%), CP-IMAPA 0.3492 (+9.9% is now versus the third-best); EBB no longer
% wins all five Carparts quantiles (TweedieGP takes q90 0.4326 and the mean; EBB keeps q10-q75).
% Auto EBB 0.2910 < TweedieGP 0.2924 (+0.5%), AutoARIMA 0.3139 (+7.3%). RAF EBB 0.2681 < TweedieGP
% 0.2801 (+4.3%), ACI 0.3085; Zero 0.2669 is below every method. M5 ACI-ADIDA 1.4192 < EBB 1.4851
% (-4.4%); EBB beats every static CP (CP-IMAPA 1.5457, +3.9%) and Zero. The 21% ACI-over-static gain
% (1.1576 -> 0.9108) is unchanged. Sentence on ACI leading on M5 is in the caption (author compromise).
"""
STALE_APP = r"""% [2026-09-06 STALE] With TweedieGP in the walk-forward roster this paragraph no longer holds on
% Online Retail (TweedieGP takes the mean 0.8526 vs 0.8540 and q90 1.3791 vs 1.4002; EBB keeps q75)
% or Carparts (TweedieGP takes the mean and q90; EBB keeps q10-q75). Auto/RAF/M5 tables are new.
"""

FULL_HEAD = r"""\begin{table*}[t]
\centering
\caption{Full %s walk-forward SPL at all five quantiles and on average,
with marginal (Cov@80) and positive-demand (Cov$^{+}$@80) coverage and
average interval width (AIW). %s The zero row is excluded from ranking.}
\label{%s}
\small
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{lrrrrrrrrr}
\toprule
Model & q10 & q25 & q50 & q75 & q90 & Mean & Cov@80 & Cov$^{+}$@80 & AIW\\
\midrule
"""


def patch_experiments(path: Path) -> None:
    s = path.read_text(encoding="utf-8")
    pat = re.compile(
        r"\\begin\{table\*\}\[t\]\n\\centering\n\\caption\{\\textbf\{Strict probabilistic walk-forward evaluation\.\}"
        r".*?\\end\{table\*\}\n", re.S)
    assert len(pat.findall(s)) == 1, path
    rows = (OUT / "tab_wfprob_main.tex").read_text(encoding="utf-8")
    s = pat.sub(lambda m: MAIN_HEAD + rows + MAIN_TAIL, s)
    anchor = "Table~\\ref{tab:wfprob} is the deployment-facing test."
    assert anchor in s, path
    if STALE_MAIN not in s:
        s = s.replace(anchor, STALE_MAIN + anchor)
    path.write_text(s, encoding="utf-8")


def patch_appendix(path: Path) -> None:
    s = path.read_text(encoding="utf-8")
    start = s.index("\\begin{table*}[t]\n\\centering\n\\caption{Full Online Retail probabilistic walk-forward results.")
    end = s.index("\\end{table*}\n", s.index("\\label{tab:wf-carparts-full}")) + len("\\end{table*}\n")
    blocks = []
    for p in PANELS:
        rows = (OUT / f"tab_wf_full_{p}.tex").read_text(encoding="utf-8")
        blocks.append(FULL_HEAD % (PNAME[p], REFIT[p], PLABEL[p]) + rows + MAIN_TAIL)
    s = s[:start] + "\n".join(blocks) + s[end:]
    anchor = "\\EBB{} applies the selected discount once to the initialization"
    assert anchor in s, path
    if STALE_APP not in s:
        s = s.replace(anchor, STALE_APP + anchor)
    path.write_text(s, encoding="utf-8")


def patch_main(path: Path) -> None:
    s = path.read_text(encoding="utf-8")
    marks = {
        "walk-forward updating it improves on adaptive conformal inference by":
            "% [2026-09-06 STALE] wf five panels: EBB best on Auto/RAF only; TweedieGP ahead on OR (0.8526 vs 0.8540) and Carparts (0.3121 vs 0.3145); ACI ahead on M5 (1.4192 vs 1.4851). +6.2% over ACI on OR still true; +9.9% on Carparts is now vs third-best (TweedieGP -0.8%).\n",
        "walk-forward updating the margin is $6.2\\%$ over adaptive conformal":
            "% [2026-09-06 STALE] see tab:wfprob comment: TweedieGP leads OR/Carparts wf; ACI leads M5; EBB leads Auto (+0.5%) and RAF (+4.3%).\n",
        "and it leads under strict walk-forward updating on both a long daily and a":
            "% [2026-09-06 STALE] under walk-forward EBB leads on Auto and RAF only (TweedieGP: OR, Carparts; ACI: M5).\n",
    }
    for anchor, comment in marks.items():
        i = s.find(anchor)
        assert i >= 0, (path, anchor)
        ls = s.rfind("\n", 0, i) + 1
        if comment not in s:
            s = s[:ls] + comment + s[ls:]
    path.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    for d in DIRS:
        patch_experiments(d / "sections" / "experiments.tex")
        patch_appendix(d / "sections" / "experiment_appendix.tex")
        patch_main(d / "main.tex")
        print("patched", d)
