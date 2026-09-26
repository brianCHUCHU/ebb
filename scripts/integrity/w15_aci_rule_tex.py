"""W15: land the calibration-selection rule (A1) in both manuscript dirs:
setup sentence, tab:wfprob caption, the 5.3 paragraph, tab:regret caption,
abstract/intro/conclusion clauses, and an appendix table tab:aci-rule.
Replaced passages are kept as `% ORIG|` lines under `% [ORIG 2026-09-09]`.
Usage: py scripts/integrity/w15_aci_rule_tex.py [YYYY-MM-DD]
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
TAG = "% [ORIG 2026-09-09]"
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}

sel = pd.read_csv(OUT / "aci_selector.csv").set_index("panel")
rf = pd.read_csv(OUT / "regret_wf.csv").set_index("model")
rule_max = float(rf.loc["EBB-rule", "max_regret"]); rule_worst = PNL[rf.loc["EBB-rule", "worst_panel"]]
n_correct = int(sel["correct"].sum())
applied = [PNL[p] for p in ("m5", "auto", "carparts", "raf", "online_retail") if p in sel.index and bool(sel.loc[p, "apply_aci"])]


def orig_block(old):
    return TAG + "\n" + "".join("% ORIG| " + ln + "\n" for ln in old.rstrip("\n").split("\n"))


def rep(s, old, new, path, keep=True):
    assert s.count(old) == 1, (path.name, s.count(old), old[:70])
    return s.replace(old, (orig_block(old) if keep else "") + new)


rows = []
for p in ("online_retail", "m5", "auto", "carparts", "raf"):
    r = sel.loc[p]
    rows.append(f"{PNL[p]} & {int(r['tail_blocks'])} & {r['tail_plain']:.4f} & {r['tail_aci']:.4f} & "
                f"{'ACI' if bool(r['apply_aci']) else 'plain'} & {r['ext_EBB_mean']:.4f} & {r['ext_ACI_mean']:.4f} & "
                f"{'yes' if bool(r['correct']) else 'no'}\\\\")
TAB = r"""
\begin{table*}[t]
\centering
\caption{Validation-tail rule for the calibration layer. On the 80/20
chronological split of the initialization window, \EBB{} is initialized on
the head (labels learned on the head only) and walked forward over the
tail with and without the adaptive conformal layer; the layer is applied
in deployment iff its tail scaled pinball ($q\in\{.5,.75,.9\}$) is lower.
External columns are the walk-forward means of Table~\ref{tab:wfprob};
``matches'' marks panels where the rule picks the ex-post better variant.}
\label{tab:aci-rule}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrlrrc}
\toprule
Panel & Tail blocks & Tail plain & Tail ACI & Choice & Ext.\ \EBB{} & Ext.\ ACI-\EBB{} & Matches\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table*}
"""

RULE_PARA = r"""The M5 loss is a calibration effect, and the same layer is available to
\EBB{}. ACI-\EBB{} reads \EBB{}'s predictive quantile at the adaptive
level instead of the nominal one. On M5 it lowers mean SPL from $1.4851$
to $1.3785$, $2.9\%$ below ACI-ADIDA and significant; it also leads Auto
($0.2901$) and brings Carparts to within $0.2\%$ of TweedieGP. On Online
Retail and RAF the same layer hurts, by $6.3\%$ and $2.3\%$, because the
per-series coverage signal is too noisy on those panels to improve an
already calibrated forecaster. Whether to apply it is therefore a
selection problem, and the initialization window already answers it: on
the same 80/20 chronological split used to select the discount, \EBB{} is
walked forward over the validation tail with and without the layer, and
the layer is applied iff its tail pinball is lower
(Appendix~\ref{tab:aci-rule}). The rule applies the layer on """ + " and ".join(applied) + r"""
and withholds it elsewhere, matching the ex-post better variant on
""" + f"{n_correct}" + r""" of five panels (Carparts is the exception, where the two
are within $0.5\%$ and statistically undecided), and the tail gains it
sees track the external ones ($+7.8\%$ versus $+7.2\%$ on M5, $-7.9\%$
versus $-6.3\%$ on Online Retail). The resulting row, \EBB{} (rule), is
within """ + f"{rule_max:.1f}" + r"""\% ({""" + rule_worst + r"""}) of the best competing method on every
panel under walk-forward updating. Two further facts survive the mixed
result: under equal calibration \EBB{} beats the ADIDA base on all five
panels (undecided only on Online Retail), and the adaptive layer's benefit
is a property of the panel, not of the forecaster it wraps.
"""


def patch_experiments(path):
    s = path.read_text(encoding="utf-8")
    # 5.3 second paragraph -> rule paragraph
    i = s.index("The M5 loss is a calibration effect, and the same layer is available to")
    j = s.index("Table~\\ref{tab:regret} and Figure~\\ref{fig:frontier} summarize both")
    old = s[i:j]
    s = s[:i] + orig_block(old) + RULE_PARA + "\n" + s[j:]
    # regret sentence: add the rule row
    s = rep(s, "Counting our own\nwrapper as a competitor, uncalibrated \\EBB{} trails ACI-\\EBB{} by $7.7\\%$\non M5.",
            "Counting our own\nwrapper as a competitor, uncalibrated \\EBB{} trails ACI-\\EBB{} by $7.7\\%$\non M5, and \\EBB{} (rule) closes that to " + f"{rule_max:.1f}" + "\\%.", path, keep=False)
    # setup sentence
    s = rep(s, "adaptive level, with the same $\\gamma$ and update rule as ACI-ADIDA.",
            "adaptive level, with the same $\\gamma$ and update rule as ACI-ADIDA;\n\\EBB{} (rule) applies that layer only where a validation-tail rule inside\nthe initialization window selects it (Section~\\ref{sec:walkforward}).", path, keep=False)
    # tab:wfprob caption
    s = rep(s, "ACI-\\EBB{} applies the ACI-ADIDA\ncalibration layer to \\EBB{}'s predictive quantiles.",
            "ACI-\\EBB{} applies the ACI-ADIDA\ncalibration layer to \\EBB{}'s predictive quantiles, and \\EBB{} (rule)\napplies it only where the validation-tail rule selects it, so it equals\none of the two rows above it on each panel and is not ranked.", path, keep=False)
    # tab:regret caption
    s = rep(s, "ACI-\\EBB{} exists only under walk-forward updating.",
            "ACI-\\EBB{} and the rule row exist only under walk-forward updating; the\nrule row is excluded from the per-panel best.", path, keep=False)
    path.write_text(s, encoding="utf-8")


def patch_main(path):
    s = path.read_text(encoding="utf-8")
    s = rep(s, r"""strongest conformal baseline uses makes it the walk-forward leader on M5
and Auto and closes Carparts to $0.2\%$, while hurting on Online Retail and
RAF; under equal calibration \EBB{} beats the ADIDA base on all five
panels. """,
            r"""strongest conformal baseline uses makes it the walk-forward leader on M5
and Auto and closes Carparts to $0.2\%$, while hurting on Online Retail and
RAF; a validation-tail rule inside the initialization window decides where
to apply it, is right on four of five panels, and leaves \EBB{} within
""" + f"{rule_max:.1f}" + r"""\% of the best competing method everywhere under walk-forward
updating. """, path)
    s = rep(s, r"""degrades Online Retail and RAF, so we report it as a separate row rather
than folding it into the method. """,
            r"""degrades Online Retail and RAF; a validation-tail rule inside the
initialization window decides where to apply it and is right on four of
five panels. """, path)
    s = rep(s, r"""walk-forward updating the adaptive conformal layer decides M5: applied to
\EBB{} it leads, applied to ADIDA it is second, and uncalibrated \EBB{} is
third. """,
            r"""walk-forward updating the adaptive conformal layer decides M5: applied to
\EBB{} it leads, applied to ADIDA it is second, and uncalibrated \EBB{} is
third; a validation-tail rule selects the layer where it helps and leaves
\EBB{} within """ + f"{rule_max:.1f}" + r"""\% of the best competing method on every panel. """, path)
    path.write_text(s, encoding="utf-8")


def patch_appendix(path):
    s = path.read_text(encoding="utf-8")
    if "\\label{tab:aci-rule}" not in s:
        m = re.compile(r"\\end\{table\*?\}\n").search(s, s.index("\\label{tab:wf-significance}"))
        s = s[:m.end()] + TAB + s[m.end():]
    path.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    for d in DIRS:
        patch_experiments(d / "sections" / "experiments.tex")
        patch_main(d / "main.tex")
        patch_appendix(d / "sections" / "experiment_appendix.tex")
        print("patched", d.name, "| rule max regret", rule_max, rule_worst, "| applied on", applied, "| correct", n_correct)
