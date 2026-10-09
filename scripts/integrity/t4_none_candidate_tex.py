"""T4: LaTeX table for the no-pooling candidate robustness analysis (tab:none-candidate).

Inputs : outputs/2026-09-29/selected_pairs_with_none.csv   (t2_selection_with_none.py)
         outputs/2026-09-27/coldstart_nopool_wide.csv       (c3, full-length rows: external, B=0)
         outputs/2026-09-29/wf_or_none.csv                  (t3, Online Retail walk-forward)
Output : paper_v2/v7_paper/sections/tab_none_candidate.tex (full table environment)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_v2" / "v7_paper" / "sections" / "tab_none_candidate.tex"
ORDER = ["online_retail", "m5", "auto", "carparts", "raf"]
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}


def fw(w):
    w = float(w)
    return f"{w:.2f}" if abs(round(w, 2) - w) < 1e-12 else f"{w:g}"


def cfg(s, w):
    return f"({s}, {fw(w)})"


def main() -> None:
    sel = pd.read_csv(ROOT / "outputs" / "2026-09-29" / "selected_pairs_with_none.csv").set_index("panel")
    ext = pd.read_csv(ROOT / "outputs" / "2026-09-27" / "coldstart_nopool_wide.csv")
    ext = ext[ext["L"].astype(str) == "full"].set_index("panel")
    wf = pd.read_csv(ROOT / "outputs" / "2026-09-29" / "wf_or_none.csv").set_index("arm")
    lines = []
    for p in ORDER:
        s, e = sel.loc[p], ext.loc[p]
        ext_pct = (e["spl_mean_EBB-nopool"] / e["spl_mean_EBB-B0"] - 1) * 100
        lines.append(
            f"{PNL[p]} & {cfg(s['audited_structure'], s['audited_discount'])} & "
            f"{cfg(s['selected_structure'], s['selected_discount'])} & {fw(s['best_none_discount'])} & "
            f"{int(s['best_none_rank_of_32'])} & {s['none_vs_audited_pct']:+.2f} & {ext_pct:+.2f}\\\\"
        )
    o = ext.loc["online_retail"]
    head = (
        "\\begin{table*}[!htbp]\n\\centering\n"
        "\\caption{A no-pooling candidate in the selection family. The family is extended from 24 to\n"
        "32 candidates by adding \\emph{none} (the model of Table~\\ref{tab:coldstart-nopool} with\n"
        "its prior switched off) at each of the eight discounts, scored on the same validation tail\n"
        "by the same criterion. ``Validation'' is the best \\emph{none} candidate's validation loss\n"
        "relative to the 24-candidate winner, ``External'' the full-length mean SPL of the model\n"
        "with its prior off relative to the prior on (both without the bootstrap ensemble);\n"
        "negative favors \\emph{none}. The selection changes on Online Retail only, where\n"
        f"(none, 0.95) gives ${o['spl_mean_EBB-nopool']:.4f}$ at fixed origin and\n"
        f"${wf.loc['none_0.95', 'spl_mean']:.4f}$ under walk-forward against\n"
        f"${o['spl_mean_EBB-B0']:.4f}$ and ${wf.loc['global_0.95', 'spl_mean']:.4f}$ for the\n"
        "configuration used in the paper. On Carparts and M5 the validation tail and the external\n"
        "evaluation disagree in sign. Headline results use the 24-candidate family.}\n"
        "\\label{tab:none-candidate}\n\\small\n\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{lllrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Selected configuration} & \\multicolumn{2}{c}{Best \\emph{none}}"
        " & \\multicolumn{2}{c}{\\emph{none} vs.\\ pooled (\\%)}\\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Panel & 24 candidates & 32 candidates & $w$ & rank of 32 & Validation & External\\\\\n\\midrule\n"
    )
    tail = "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    OUT.write_text(head + "\n".join(lines) + "\n" + tail, encoding="utf-8")
    print(head + "\n".join(lines) + "\n" + tail)


if __name__ == "__main__":
    main()
