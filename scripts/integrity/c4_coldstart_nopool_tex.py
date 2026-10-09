"""C4: LaTeX table for the within-model no-pooling comparison (tab:coldstart-nopool).

Reads outputs/2026-09-27/coldstart_nopool_wide.csv (c3_coldstart_nopool.py v2: prior on/off,
total gain, advantage over TweedieGP) and outputs/2026-09-28/coldstart_blocks_wide.csv
(c5_coldstart_blocks.py: one block's prior switched off at a time), writes
paper_v2/v7_paper/sections/tab_coldstart_nopool.tex (full table environment).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "2026-09-27"
BLK = ROOT / "outputs" / "2026-09-28" / "coldstart_blocks_wide.csv"
OUT = ROOT / "paper_v2" / "v7_paper" / "sections" / "tab_coldstart_nopool.tex"
ORDER = ["online_retail", "m5", "auto", "carparts", "raf"]
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}


def main() -> None:
    m = pd.read_csv(SRC / "coldstart_nopool_wide.csv")
    m["L"] = m["L"].astype(str)
    b = pd.read_csv(BLK)
    b["L"] = b["L"].astype(str)
    m = m.merge(b[["panel", "L", "spl_both", "spl_none", "occ_effect_mean_pct", "size_effect_mean_pct",
                   "total_mean_pct"]], on=["panel", "L"], how="left")
    # the two runs fit the same arms; they must agree on the shared quantities
    chk = (m["spl_both"] - m["spl_mean_EBB-B0"]).abs().max(), (m["spl_none"] - m["spl_mean_EBB-nopool"]).abs().max()
    assert max(chk) < 5e-4, f"c3 and c5 disagree on shared arms: {chk}"
    assert m["occ_effect_mean_pct"].notna().all(), "block decomposition missing for some rows"
    lines = []
    for i, p in enumerate(ORDER):
        sub = m[m["panel"] == p].sort_values("median_len")
        for j, r in enumerate(sub.to_dict("records")):
            name = PNL[p] if j == 0 else ""
            lines.append(
                f"{name} & {r['L']} & {r['median_lambda_occ']:.3f} & {r['median_lambda_size']:.3f} & "
                f"{r['spl_mean_EBB-B0']:.4f} & {r['spl_mean_EBB-nopool']:.4f} & "
                f"{r['pool_gain_mean_pct']:+.1f} & {r['occ_effect_mean_pct']:+.1f} & "
                f"{r['size_effect_mean_pct']:+.1f} & {r['pool_gain_q90_pct']:+.1f} & "
                f"{r['adv_vs_TweedieGP']:+.1f}\\\\"
            )
        if i < len(ORDER) - 1:
            lines.append("\\addlinespace")
    head = r"""\begin{table*}[!htbp]
\centering
\caption{Within-model value of the shared prior under history truncation.
Both arms use the audited structure and discount, labels relearned on the
truncated window, and no bootstrap ensemble; ``prior off'' scales the
prior strength $(\alpha_g,\beta_g,\kappa_g)$ by $10^{-6}$, so every item
with data keeps $\lambda_i=1$ and only items without a positive
observation fall back to the group mean. Gain is the percentage by which
switching the prior off raises the loss (positive: pooling helps):
``both'' switches off both blocks, ``occ.''\ and ``size'' one block at a
time with the other left on, so the two need not sum to the total. The
last column repeats \EBB{}'s advantage over TweedieGP from
Table~\ref{tab:coldstart}. On RAF at $L=12$ the mixture labels are
relearned on twelve observations and the comparison is unstable.}
\label{tab:coldstart-nopool}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrrrrr}
\toprule
 & & & & \multicolumn{2}{c}{mean SPL} & \multicolumn{3}{c}{gain from prior, mean SPL (\%)} & q90 (\%) & vs.\ TweedieGP\\
\cmidrule(lr){5-6}\cmidrule(lr){7-9}
Panel & $L$ & $\lambda^{(o)}$ & $\lambda^{(+)}$ & prior on & prior off & both & occ. & size & both & (\%)\\
\midrule
"""
    tail = r"""\bottomrule
\end{tabular}
\end{table*}
"""
    OUT.write_text(head + "\n".join(lines) + "\n" + tail, encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
