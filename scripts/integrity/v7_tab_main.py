"""v7 Table 1: full main table (mean SPL, fixed-origin + walk-forward, all methods).

Inputs : outputs/integrity_2026-09-05/rescored_tab_prob.csv   (fixed-origin, spl_mean)
         outputs/2026-09-09/wf_all_panels_wide.csv           (walk-forward, mean)
         walk-forward wall-clock from tab:wfcost (experiment_appendix.tex), hard-coded below
Output : paper_v2/v7_paper/sections/tab_main.tex  (table body only; wrapped by experiments_v6.tex)

Ranking: bold = best, underline = second best per column among ranked rows.
Zero is a reference row and is not ranked. Max shortfall = largest percentage
gap to the best ranked value over the panels the method was run on.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_v2" / "v7_paper" / "sections" / "tab_main.tex"
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]

# (display name, fixed key, wf key, ranked?, cost string)
ROWS = [
    ("group", "Conformal wrappers on classical forecasters"),
    ("CP-Croston", "CP-Croston", "CP-Croston", True, "$\\approx$\\,CP-ADIDA"),
    ("CP-SBA", "CP-SBA", "CP-SBA", True, "$\\approx$\\,CP-ADIDA"),
    ("CP-TSB", "CP-TSB", "CP-TSB", True, "$\\approx$\\,CP-ADIDA"),
    ("CP-ADIDA", "CP-ADIDA", "CP-ADIDA", True, "4\\,s--7.8\\,min"),
    ("CP-IMAPA", "CP-IMAPA", "CP-IMAPA", True, "$\\approx$\\,CP-ADIDA"),
    ("ACI-ADIDA", None, "ACI-ADIDA", True, "7\\,s--9.7\\,min"),
    ("group", "Model-based forecasters"),
    ("AutoARIMA", "AutoARIMA", "AutoARIMA", True, "13\\,min--2.7\\,h"),
    ("AutoTheta", "AutoTheta", "AutoTheta", True, "13\\,min--2.7\\,h"),
    ("iETS", "iETS", "iETS", True, "3.6--30\\,min"),
    ("TweedieGP", "TweedieGP", "TweedieGP", True, "36\\,min--2.4\\,h$^{\\ddagger}$"),
    ("group", "Reference (unranked)"),
    ("All-zero forecast", "Zero", "Zero", False, "---"),
    ("group", "Ours"),
    ("\\EBB{}", "EB-Hurdle", "EBB", True, "3--37\\,s"),
    ("\\EBB{} (rule)$^{\\dagger}$", None, "EBB-rule", True, "3\\,s--3.6\\,min"),
]


def main() -> None:
    fx = pd.read_csv(ROOT / "outputs" / "integrity_2026-09-05" / "rescored_tab_prob.csv")
    fx = fx.set_index(["panel", "model"])["spl_mean"]
    wf = pd.read_csv(ROOT / "outputs" / "2026-09-09" / "wf_all_panels_wide.csv")
    wf = wf.set_index(["panel", "model"])["mean"]

    def get(series, panel, key):
        if key is None:
            return None
        try:
            return float(series.loc[(panel, key)])
        except KeyError:
            return None

    data = []  # rows with values
    for r in ROWS:
        if r[0] == "group":
            data.append(r)
            continue
        name, kf, kw, ranked, cost = r
        vf = [get(fx, p, kf) for p in PANELS]
        vw = [get(wf, p, kw) for p in PANELS]
        data.append((name, vf, vw, ranked, cost))

    # best / second best per column among ranked rows
    def rank(col_vals):
        vals = sorted({round(v, 4) for v in col_vals if v is not None})
        best = vals[0] if vals else None
        second = vals[1] if len(vals) > 1 else None
        return best, second

    ranked_rows = [d for d in data if d[0] != "group" and d[3]]
    best_f, sec_f, best_w, sec_w = [], [], [], []
    for j in range(5):
        b, s = rank([d[1][j] for d in ranked_rows]); best_f.append(b); sec_f.append(s)
        b, s = rank([d[2][j] for d in ranked_rows]); best_w.append(b); sec_w.append(s)

    # Statistical ties with the column best (paired t-test on per-series SPL,
    # Holm-adjusted, 5%). Sources: outputs/integrity_2026-08-07/spl_significance_corrected.csv
    # and outputs/2026-09-05/SUMMARY.md (fixed origin; EBB vs TweedieGP undecided on
    # OR p=0.71 and Carparts, significant on M5/Auto/RAF), and
    # outputs/2026-09-07/spl_significance_wf.csv (walk-forward; EBB vs TweedieGP
    # undecided on OR/Carparts/Auto, significant on RAF; ACI-EBB vs ACI-ADIDA
    # significant on M5). EBB (rule) equals EBB where the rule withholds ACI
    # (OR, Carparts, RAF) and ACI-EBB where it applies it (M5, Auto); it is
    # not tested against EBB on Auto (0.3% apart) and is treated as tied.
    # Every other method is >=15% from the column best and significantly worse.
    TIES_F = {  # panel index -> display names tied with the best
        0: {"TweedieGP", "\\EBB{}"},            # OR
        1: {"TweedieGP"},                        # M5
        2: {"\\EBB{}"},                          # Auto
        3: {"TweedieGP", "\\EBB{}"},            # Carparts
        4: {"\\EBB{}"},                          # RAF
    }
    RULE = "\\EBB{} (rule)$^{\\dagger}$"
    TIES_W = {
        0: {"TweedieGP", "\\EBB{}", RULE},      # OR
        1: {RULE},                               # M5
        2: {RULE, "\\EBB{}", "TweedieGP"},      # Auto
        3: {"TweedieGP", "\\EBB{}", RULE},      # Carparts
        4: {"\\EBB{}", RULE},                    # RAF
    }

    def fmt(v, best, ranked, name, ties):
        if v is None:
            return "---"
        s = f"{v:.3f}"
        if not ranked:
            return s
        if round(v, 4) == best:
            s = f"\\secondbest{{{s}}}"
        if name in ties:
            s = f"\\best{{{s}}}"
        return s

    # shortfall is measured against the best *competing* method: for our rows the
    # competitor set excludes our own variants (same convention as tab:regret).
    OURS = {"\\EBB{}", "\\EBB{} (rule)$^{\\dagger}$"}
    comp_rows = [d for d in ranked_rows if d[0] not in OURS]
    comp_f = [rank([d[1][j] for d in comp_rows])[0] for j in range(5)]
    comp_w = [rank([d[2][j] for d in comp_rows])[0] for j in range(5)]

    def shortfall(vals, bests):
        gaps = [(v / b - 1) * 100 for v, b in zip(vals, bests) if v is not None and b is not None]
        return f"{max(gaps):.1f}" if gaps else "---"

    SAME_AS_FIXED = {"ACI-ADIDA": "CP-ADIDA", "\\EBB{} (rule)$^{\\dagger}$": "\\EBB{}"}
    lines = []
    for d in data:
        if d[0] == "group":
            lines.append(f"\\multicolumn{{14}}{{l}}{{\\emph{{{d[1]}}}}}\\\\")
            continue
        name, vf, vw, ranked, cost = d
        cf = " & ".join(fmt(v, b, ranked, name, TIES_F[j]) for j, (v, b) in enumerate(zip(vf, best_f)))
        cw = " & ".join(fmt(v, b, ranked, name, TIES_W[j]) for j, (v, b) in enumerate(zip(vw, best_w)))
        bf, bw = (comp_f, comp_w) if name in OURS else (best_f, best_w)
        sf = shortfall(vf, bf) if ranked else "---"
        sw = shortfall(vw, bw) if ranked else "---"
        # Rows that exist only under walk-forward coincide, at fixed origin, with their base
        # method (no sequential update, so the adaptive layer is never applied): one spanning
        # cell instead of six dashes.
        if all(v is None for v in vf) and name in SAME_AS_FIXED:
            fixed_block = f"\\multicolumn{{6}}{{c}}{{= {SAME_AS_FIXED[name]}}}"
            lines.append(f"{name} & {fixed_block} & {cw} & {sw} & {cost}\\\\")
            continue
        lines.append(f"{name} & {cf} & {sf} & {cw} & {sw} & {cost}\\\\")
    head = [
        "\\begin{tabular}{lrrrrrrrrrrrrl}",
        "\\toprule",
        " & \\multicolumn{6}{c}{Fixed origin} & \\multicolumn{6}{c}{Walk-forward} & \\\\",
        "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
        "Method & OR & M5 & Auto & Carp. & RAF & $\\Delta_{\\max}$"
        " & OR & M5 & Auto & Carp. & RAF & $\\Delta_{\\max}$ & Cost\\\\",
        "\\midrule",
    ]
    tail = ["\\bottomrule", "\\end{tabular}"]
    OUT.write_text("\n".join(head + lines + tail) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("best fixed:", best_f, "\nbest wf:", best_w)


if __name__ == "__main__":
    main()
