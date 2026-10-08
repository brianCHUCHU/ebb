"""v8 Table 1, in two variants written to paper_v2/v8_paper_runs/sections/:

  tab_main.tex          the comparator set of the v7 manuscript
  tab_main_deepar.tex   the same with a DeepAR row (docs/DESIGN_deepar.md)

Inputs : outputs/integrity_2026-09-05/rescored_tab_prob.csv   (fixed origin, mean SPL)
         outputs/2026-09-09/wf_all_panels_wide.csv           (walk-forward, mean SPL)
         outputs/deepar/deepar_results.csv, deepar_paired_all.csv, deepar_run_meta_*.json
         walk-forward wall-clock of the other methods from tab:wfcost (hard-coded strings)

Marking: underline = lowest value in the column among ranked rows; bold = that method together
with every method a paired per-series test does not distinguish from it (5%). The tie sets are
listed explicitly below with their sources. The all-zero forecast is a reference row and is not
ranked. Delta_max = largest percentage gap to the best competing method over the panels a method
was run on; for our rows the competitor set excludes our own variants (as in tab:regret).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEC = ROOT / "paper_v2" / "v8_paper_runs" / "sections"
DA = ROOT / "outputs" / "deepar"
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]
# DeepAR variant: columns in increasing median occurrence credibility (RAF .11, Auto .42,
# Carparts .63, Online Retail .90, M5 .93), as indices into PANELS
COL_ORDER = [4, 2, 3, 0, 1]
SHORT = {"online_retail": "OR", "m5": "M5", "auto": "Auto", "carparts": "Carp.", "raf": "RAF"}
EBB = "\\EBB{}"
RULE = "\\EBB{} (rule)$^{\\dagger}$"
OURS = {EBB, RULE}
# median occurrence credibility at the audited configuration (tab:leverage; s2_real_separation.py)
CRED_FILE = ROOT / "outputs" / "2026-09-13" / "real_separation.csv"
SAME_AS_FIXED = {"ACI-ADIDA": "CP-ADIDA", RULE: EBB}

BASE_ROWS = [
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
    ("DEEPAR_SLOT",),
    ("TweedieGP", "TweedieGP", "TweedieGP", True, "36\\,min--2.4\\,h$^{\\ddagger}$"),
    ("group", "Reference (unranked)"),
    ("All-zero forecast", "Zero", "Zero", False, "---"),
    ("group", "Ours"),
    (EBB, "EB-Hurdle", "EBB", True, "3--37\\,s"),
    (RULE, None, "EBB-rule", True, "3\\,s--3.6\\,min"),
]

# ---- tie sets without DeepAR (unchanged from v7_tab_main.py; sources documented there) ----
TIES_F0 = {0: {"TweedieGP", EBB}, 1: {"TweedieGP"}, 2: {EBB}, 3: {"TweedieGP", EBB}, 4: {EBB}}
TIES_W0 = {0: {"TweedieGP", EBB, RULE}, 1: {RULE}, 2: {RULE, EBB, "TweedieGP"},
           3: {"TweedieGP", EBB, RULE}, 4: {EBB, RULE}}


def fmt_dur(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f}\\,s"
    if sec < 5400:
        return f"{sec / 60:.0f}\\,min"
    return f"{sec / 3600:.1f}\\,h"


def deepar_ties(paired: pd.DataFrame, best_name_f, best_name_w):
    """Tie sets with DeepAR, derived from the stored paired tests.

    A method is tied with the column best if the paired test between the two is not significant
    at 5%. Pairs tested: EBB-TweedieGP (v7 sources), DeepAR-EBB and DeepAR-TweedieGP
    (outputs/deepar/deepar_paired_all.csv). EBB (rule) equals EBB or ACI-EBB by construction.
    """
    sig = {}
    for r in paired.itertuples():
        sig[(r.protocol, r.panel, r.other)] = bool(r.significant_5pct)
    tf, tw = {}, {}
    for j, p in enumerate(PANELS):
        # fixed origin
        s = set(TIES_F0[j])
        if best_name_f[j] == "DeepAR":
            s = {"DeepAR"}
            if not sig.get(("fixed", p, "TweedieGP"), True):
                s.add("TweedieGP")
            if not sig.get(("fixed", p, "EBB"), True):
                s.add(EBB)
        elif ("fixed", p, "EBB") in sig and best_name_f[j] in (EBB, "TweedieGP"):
            other = "EBB" if best_name_f[j] == EBB else "TweedieGP"
            if not sig.get(("fixed", p, other), True):
                s.add("DeepAR")
        tf[j] = s
        # walk-forward
        s = set(TIES_W0[j])
        if best_name_w[j] == "DeepAR":
            s = {"DeepAR"}
            if not sig.get(("wf", p, "TweedieGP"), True):
                s.add("TweedieGP")
            if not sig.get(("wf", p, "EBB"), True):
                s |= {EBB}
                if p in ("online_retail", "carparts", "raf"):   # rule withholds ACI there: rule == EBB
                    s.add(RULE)
        elif ("wf", p, "EBB") in sig:
            ref = {"TweedieGP": "TweedieGP", EBB: "EBB", RULE: "EBB"}.get(best_name_w[j])
            # the rule row is ACI-EBB on M5 and Auto; DeepAR is tested against EBB and TweedieGP only,
            # so on those panels it is marked tied only if indistinguishable from both
            if p in ("m5", "auto") and best_name_w[j] == RULE:
                tied = (not sig.get(("wf", p, "EBB"), True)) and (not sig.get(("wf", p, "TweedieGP"), True)
                                                                   if ("wf", p, "TweedieGP") in sig else True)
            else:
                tied = ref is not None and not sig.get(("wf", p, ref), True)
            if tied:
                s.add("DeepAR")
        tw[j] = s
    return tf, tw


def build(with_deepar: bool) -> str:
    fx = pd.read_csv(ROOT / "outputs" / "integrity_2026-09-05" / "rescored_tab_prob.csv")
    fx = fx.set_index(["panel", "model"])["spl_mean"]
    wf = pd.read_csv(ROOT / "outputs" / "2026-09-09" / "wf_all_panels_wide.csv")
    wf = wf.set_index(["panel", "model"])["mean"]
    rows = []
    for r in BASE_ROWS:
        if r[0] == "DEEPAR_SLOT":
            if with_deepar:
                res = pd.read_csv(DA / "deepar_results.csv")
                secs = []
                for p in PANELS:
                    f = DA / f"deepar_run_meta_wf_{p}.json"
                    if f.exists():
                        secs.append(json.loads(f.read_text())["wall_clock_seconds"])
                cost = f"{fmt_dur(min(secs))}--{fmt_dur(max(secs))}" if secs else "---"
                rows.append(("DeepAR", "DeepAR", "DeepAR", True, cost))
                for r_ in res.itertuples():
                    (fx if r_.protocol == "fixed" else wf).loc[(r_.panel, "DeepAR")] = r_.DeepAR
            continue
        rows.append(r)

    def get(series, panel, key):
        if key is None:
            return None
        try:
            return float(series.loc[(panel, key)])
        except KeyError:
            return None

    data = []
    for r in rows:
        if r[0] == "group":
            data.append(r); continue
        name, kf, kw, ranked, cost = r
        data.append((name, [get(fx, p, kf) for p in PANELS], [get(wf, p, kw) for p in PANELS], ranked, cost))
    ranked_rows = [d for d in data if d[0] != "group" and d[3]]

    def best_of(idx, j, pool):
        vals = [(round(d[idx][j], 4), d[0]) for d in pool if d[idx][j] is not None]
        return min(vals) if vals else (None, None)

    best_f = [best_of(1, j, ranked_rows) for j in range(5)]
    best_w = [best_of(2, j, ranked_rows) for j in range(5)]
    comp = [d for d in ranked_rows if d[0] not in OURS]
    comp_f = [best_of(1, j, comp)[0] for j in range(5)]
    comp_w = [best_of(2, j, comp)[0] for j in range(5)]
    if with_deepar:
        paired = pd.read_csv(DA / "deepar_paired_all.csv")
        ties_f, ties_w = deepar_ties(paired, [b[1] for b in best_f], [b[1] for b in best_w])
    else:
        ties_f, ties_w = TIES_F0, TIES_W0

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

    def shortfall(vals, bests):
        gaps = [(v / b - 1) * 100 for v, b in zip(vals, bests) if v is not None and b is not None]
        return f"{max(gaps):.1f}" if gaps else "---"

    # The DeepAR variant drops the Delta_max columns (a maximum over "the panels run" favours
    # methods that were not run on M5) and carries the occurrence credibility of each panel instead.
    ncol = 12 if with_deepar else 14
    lines = []
    for d in data:
        if d[0] == "group":
            lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{d[1]}}}}}\\\\"); continue
        name, vf, vw, ranked, cost = d
        # column order: the v7 layout without DeepAR; increasing occurrence credibility with it
        order = COL_ORDER if with_deepar else list(range(len(PANELS)))
        cf = " & ".join(fmt(vf[j], best_f[j][0], ranked, name, ties_f[j]) for j in order)
        cw = " & ".join(fmt(vw[j], best_w[j][0], ranked, name, ties_w[j]) for j in order)
        bf, bw = (comp_f, comp_w) if name in OURS else ([b[0] for b in best_f], [b[0] for b in best_w])
        sf = shortfall(vf, bf) if ranked else "---"
        sw = shortfall(vw, bw) if ranked else "---"
        if with_deepar:
            if all(v is None for v in vf) and name in SAME_AS_FIXED:
                lines.append(f"{name} & \\multicolumn{{5}}{{c}}{{= {SAME_AS_FIXED[name]}}} & {cw} & {cost}\\\\")
            else:
                lines.append(f"{name} & {cf} & {cw} & {cost}\\\\")
            continue
        if all(v is None for v in vf) and name in SAME_AS_FIXED:
            lines.append(f"{name} & \\multicolumn{{6}}{{c}}{{= {SAME_AS_FIXED[name]}}} & {cw} & {sw} & {cost}\\\\")
            continue
        lines.append(f"{name} & {cf} & {sf} & {cw} & {sw} & {cost}\\\\")
    head = ["\\begin{tabular}{lrrrrrrrrrrrrl}", "\\toprule",
            " & \\multicolumn{6}{c}{Fixed origin} & \\multicolumn{6}{c}{Walk-forward} & \\\\",
            "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
            "Method & OR & M5 & Auto & Carp. & RAF & $\\Delta_{\\max}$"
            " & OR & M5 & Auto & Carp. & RAF & $\\Delta_{\\max}$ & Cost\\\\", "\\midrule"]
    if with_deepar:
        cred = pd.read_csv(CRED_FILE).set_index("panel")["median_lambda_occ"]
        cr = " & ".join(f"{float(cred.loc[PANELS[j]]):.2f}" for j in COL_ORDER)
        hd = " & ".join(SHORT[PANELS[j]] for j in COL_ORDER)
        head = ["\\begin{tabular}{lrrrrrrrrrrl}", "\\toprule",
                " & \\multicolumn{5}{c}{Fixed origin} & \\multicolumn{5}{c}{Walk-forward} & \\\\",
                "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
                f"Method & {hd} & {hd} & Cost\\\\",
                f"\\emph{{Median}} $\\lambda^{{(o)}}$ & {cr} & {cr} & \\\\", "\\midrule"]
    tail = ["\\bottomrule", "\\end{tabular}"]
    text ="\n".join(head + lines + tail) + "\n"
    info = {"best_fixed": best_f, "best_wf": best_w, "ties_fixed": ties_f, "ties_wf": ties_w}
    return text, lines, info


def main() -> None:
    t0, l0, _ = build(False)
    old = (SEC / "tab_main.tex").read_text(encoding="utf-8") if (SEC / "tab_main.tex").exists() else None
    (SEC / "tab_main.tex").write_text(t0, encoding="utf-8")
    print("tab_main.tex identical to previous:", old == t0)
    if (DA / "deepar_results.csv").exists() and (DA / "deepar_paired_all.csv").exists():
        t1, l1, info = build(True)
        (SEC / "tab_main_deepar.tex").write_text(t1, encoding="utf-8")
        for l in l1:
            if l.startswith(("DeepAR", "TweedieGP", "\\EBB", "ACI-ADIDA")):
                print(l)
        print(info)


if __name__ == "__main__":
    sys.exit(main())
