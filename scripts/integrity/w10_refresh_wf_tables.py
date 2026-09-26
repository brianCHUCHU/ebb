"""W10: refresh the bodies of the walk-forward-derived tables in both manuscript
directories from the latest generated rows (W4 / W6 outputs), leaving captions
and headers untouched.

Tables: tab:wfprob, tab:wfprob-full, tab:wf-carparts-full, tab:wf-auto-full,
        tab:wf-raf-full, tab:wf-m5-full, tab:regret, tab:regret-full, tab:wfcost
Body = everything between the header's first `\\midrule` after the label and
`\\bottomrule`.
Usage: py scripts/integrity/w10_refresh_wf_tables.py [YYYY-MM-DD]
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
DIRS = [ROOT / "paper_v2" / "v5", ROOT / "paper_v2" / "v5_aistats"]
PANELS = ["online_retail", "carparts", "auto", "raf", "m5"]
PN = {"online_retail": "OR", "carparts": "Carp.", "auto": "Auto", "raf": "RAF", "m5": "M5"}
LAB = {"EBB": r"\EBB{}", "ACI-EBB": r"ACI-\EBB{}", "EBB-rule": r"\EBB{} (rule)"}
ROSTER = ["AutoARIMA", "AutoTheta", "Tweedie-GLM", "iETS", "CP-Croston", "CP-SBA", "CP-TSB",
          "CP-ADIDA", "CP-IMAPA", "ACI-ADIDA", "TweedieGP", "EBB", "ACI-EBB", "EBB-rule"]


def regret_bodies():
    rf = pd.read_csv(OUT / "regret_fixed.csv").set_index("model")
    rw = pd.read_csv(OUT / "regret_wf.csv").set_index("model")
    main, full = [], []
    for m in ROSTER:
        if m not in rf.index and m not in rw.index:
            continue
        if m == "EBB":
            main.append(r"\midrule"); full.append(r"\midrule")
        c, fc = [LAB.get(m, m)], [LAB.get(m, m)]
        for r in (rf, rw):
            if m in r.index:
                n = int(r.loc[m, "n_panels"]); sup = "" if n == 5 else f"$^{{{n}}}$"
                c.append(f"{r.loc[m, 'max_regret']:.1f} ({PN[r.loc[m, 'worst_panel']]}){sup}")
                c.append(f"{int(r.loc[m, 'within5'])}/{n}")
                fc += ["---" if pd.isna(r.loc[m, p]) else f"{r.loc[m, p]:.1f}" for p in PANELS]
            else:
                c += ["---", "---"]; fc += ["---"] * 5
        main.append(" & ".join(c) + r"\\"); full.append(" & ".join(fc) + r"\\")
    return "\n".join(main) + "\n", "\n".join(full) + "\n"


def swap_body(s: str, label: str, body: str) -> str:
    i = s.index("\\label{" + label + "}")
    j = s.index("\\midrule\n", i) + len("\\midrule\n")
    k = s.index("\\bottomrule", j)
    return s[:j] + body + s[k:]


def main():
    rm, rfull = regret_bodies()
    bodies = {"tab:wfprob": (OUT / "tab_wfprob_main.tex").read_text(encoding="utf-8"),
              "tab:wfprob-full": (OUT / "tab_wf_full_online_retail.tex").read_text(encoding="utf-8"),
              "tab:wf-carparts-full": (OUT / "tab_wf_full_carparts.tex").read_text(encoding="utf-8"),
              "tab:wf-auto-full": (OUT / "tab_wf_full_auto.tex").read_text(encoding="utf-8"),
              "tab:wf-raf-full": (OUT / "tab_wf_full_raf.tex").read_text(encoding="utf-8"),
              "tab:wf-m5-full": (OUT / "tab_wf_full_m5.tex").read_text(encoding="utf-8"),
              "tab:regret": rm, "tab:regret-full": rfull,
              "tab:wfcost": (OUT / "tab_wfcost.tex").read_text(encoding="utf-8")}
    for d in DIRS:
        for fname in ("sections/experiments.tex", "sections/experiment_appendix.tex"):
            p = d / fname
            s = p.read_text(encoding="utf-8")
            for lab, body in bodies.items():
                if "\\label{" + lab + "}" in s:
                    s = swap_body(s, lab, body)
            p.write_text(s, encoding="utf-8")
        print("refreshed", d)


if __name__ == "__main__":
    main()
