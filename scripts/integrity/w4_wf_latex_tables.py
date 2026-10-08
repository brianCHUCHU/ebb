"""W4: emit LaTeX for the five-panel walk-forward tables from W3 output.

Reads outputs/<date>/wf_all_panels_wide.csv (W3). Writes
  outputs/<date>/tab_wfprob_main.tex        five panels x (Mean, q90)
  outputs/<date>/tab_wf_full_<panel>.tex     q10..q90, Mean, Cov@80, Cov+@80, AIW
  outputs/<date>/wf_standings.csv, w4_run_meta.json
Bold = best, underline = second best among non-degenerate rows (Zero excluded).
Usage: py scripts/integrity/w4_wf_latex_tables.py [YYYY-MM-DD]
"""
from __future__ import annotations
import json, platform, sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
PANELS = ["online_retail", "carparts", "auto", "raf", "m5"]
MAIN_ROWS = ["AutoARIMA", "AutoTheta", "iETS", "CP-TSB", "CP-ADIDA", "CP-IMAPA", "ACI-ADIDA",
             "TweedieGP", "Zero", "EBB", "ACI-EBB", "EBB-rule"]
FULL_ROWS = ["AutoARIMA", "AutoTheta", "iETS", "CP-Croston", "CP-SBA", "CP-TSB", "CP-ADIDA",
             "CP-IMAPA", "ACI-ADIDA", "ACI-TSB", "TweedieGP", "Zero", "EBB", "ACI-EBB", "EBB-rule"]
LABEL = {"Zero": "Zero (diagnostic)", "EBB": r"\EBB{}", "ACI-EBB": r"ACI-\EBB{}", "EBB-rule": r"\EBB{} (rule)"}


def fmt(v, nd, mark):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    s = f"{v:.{nd}f}"
    return {1: rf"\best{{{s}}}", 2: rf"\secondbest{{{s}}}"}.get(mark, s)


def ranks(sub, col):
    # ranking excludes the diagnostic row and the rule row (a per-panel choice between two rows above it)
    s = sub.loc[[m for m in sub.index if m not in ("Zero", "EBB-rule")], col].dropna().sort_values()
    out = {}
    if len(s) > 0: out[s.index[0]] = 1
    if len(s) > 1: out[s.index[1]] = 2
    return out


def main():
    wide = pd.read_csv(OUT / "wf_all_panels_wide.csv").set_index(["panel", "model"])
    lines = []
    for m in MAIN_ROWS:
        if m == "EBB":
            lines.append(r"\midrule")
        cells = [LABEL.get(m, m)]
        for p in PANELS:
            sub = wide.loc[p]
            rm, rq = ranks(sub, "mean"), ranks(sub, "0.9")
            for col, rk in (("mean", rm), ("0.9", rq)):
                v = sub.loc[m, col] if m in sub.index else None
                cells.append(fmt(v, 4, rk.get(m, 0)))
        lines.append(" & ".join(cells) + r"\\")
    (OUT / "tab_wfprob_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in PANELS:
        sub = wide.loc[p]
        rk = {c: ranks(sub, c) for c in ["0.1", "0.25", "0.5", "0.75", "0.9", "mean"]}
        lines = []
        for m in FULL_ROWS:
            if m not in sub.index:
                continue
            if m == "EBB":
                lines.append(r"\midrule")
            cells = [LABEL.get(m, m)]
            for c in ["0.1", "0.25", "0.5", "0.75", "0.9", "mean"]:
                cells.append(fmt(sub.loc[m, c], 3, rk[c].get(m, 0)))
            for c in ["coverage80", "coverage80_positive"]:
                cells.append(fmt(sub.loc[m, c], 3, 0))
            cells.append(fmt(sub.loc[m, "aiw80"], 2, 0))
            lines.append(" & ".join(cells) + r"\\")
        (OUT / f"tab_wf_full_{p}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    for p in PANELS:
        sub = wide.loc[p]
        nd = sub.loc[[m for m in sub.index if m != "Zero"]].sort_values("mean")
        ebb = float(sub.loc["EBB", "mean"])
        rival_name = [m for m in nd.index if m != "EBB"][0]
        rival = float(nd.loc[rival_name, "mean"])
        rows.append({"panel": p, "best": nd.index[0], "best_mean": round(float(nd.iloc[0]["mean"]), 4),
                     "second": nd.index[1], "second_mean": round(float(nd.iloc[1]["mean"]), 4),
                     "ebb_mean": round(ebb, 4), "strongest_rival": rival_name,
                     "ebb_margin_pct": round(100 * (rival - ebb) / rival, 2),
                     "zero_mean": round(float(sub.loc["Zero", "mean"]), 4)})
    st = pd.DataFrame(rows); st.to_csv(OUT / "wf_standings.csv", index=False); print(st.to_string())
    meta = {"task": "w4_wf_latex_tables", "input": "wf_all_panels_wide.csv", "day": DAY,
            "ranking": "best/second among non-degenerate rows; Zero excluded",
            "platform": platform.platform(), "python": sys.version.split()[0]}
    (OUT / "w4_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
