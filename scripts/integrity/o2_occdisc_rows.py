"""O2: LaTeX rows for the two-rate forgetting gate table (tab:occdisc) from
outputs/<date>/occdisc_eval.csv.
Columns: Panel & w & w_o* & tied mean & selected mean & Delta (%) & q90-zero share tied -> selected & TweedieGP
Usage: py scripts/integrity/o2_occdisc_rows.py [YYYY-MM-DD]
"""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
OUT = ROOT / "outputs" / DAY
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]
PNL = {"online_retail": "Online Retail", "m5": "M5", "auto": "Auto", "carparts": "Carparts", "raf": "RAF"}

d = pd.read_csv(OUT / "occdisc_eval.csv")
lines = []
for p in PANELS:
    g = d[d["panel"] == p]
    if g.empty:
        continue
    t = g[g["w_occ"].astype(str) == "tied"].iloc[0]
    s = g[g["selected"]].iloc[0]
    wo = "tied" if str(s["w_occ"]) == "tied" else f"{float(s['w_occ']):g}"
    delta = 100 * (s["spl_mean"] - t["spl_mean"]) / t["spl_mean"]
    lines.append(f"{PNL[p]} & {t['w']:g} & {wo} & {t['spl_mean']:.4f} & {s['spl_mean']:.4f} & {delta:+.2f} & "
                 f"{t['q90_zero_share']:.3f} $\\to$ {s['q90_zero_share']:.3f} & {t['tweediegp']:.4f}\\\\")
(OUT / "tab_occdisc.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
