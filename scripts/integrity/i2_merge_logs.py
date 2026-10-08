"""Merge the two predictive-law variant runs (logs) into one CSV.

The i2 script rewrites its CSV per invocation, so the OR/Auto/Carparts rows
from the first run live only in outputs/logs/i2_variants.log. This parses
both logs and writes outputs/2026-09-05/predictive_law_variants_all.csv.
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
pat = re.compile(r"^\[(\w+)\] (\S+)\s+mean ([0-9.]+)\s+q90 ([0-9.]+)")
rows = []
for name in ("i2_variants.log", "i2_variants_b.log"):
    p = ROOT / "outputs" / "logs" / name
    if not p.exists():
        continue
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line.strip())
        if m:
            rows.append({"panel": m.group(1), "law": m.group(2),
                         "spl_mean": float(m.group(3)), "spl_q90": float(m.group(4))})
df = pd.DataFrame(rows).drop_duplicates(subset=["panel", "law"], keep="last")
out = ROOT / "outputs" / "2026-09-05" / "predictive_law_variants_all.csv"
df.to_csv(out, index=False)
print(df.pivot_table(index="panel", columns="law", values="spl_mean").round(4).to_string())
