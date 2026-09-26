"""Format m5_walkforward.csv into LaTeX rows for tab:wfprob (M5 column) and
the appendix full table. Prints only; no .tex is modified."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/2026-09-05/m5_walkforward.csv"
df = pd.read_csv(src)
piv = df.pivot_table(index="model", columns=["metric", "quantile"], values="value")
BS = chr(92) * 2
order = ["CP-CrostonClassic", "CP-CrostonSBA", "CP-TSB", "CP-ADIDA", "CP-IMAPA",
         "ACI-ADIDA", "Zero", "TSB-HB"]
name = {"CP-CrostonClassic": "CP-Croston", "CP-CrostonSBA": "CP-SBA",
        "Zero": "Zero (diagnostic)", "TSB-HB": "EBB"}
print("=== full table rows: q10 q25 q50 q75 q90 Mean Cov@80 Cov+@80 AIW ===")
for m in order:
    if m not in piv.index:
        continue
    r = piv.loc[m]
    g = lambda met, q="": float(r[(met, q)]) if (met, q) in r.index else float("nan")
    qs = [g("scaled_pinball", str(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
    mean = g("scaled_pinball", "mean")
    print(f"{name.get(m, m)} & " + " & ".join(f"{v:.3f}" for v in qs)
          + f" & {mean:.3f} & {g('coverage80'):.3f} & {g('coverage80_positive'):.3f}"
          + f" & {g('aiw80'):.2f}{BS}")
print("=== tab:wfprob M5 cells: Mean q90 Cov+ AIW ===")
for m in order:
    if m not in piv.index:
        continue
    r = piv.loc[m]
    g = lambda met, q="": float(r[(met, q)]) if (met, q) in r.index else float("nan")
    print(f"{name.get(m, m)}: {g('scaled_pinball','mean'):.4f} & "
          f"{g('scaled_pinball','0.9'):.4f} & {g('coverage80_positive'):.3f} & {g('aiw80'):.2f}")
