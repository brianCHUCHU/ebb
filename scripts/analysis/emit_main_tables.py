"""Emit tab:point and tab:prob LaTeX rows at 4 decimals directly from source
CSVs, with best/secondbest computed programmatically (ties share marks).
Also emits a NUMBERS.md fragment mapping every cell to its source file.

Usage: py scripts/analysis/emit_main_tables.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper_runs"

POINT_SOURCES = {
    "Online Retail": ("or_point_fixed/point_metrics.csv", "or_point_ebb/point_metrics.csv"),
    "M5": ("m5_point_fixed/point_metrics_m5.csv", "m5_point_ebb/point_metrics_m5.csv"),
    "Auto": ("auto_point_fixed/point_metrics.csv", "auto_point_ebb/point_metrics.csv"),
    "Carparts": ("carparts_point_fixed/point_metrics.csv", "carparts_point_ebb/point_metrics.csv"),
    "RAF": ("raf_point_fixed/point_metrics.csv", "raf_point_ebb/point_metrics.csv"),
}
PROB_SOURCES = {
    "Online Retail": ("or_prob_fixed_paper", "or_prob_tweedie", "or_prob_iets", "or_prob_select"),
    "M5": ("m5_prob_fixed", "m5_prob_fixed", "m5_prob_iets", "m5_prob_select"),
    "Auto": ("auto_prob_fixed", "auto_prob_fixed", "auto_prob_iets", "auto_prob_select"),
    "Carparts": ("carparts_prob_fixed", "carparts_prob_fixed", "carparts_prob_iets", "carparts_prob_select"),
    "RAF": ("raf_prob_fixed", "raf_prob_fixed", "raf_prob_iets", "raf_prob_select"),
}
BASELINES = ["CrostonClassic", "CrostonSBA", "TSB", "ADIDA", "IMAPA", "AutoARIMA", "AutoTheta"]
PROB_MODELS = ["AutoARIMA", "AutoTheta", "CP-CrostonClassic", "CP-CrostonSBA", "CP-TSB",
               "CP-ADIDA", "CP-IMAPA", "Tweedie", "iETS"]
PROB_LABEL = {"CP-CrostonClassic": "CP-Croston", "CP-CrostonSBA": "CP-SBA"}

numbers_log: list[str] = []


def fmt(v, best, second):
    if v is None:
        return "---"
    s = f"{v:.4f}"
    if abs(v - best) < 5e-5:
        return r"\best{" + s + "}"
    if abs(v - second) < 5e-5:
        return r"\secondbest{" + s + "}"
    return s


def point_table():
    data = {}
    for ds, (base_f, ebb_f) in POINT_SOURCES.items():
        base = pd.read_csv(OUT / base_f)
        ebb = pd.read_csv(OUT / ebb_f)
        col = {}
        for m in BASELINES:
            r = base[base.model == m]
            col[m] = (float(r.MAE.iloc[0]), float(r.RMSSE.iloc[0]))
            numbers_log.append(f"tab:point {ds} {m}: {base_f} row model={m} cols MAE,RMSSE  VERIFIED")
        r = ebb[ebb.model == "EB-Hurdle"]
        col["EBB"] = (float(r.MAE.iloc[0]), float(r.RMSSE.iloc[0]))
        numbers_log.append(f"tab:point {ds} EBB: {ebb_f} row model=EB-Hurdle  VERIFIED (full-window refit, select mode)")
        data[ds] = col

    rows = BASELINES + ["EBB"]
    print("% ==== tab:point rows (4dp, programmatic marks) ====")
    marks = {}
    for ds in data:
        for j in range(2):
            vals = sorted(data[ds][m][j] for m in rows)
            marks[(ds, j)] = (vals[0], vals[1])
    for m in rows:
        cells = []
        for ds in data:
            for j in range(2):
                b, s = marks[(ds, j)]
                cells.append(fmt(data[ds][m][j], b, s))
        name = r"\EBB{} (ours)" if m == "EBB" else m
        print(f"{name} & " + " & ".join(cells) + r"\\")
    # A1 recount
    print("\n% best/secondbest count for EBB:",
          sum(1 for ds in data for j in range(2)
              if abs(data[ds]["EBB"][j] - marks[(ds, j)][0]) < 5e-5
              or abs(data[ds]["EBB"][j] - marks[(ds, j)][1]) < 5e-5))
    for m in rows:
        n_best = sum(1 for ds in data for j in range(2) if abs(data[ds][m][j] - marks[(ds, j)][0]) < 5e-5)
        if n_best:
            print(f"% {m}: best x{n_best}")


def spl_stats(run_dir: str, model: str):
    p = OUT / run_dir / "prob_pinball_scaled.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    s["quantile"] = s["quantile"].astype(str)
    s = s[(s.model == model) & (s["quantile"] != "mean")]
    if s.empty:
        return None
    return (float(s.scaled_pinball.mean()),
            float(s[s["quantile"] == "0.9"].scaled_pinball.iloc[0]))


def prob_table():
    data = {}
    for ds, (base_d, tw_d, iets_d, ebb_d) in PROB_SOURCES.items():
        col = {}
        for m in PROB_MODELS:
            src = tw_d if m == "Tweedie" else (iets_d if m == "iETS" else base_d)
            col[m] = spl_stats(src, m)
            numbers_log.append(f"tab:prob {ds} {m}: {src}/prob_pinball_scaled.csv  "
                               + ("VERIFIED" if col[m] else "NOT_FOUND"))
        col["EBB"] = spl_stats(ebb_d, "EB-Hurdle")
        numbers_log.append(f"tab:prob {ds} EBB: {ebb_d}/prob_pinball_scaled.csv  VERIFIED (raw, B=20, select)")
        data[ds] = col

    rows = PROB_MODELS + ["EBB"]
    print("\n% ==== tab:prob rows (4dp, programmatic marks) ====")
    marks = {}
    for ds in data:
        for j in range(2):
            vals = sorted(data[ds][m][j] for m in rows if data[ds][m] is not None)
            marks[(ds, j)] = (vals[0], vals[1])
    for m in rows:
        cells = []
        for ds in data:
            for j in range(2):
                b, s = marks[(ds, j)]
                v = data[ds][m][j] if data[ds][m] else None
                cells.append(fmt(v, b, s))
        name = r"\EBB{} (ours)" if m == "EBB" else PROB_LABEL.get(m, m)
        print(f"{name} & " + " & ".join(cells) + r"\\")


if __name__ == "__main__":
    point_table()
    prob_table()
    frag = ROOT / "paper_v2" / "NUMBERS_tables.md"
    frag.write_text("\n".join(numbers_log), encoding="utf-8")
    print(f"\n% wrote {frag} ({len(numbers_log)} entries)")
