"""Emit LaTeX table rows for paper_v2/main.tex from experiment CSVs.

Usage: py scripts/analysis/latex_tables.py
Prints: monthly point table, monthly prob (SPL mean) table, walk-forward table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "aistats2027"

MODEL_ORDER = [
    "CrostonClassic", "CrostonSBA", "TSB", "ADIDA", "IMAPA",
    "AutoARIMA", "AutoTheta", "Tweedie",
    "TSB-HB", "TSB-HB-Global", "TSB-HB-Taxonomy", "TSB-HB-Mixture",
    "TSB-HB-Discount", "TSB-HB-Mix-Disc",
]


def fmt(v: float, best: float, second: float) -> str:
    s = f"{v:.4f}"
    if abs(v - best) < 1e-9:
        return r"\best{" + s + "}"
    if abs(v - second) < 1e-9:
        return r"\secondbest{" + s + "}"
    return s


def monthly_point() -> None:
    for ds in ["auto", "carparts", "raf"]:
        p = OUT / f"{ds}_point_fixed" / "point_metrics.csv"
        if not p.exists():
            print(f"% {ds}: point metrics missing")
            continue
        df = pd.read_csv(p)
        df = df[df["model"].isin(MODEL_ORDER)].copy()
        df["order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
        df = df.sort_values("order")
        print(f"% ---- {ds} point ----")
        for metric_set in [["MAE", "RMSE", "RMSSE"]]:
            bests = {m: df[m].min() for m in metric_set}
            seconds = {m: df[m].nsmallest(2).iloc[-1] for m in metric_set}
            for _, r in df.iterrows():
                cells = " & ".join(fmt(r[m], bests[m], seconds[m]) for m in metric_set)
                print(f"{r['model']} & {cells}\\\\")
        print()


def monthly_prob() -> None:
    for ds in ["auto", "carparts", "raf"]:
        p = OUT / f"{ds}_prob_fixed" / "prob_pinball_scaled.csv"
        if not p.exists():
            print(f"% {ds}: prob SPL missing")
            continue
        spl = pd.read_csv(p)
        piv = spl.pivot_table(index="model", columns="quantile", values="scaled_pinball", aggfunc="first")
        cols = [c for c in ["0.5", "0.75", "0.9", "mean"] if c in [str(x) for x in piv.columns]]
        piv.columns = [str(c) for c in piv.columns]
        piv = piv[cols]
        print(f"% ---- {ds} prob SPL ----")
        bests = piv.min()
        seconds = piv.apply(lambda s: s.nsmallest(2).iloc[-1])
        for model, r in piv.iterrows():
            cells = " & ".join(fmt(r[c], bests[c], seconds[c]) for c in cols)
            print(f"{model} & {cells}\\\\")
        print()


def walkforward() -> None:
    frames = {}
    for tag, d in [("v1", "or_point_wf_v1"), ("disc", "or_point_wf_disc")]:
        p = OUT / d / "point_metrics.csv"
        if p.exists():
            frames[tag] = pd.read_csv(p)
    if not frames:
        print("% walk-forward: missing")
        return
    print("% ---- walk-forward OR point ----")
    for tag, df in frames.items():
        for _, r in df.sort_values("RMSSE").iterrows():
            print(f"% [{tag}] {r['model']}: MAE {r['MAE']:.4f} RMSE {r['RMSE']:.4f} RMSSE {r['RMSSE']:.4f}")


if __name__ == "__main__":
    monthly_point()
    monthly_prob()
    walkforward()
