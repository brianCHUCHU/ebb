"""Integrity task 4: reconcile the 10x gap in refinement/forgetting gains.

Two sets of numbers exist for "refinement vs forgetting on the internal
validation surface":
  (a) v4 draft: refinement 0.54%, forgetting 1.4%
      (author hand-check quotes 0.54% / 1.18%, citing global w=1 0.75320 ->
       global w=0.95 0.74434 -- values that appear verbatim in the LEGACY
       Online Retail surface, not in any M5 surface)
  (b) this repo, m5_prob_correct_split surface at w*=0.90:
      refinement 4.62%, forgetting (mixture, w=1 -> 0.90) 12.28%

This script scans every stored selection surface and, for every discount,
tabulates every plausible definition of the two gains, then reports which
(surface, discount, definition) combinations reproduce 0.54 / 1.4 / 1.18.
Output: t4_gain_scan.csv + console report (conclusions go to NOTES.md).

Usage: py scripts/integrity/t4_gain_reconciliation.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
OUT.mkdir(parents=True, exist_ok=True)
A = ROOT / "outputs" / "paper_runs"
B = ROOT / "outputs" / "paper_rebuild"

SURFACES = {
    "or_legacy": A / "or_prob_select" / "ebb_selection_diagnostics.csv",
    "m5_legacy_thirdsplit": A / "m5_prob_select" / "ebb_selection_diagnostics.csv",
    "auto_legacy": A / "auto_prob_select" / "ebb_selection_diagnostics.csv",
    "carparts_legacy": A / "carparts_prob_select" / "ebb_selection_diagnostics.csv",
    "raf_legacy": A / "raf_prob_select" / "ebb_selection_diagnostics.csv",
    "m5_correct_seed42": B / "m5_prob_correct_split" / "ebb_selection_diagnostics.csv",
    "m5_thirdsplit_seed43": B / "m5_prob_seed43" / "ebb_selection_diagnostics.csv",
    "m5_correct_seed43": B / "m5_prob_seed43_correct" / "ebb_selection_diagnostics.csv",
    "m5_thirdsplit_seed44": B / "m5_prob_seed44" / "ebb_selection_diagnostics.csv",
    "m5_correct_seed44": B / "m5_prob_seed44_correct" / "ebb_selection_diagnostics.csv",
}
# Task-1 corrected surfaces, if already produced by t1_leakage_safe_selection.
T1 = OUT / "selection_surfaces.csv"

TARGETS = {"draft_refine": 0.54, "draft_forget": 1.4, "author_forget": 1.18}
TOL = 0.03  # percentage points


def load_all() -> pd.DataFrame:
    frames = []
    for name, path in SURFACES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path).rename(columns={"scaled_pinball": "validation_spl"})
        df["surface"] = name
        frames.append(df)
    if T1.exists():
        df = pd.read_csv(T1)
        df["surface"] = "t1_corrected_" + df["panel"]
        frames.append(df[["surface", "structure", "discount", "validation_spl"]])
    return pd.concat(frames, ignore_index=True)


def scan(all_surf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in all_surf.groupby("surface"):
        surf = df.set_index(["structure", "discount"])["validation_spl"]
        ws = sorted(df["discount"].unique(), reverse=True)
        structs = [s for s in ("global", "taxonomy", "mixture") if (s, ws[0]) in surf]
        for w in ws:
            g = surf.get(("global", w))
            t = surf.get(("taxonomy", w))
            m = surf.get(("mixture", w))
            defs = {
                "refine_global_to_best": 100 * (g - min(t, m)) / g,
                "refine_global_to_mixture": 100 * (g - m) / g,
                "refine_taxonomy_to_mixture": 100 * (t - m) / t,
                "refine_global_to_taxonomy": 100 * (g - t) / g,
            }
            for s in structs:
                defs[f"forget_{s}_w1_to_here"] = (
                    100 * (surf[(s, 1.0)] - surf[(s, w)]) / surf[(s, 1.0)]
                )
            for defname, val in defs.items():
                rows.append({"surface": name, "discount": w,
                             "definition": defname, "gain_pct": round(float(val), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    all_surf = load_all()
    scan_df = scan(all_surf)
    scan_df.to_csv(OUT / "t4_gain_scan.csv", index=False)

    print("=== matches within +/-%.2f pp ===" % TOL)
    for tname, tval in TARGETS.items():
        hits = scan_df[(scan_df["gain_pct"] - tval).abs() <= TOL]
        print(f"\n-- target {tname} = {tval}% --")
        print(hits.to_string(index=False) if len(hits) else "  (no match)")

    # The author's hand-check values, verbatim lookup:
    print("\n=== where do 0.75320 / 0.74434 live? ===")
    v = all_surf[all_surf["validation_spl"].round(5).isin([0.75320, 0.74434])]
    print(v.to_string(index=False) if len(v) else "  (not found)")

    print("\nwrote", OUT / "t4_gain_scan.csv")


if __name__ == "__main__":
    main()
