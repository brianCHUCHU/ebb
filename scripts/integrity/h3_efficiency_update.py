"""Task 3 (redo): efficiency table with the aligned TweedieGP timings.

Takes efficiency_carparts.csv from outputs/2026-08-07 (all other rows are
unchanged same-session measurements), replaces the TweedieGP row with the
same-runner Carparts retime (h2 runner, 12 workers, released defaults),
appends TweedieGP rows for Online Retail and M5 from the full runs, and
records the three annotations requested by the authors in the notes column.
Then regenerates fig11 with the new Carparts TweedieGP cost.

Output: outputs/<date>/efficiency_carparts.csv, fig11 (via g3b script)

Usage: py scripts/integrity/h3_efficiency_update.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / date.today().isoformat()
SRC = ROOT / "outputs" / "2026-08-07" / "efficiency_carparts.csv"

ANNOT = (
    "[1] all rows: one machine, one session; absolute values not comparable across platforms. "
    "[2] Damato et al. (2025) report 0.24 s/series for TweedieGP on Carparts (M3 MacBook Pro); "
    "on this machine the released defaults give 1.89 s (single thread, median of 12 series) and "
    "the paper-stated 100-iteration budget gives 0.97 s; the released code doubles the iteration "
    "cap for Tweedie likelihoods when T<=500 (train_longer). "
    "[3] on M5 roughly half of TweedieGP's per-series time is the 50,000-sample predictive draw "
    "over the 647-step external span (h=28 in the original protocol): protocol difference, not method cost."
)


def main() -> None:
    eff = pd.read_csv(SRC)
    eff = eff[eff["method"] != "TweedieGP"].copy()

    rows = []
    for panel, label in (("carparts", "TweedieGP"),
                         ("online_retail", "TweedieGP [Online Retail]"),
                         ("m5", "TweedieGP [M5]")):
        meta = json.loads((OUT / f"h2_run_meta_{panel}.json").read_text())
        n = int(meta["n_series"])
        rows.append({
            "method": label, "wall_clock_total_s": round(meta["wall_clock_total_s"], 1),
            "s_per_series": round(meta["wall_clock_total_s"] / n, 3),
            "includes_selection": False, "hardware": "CPU", "runtime": "Python/torch",
            "gpu_used": False, "bit_reproducible": "No", "n_series": n,
            "notes": (f"official implementation, released defaults (train_longer active), "
                      f"12 worker processes; per-series wall median {meta['per_series_wall_median_s']} s "
                      f"(single thread); zero-window fallbacks {meta['n_zero_fallback']}, "
                      f"failed {meta['n_failed_after_3_attempts']}"),
        })
    eff = pd.concat([eff, pd.DataFrame(rows)], ignore_index=True)
    order = ["EBB", "TSB-tuned", "AutoARIMA", "AutoTheta", "CP-Croston", "CP-SBA", "CP-TSB",
             "CP-ADIDA", "CP-IMAPA", "Tweedie-GLM", "iETS", "TweedieGP", "Zero", "DeepState",
             "Chronos-Bolt-small", "DRP (Deep Renewal)", "TweedieGP [Online Retail]", "TweedieGP [M5]"]
    eff["_o"] = eff["method"].map({m: i for i, m in enumerate(order)})
    eff = eff.sort_values("_o").drop(columns="_o")
    eff["method"] = eff["method"].replace({"EBB": "EBB"})
    eff.loc[eff["method"] == "EBB", "notes"] = eff.loc[eff["method"] == "EBB", "notes"] + " || " + ANNOT
    eff.to_csv(OUT / "efficiency_carparts.csv", index=False)
    print(eff[["method", "wall_clock_total_s", "s_per_series", "n_series"]].to_string(index=False))

    # fig11 with the new Carparts TweedieGP cost (script reads today's CSV; EBB label)
    subprocess.run([sys.executable, str(ROOT / "scripts/integrity/g3b_fig11_cost_accuracy.py")],
                   check=True)


if __name__ == "__main__":
    main()
