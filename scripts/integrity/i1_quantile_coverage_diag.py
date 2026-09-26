"""Diagnostic: per-quantile empirical coverage of EBB vs TweedieGP.

For each nominal level q, reports the fraction of external observations
with y <= q-hat (overall and conditional on y > 0). A fraction above q at
the upper levels means the predictive quantile sits too high (over-wide
upper tail); below q means too low. Also reports mean q-hat at q90 on
positive-demand rows, to see which model puts its q90 where.

Panels: m5, online_retail. EBB quantiles from the corrected fixed-origin
runs (integrity_2026-08-06), TweedieGP from outputs/2026-09-05.

Usage: py scripts/integrity/i1_quantile_coverage_diag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

Q = [0.1, 0.25, 0.5, 0.75, 0.9]
SRC = {
    "m5": (ROOT / "outputs/integrity_2026-08-06/m5_prob_remix_corrected/prob_quantiles.csv",
           ROOT / "outputs/2026-09-05/tweediegp_quantiles_m5.csv"),
    "online_retail": (ROOT / "outputs/integrity_2026-08-06/online_retail_prob_remix_corrected/prob_quantiles.csv",
                      ROOT / "outputs/2026-09-05/tweediegp_quantiles_online_retail.csv"),
}


def load_y(panel):
    # external-span targets from the exported panel (same protocol, no reload)
    pan = pd.read_csv(ROOT / f"outputs/2026-09-05/tweediegp_panel_{panel}.csv",
                      dtype={"unique_id": str})
    ev = pan[pan["is_train"] == 0][["unique_id", "ds", "y"]].copy()
    ev["unique_id"] = ev["unique_id"].astype(str)
    ev["ds"] = pd.to_datetime(ev["ds"])
    return ev


def main() -> None:
    rows = []
    for panel, (ebb_p, tg_p) in SRC.items():
        ev = load_y(panel)
        cols = ["unique_id", "ds"] + [f"q_{q}" for q in Q]
        ebb = pd.read_csv(ebb_p, usecols=lambda c: c in cols + ["model"])
        ebb = ebb[ebb["model"] == "TSB-HB"][cols]
        tg = pd.read_csv(tg_p, usecols=cols)
        for name, qd in (("EBB", ebb), ("TweedieGP", tg)):
            qd = qd.copy()
            qd["unique_id"] = qd["unique_id"].astype(str)
            qd["ds"] = pd.to_datetime(qd["ds"])
            m = ev.merge(qd, on=["unique_id", "ds"], how="inner")
            pos = m[m["y"] > 0]
            for q in Q:
                col = f"q_{q}"
                rows.append({"panel": panel, "model": name, "q": q,
                             "cov_all": round(float((m["y"] <= m[col]).mean()), 3),
                             "cov_pos": round(float((pos["y"] <= pos[col]).mean()), 3),
                             "mean_qhat_pos": round(float(pos[col].mean()), 3),
                             "share_qhat_zero": round(float((m[col] <= 0).mean()), 3)})
            print(f"[{panel}] {name}: rows {len(m)}, positive share "
                  f"{(m['y']>0).mean():.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/2026-09-05/quantile_coverage_diag.csv", index=False)
    print(out.pivot_table(index=["panel", "q"], columns="model",
                          values=["cov_all", "cov_pos", "mean_qhat_pos", "share_qhat_zero"])
          .round(3).to_string())


if __name__ == "__main__":
    main()
