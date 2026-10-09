"""D3: score DeepAR quantiles with the shared scoring functions and set them next to EBB and
TweedieGP (docs/DESIGN_deepar.md).

Reads outputs/deepar/deepar_<protocol>_<panel>.csv; writes outputs/deepar/deepar_results.csv.
Usage: py scripts/integrity/d3_deepar_score.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("c1", ROOT / "scripts" / "integrity" / "c1_coldstart.py")
c1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c1)

OUT = ROOT / "outputs" / "deepar"
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]


def main() -> None:
    fx = pd.read_csv(ROOT / "outputs" / "integrity_2026-09-05" / "rescored_tab_prob.csv").set_index(["panel", "model"])["spl_mean"]
    wf = pd.read_csv(ROOT / "outputs" / "2026-09-09" / "wf_all_panels_wide.csv").set_index(["panel", "model"])["mean"]
    rows = []
    for protocol in ("fixed", "wf"):
        for p in PANELS:
            f = OUT / f"deepar_{protocol}_{p}.csv"
            if not f.exists():
                continue
            init = pd.read_csv(OUT / f"panel_{p}_init.csv", dtype={"unique_id": str})
            ev = pd.read_csv(OUT / f"panel_{p}_eval.csv", dtype={"unique_id": str})
            for d in (init, ev):
                d["ds"] = pd.to_datetime(d["ds"])
            q = pd.read_csv(f, dtype={"unique_id": str})
            q["ds"] = pd.to_datetime(q["ds"])
            for c in c1.QCOLS:
                q[c] = np.maximum(q[c].astype(float), 0.0)
            m, q90, n = c1.score(q, "DeepAR", ev[["unique_id", "ds", "y"]], init)
            meta_f = OUT / f"deepar_run_meta_{protocol}_{p}.json"
            meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
            ref = fx if protocol == "fixed" else wf
            ebb = float(ref.get((p, "EB-Hurdle" if protocol == "fixed" else "EBB"), np.nan))
            tg = float(ref.get((p, "TweedieGP"), np.nan))
            rows.append({"protocol": protocol, "panel": p, "DeepAR": round(m, 4), "DeepAR_q90": round(q90, 4),
                         "EBB": ebb, "TweedieGP": tg, "EBB_vs_DeepAR_pct": round((m / ebb - 1) * 100, 2),
                         "n_series": n, "rows": len(q), "epochs": meta.get("epochs"),
                         "wall_s": meta.get("wall_clock_seconds")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "deepar_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
