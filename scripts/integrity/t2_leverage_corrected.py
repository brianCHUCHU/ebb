"""Integrity task 2: dual-credibility leverage table at the corrected w*.

Recomputes median lambda^(o), lambda^(+), the lambda^(+) < 0.1 share, and the
w=1 references for every panel at the leakage-safe selected discounts from
t1_leakage_safe_selection (2026-08-06 adjudication):

    Online Retail 0.95   M5 0.95   Auto 0.99   Carparts 0.90   RAF 0.997

The diagnostic itself is imported unchanged from
scripts/analysis/leverage_dual_lambda.py (dual_lambda, load_init), so any
difference from the published table comes from the discount alone.

Output: outputs/integrity_<date>/leverage_dual_lambda_corrected.csv
        + t2_run_meta.json

Usage: py scripts/integrity/t2_leverage_corrected.py
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "leverage_dual_lambda", ROOT / "scripts" / "analysis" / "leverage_dual_lambda.py")
ldl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ldl)

import pandas as pd  # noqa: E402  (after module exec so src path is set)

W_STAR_CORRECTED = {"online_retail": 0.95, "m5": 0.95, "auto": 0.99,
                    "carparts": 0.90, "raf": 0.997}


def main() -> None:
    t0 = time.time()
    rows = []
    for name in ["online_retail", "auto", "carparts", "raf", "m5"]:
        init = ldl.load_init(name)
        for w in (1.0, W_STAR_CORRECTED[name]):
            r = ldl.dual_lambda(init, w)
            r.update({"panel": name, "w": w})
            rows.append(r)
            print(f"{name:14s} w={w:<6} "
                  f"occ: med={r['median_lambda_occ']:.3f} <0.1={r['share_occ_lt_0.1']:.1%} | "
                  f"size: med={r['median_lambda_size']:.3f} <0.1={r['share_size_lt_0.1']:.1%}",
                  flush=True)
    out = pd.DataFrame(rows)[[
        "panel", "w", "median_lambda_occ", "share_occ_lt_0.1",
        "median_lambda_size", "share_size_lt_0.1",
        "median_n_obs", "median_n_pos", "phi_global", "kappa_global",
    ]]
    out.to_csv(OUT / "leverage_dual_lambda_corrected.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "t2_leverage_corrected",
        "commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "w_star": W_STAR_CORRECTED,
        "diagnostic_source": "scripts/analysis/leverage_dual_lambda.py (imported unchanged)",
        "seed": {"m5_sample": 42},
        "wall_clock_total_s": round(time.time() - t0, 1),
    }
    (OUT / "t2_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote leverage_dual_lambda_corrected.csv, t2_run_meta.json")


if __name__ == "__main__":
    main()
