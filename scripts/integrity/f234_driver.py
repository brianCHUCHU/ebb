"""F2/F3/F4: remaining fixed-origin reruns under audited configurations.

  F2  tab:leverage learned-vs-pool cells:
        Online Retail point (mixture, 0.95)   vs stored global 5.5325
        Auto          point (mixture, 0.99)   vs stored global 3.2145
  F3  Online Retail ablation "auto" rows at w*=0.95:
        prob + point for taxonomy@0.95 and mixture@0.95
        (global@0.95 already exists in integrity_2026-08-06)
  F4  M5 seed variance at (mixture, 0.95): prob for seeds 43 and 44
        (seed 42 = m5_prob_ebb_corrected, 1.6707)

All runs are the existing run_prob / run_point CLIs with the recorded
release flags (hb_only, calibration none, B=20). Outputs under
outputs/integrity_<date>/.

Usage: py scripts/integrity/f234_driver.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / f"integrity_{date.today().isoformat()}"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    # (dirname, module, args)
    ("or_prob_taxonomy_095", "run_prob",
     ["--dataset", "online", "--hb-grouping", "taxonomy", "--hb-fit-discount", "0.95"]),
    ("or_prob_mixture_095", "run_prob",
     ["--dataset", "online", "--hb-grouping", "mixture", "--hb-fit-discount", "0.95"]),
    ("or_point_taxonomy_095", "run_point",
     ["--dataset", "online_retail", "--hb-grouping", "taxonomy", "--hb-fit-discount", "0.95"]),
    ("or_point_mixture_095", "run_point",
     ["--dataset", "online_retail", "--hb-grouping", "mixture", "--hb-fit-discount", "0.95"]),
    ("auto_point_mixture_099", "run_point",
     ["--dataset", "auto", "--hb-grouping", "mixture", "--hb-fit-discount", "0.99"]),
    ("m5_prob_seed43_mixture_095", "run_prob",
     ["--dataset", "m5", "--seed", "43", "--init-ratio", "0.6667",
      "--hb-grouping", "mixture", "--hb-fit-discount", "0.95"]),
    ("m5_prob_seed44_mixture_095", "run_prob",
     ["--dataset", "m5", "--seed", "44", "--init-ratio", "0.6667",
      "--hb-grouping", "mixture", "--hb-fit-discount", "0.95"]),
]


def main() -> None:
    t0 = time.time()
    meta_runs = []
    for dirname, module, extra in RUNS:
        d = OUT / dirname
        cmd = ["py", "-u", "-m", f"experiments.{module}",
               "--baseline-mode", "hb_only", "--out", str(d)] + extra
        if module == "run_prob":
            cmd += ["--hb-calibration-mode", "none", "--hb-bootstrap-draws", "20"]
        tr = time.time()
        with open(OUT / f"{dirname}.log", "w", encoding="utf-8") as fh:
            subprocess.run(cmd, cwd=ROOT / "src", stdout=fh,
                           stderr=subprocess.STDOUT, check=True)
        wall = time.time() - tr
        meta_runs.append({"run": dirname, "wall_clock_s": round(wall, 1),
                          "cmd": " ".join(cmd)})
        print(f"[{dirname}] done in {wall:.0f}s", flush=True)

    # ---- collect summaries ----
    def spl_row(d):
        pin = pd.read_csv(OUT / d / "prob_pinball_scaled.csv")
        pin = pin[pin["model"] == "EB-Hurdle"]
        cov = pd.read_csv(OUT / d / "coverage_summary.csv")
        cov = cov[cov["model"] == "EB-Hurdle"].iloc[0]
        return {
            "run": d,
            "spl_mean": round(float(pin["scaled_pinball"].mean()), 4),
            "spl_q90": round(float(pin.loc[np.isclose(pin["quantile"].astype(float), 0.9),
                                           "scaled_pinball"].iloc[0]), 4),
            "coverage80": round(float(cov["Coverage@80"]), 3),
            "aiw80": round(float(cov["AIW@80"]), 2),
        }

    def point_row(d):
        pt = pd.read_csv(OUT / d / "point_metrics.csv")
        hb = pt[pt["model"] == "EB-Hurdle"].iloc[0]
        return {"run": d, "mae": round(float(hb["MAE"]), 4),
                "rmse": round(float(hb["RMSE"]), 4),
                "rmsse": round(float(hb["RMSSE"]), 4)}

    summary = []
    for dirname, module, _ in RUNS:
        summary.append(spl_row(dirname) if module == "run_prob" else point_row(dirname))
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "f234_summary.csv", index=False)
    print(sm.to_string(index=False))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "f234_driver", "commit": commit, "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__,
        "flags": "baseline-mode hb_only; prob adds calibration none + B=20",
        "wall_clock_total_s": round(time.time() - t0, 1),
        "runs": meta_runs,
    }
    (OUT / "f234_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote f234_summary.csv, f234_run_meta.json")


if __name__ == "__main__":
    main()
