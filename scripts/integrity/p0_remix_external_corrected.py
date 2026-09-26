"""Integrity P0: rerun the REMIX external evaluation under the corrected
(leakage-safe) selection outcomes.

Configurations (adjudicated 2026-08-06, from t1_leakage_safe_selection):
    Online Retail (global, 0.95)   M5 (mixture, 0.95)   Auto (global, 0.99)
    Carparts (global, 0.90)        RAF (mixture, 0.997)

Each panel runs the existing run_prob / run_point CLIs with the REMIX-only
baseline set and the recorded v2 release flags (`--baseline-mode hb_only`,
`--hb-calibration-mode none`, `--hb-bootstrap-draws 20`; template recorded in
paper_v2/REVIEW_ACTIONS.md). The mixture partition is relearned on the full
initialization window at w* by the existing pipeline, which is the paper's
stated protocol; nothing in the model or scoring is modified.

Outputs: outputs/integrity_<date>/<panel>_{prob,point}_remix_corrected/
plus remix_corrected_rows.csv (tab:prob / tab:point REMIX row values) and
p0_run_meta.json.

Usage: py scripts/integrity/p0_remix_external_corrected.py
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

CONFIG = {
    "online_retail": ("global", "0.95"),
    "m5": ("mixture", "0.95"),
    "auto": ("global", "0.99"),
    "carparts": ("global", "0.90"),
    "raf": ("mixture", "0.997"),
}
PROB_DATASET = {"online_retail": "online", "m5": "m5", "auto": "auto",
                "carparts": "carparts", "raf": "raf"}


def run(cmd: list[str], log: Path) -> float:
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        subprocess.run(cmd, cwd=ROOT / "src", stdout=fh, stderr=subprocess.STDOUT, check=True)
    return time.time() - t0


def main() -> None:
    collect_only = "--collect-only" in sys.argv
    t0_all = time.time()
    meta_runs = []
    for panel, (structure, w) in ([] if collect_only else list(CONFIG.items())):
        # --- probabilistic ---
        d = OUT / f"{panel}_prob_remix_corrected"
        cmd = [
            "py", "-u", "-m", "experiments.run_prob",
            "--dataset", PROB_DATASET[panel],
            "--baseline-mode", "hb_only",
            "--hb-grouping", structure,
            "--hb-fit-discount", w,
            "--hb-calibration-mode", "none",
            "--hb-bootstrap-draws", "20",
            "--out", str(d),
        ]
        if panel == "m5":
            cmd += ["--init-ratio", "0.6667"]
        wall = run(cmd, OUT / f"{panel}_prob_remix_corrected.log")
        meta_runs.append({"panel": panel, "kind": "prob", "structure": structure,
                          "discount": w, "wall_clock_s": round(wall, 1),
                          "cmd": " ".join(cmd)})
        print(f"[{panel}] prob done in {wall:.0f}s", flush=True)

        # --- point ---
        d = OUT / f"{panel}_point_remix_corrected"
        cmd = [
            "py", "-u", "-m", "experiments.run_point",
            "--dataset", panel if panel != "online_retail" else "online_retail",
            "--baseline-mode", "hb_only",
            "--hb-grouping", structure,
            "--hb-fit-discount", w,
            "--out", str(d),
        ]
        wall = run(cmd, OUT / f"{panel}_point_remix_corrected.log")
        meta_runs.append({"panel": panel, "kind": "point", "structure": structure,
                          "discount": w, "wall_clock_s": round(wall, 1),
                          "cmd": " ".join(cmd)})
        print(f"[{panel}] point done in {wall:.0f}s", flush=True)

    # --- collect REMIX rows ---
    rows = []
    for panel, (structure, w) in CONFIG.items():
        pin = pd.read_csv(OUT / f"{panel}_prob_remix_corrected" / "prob_pinball_scaled.csv")
        pin = pin[pin["model"] == "TSB-HB"]
        spl_mean = float(pin["scaled_pinball"].mean())
        q90 = float(pin.loc[np.isclose(pin["quantile"].astype(float), 0.9), "scaled_pinball"].iloc[0])
        cov = pd.read_csv(OUT / f"{panel}_prob_remix_corrected" / "coverage_summary.csv")
        cov = cov[cov["model"] == "TSB-HB"].iloc[0]
        pt_name = "point_metrics_m5.csv" if panel == "m5" else "point_metrics.csv"
        pt = pd.read_csv(OUT / f"{panel}_point_remix_corrected" / pt_name)
        hb = pt[pt["model"].str.contains("TSB-HB", case=False, na=False)]
        rows.append({
            "panel": panel, "structure": structure, "discount": w,
            "spl_mean": round(spl_mean, 4), "spl_q90": round(q90, 4),
            "coverage80": round(float(cov["Coverage@80"]), 3),
            "aiw80": round(float(cov["AIW@80"]), 2),
            "point_mae": round(float(hb["MAE"].iloc[0]), 4) if len(hb) else None,
            "point_rmsse": round(float(hb["RMSSE"].iloc[0]), 4) if len(hb) else None,
        })
    out_rows = pd.DataFrame(rows)
    out_rows.to_csv(OUT / "remix_corrected_rows.csv", index=False)
    print(out_rows.to_string(index=False))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "task": "p0_remix_external_corrected",
        "commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": 42,
        "flags": "baseline-mode hb_only, hb-calibration-mode none, hb-bootstrap-draws 20 (per paper_v2/REVIEW_ACTIONS.md template)",
        "wall_clock_total_s": round(time.time() - t0_all, 1),
        "runs": meta_runs or "executed in prior invocation; per-run wall-clocks in outputs/logs/integrity_p0.log",
    }
    (OUT / "p0_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote remix_corrected_rows.csv, p0_run_meta.json")


if __name__ == "__main__":
    main()
