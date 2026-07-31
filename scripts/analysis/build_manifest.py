"""Build outputs/aistats2027/MANIFEST.md — the master experiment record.

Scans every run directory under outputs/aistats2027, records which artifact
files exist, and extracts headline metrics (best TSB-HB row + best baseline)
so the paper's numbers are traceable to a specific file.

Usage: py scripts/analysis/build_manifest.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "aistats2027"

EXPECTED_POINT = ["point_metrics.csv", "point_metrics_m5.csv"]
ARTIFACTS = [
    "point_metrics.csv",
    "point_metrics_m5.csv",
    "tsbhb_variant_diagnostics.csv",
    "point_predictions.csv.gz",
    "paired_tests.csv",
    "prob_metrics.csv",
    "prob_pinball.csv",
    "prob_pinball_scaled.csv",
    "coverage_summary.csv",
    "hb_calibration_params.csv",
    "point_slice_metrics.csv",
    "segmentation_rmsse.csv",
]


def summarize_point(path: Path) -> list[str]:
    df = pd.read_csv(path)
    lines = []
    for metric in ["MAE", "RMSE", "RMSSE"]:
        if metric not in df.columns:
            continue
        best = df.loc[df[metric].idxmin()]
        lines.append(f"  - best {metric}: **{best['model']}** = {best[metric]:.4f}")
    return lines


def summarize_prob(run_dir: Path) -> list[str]:
    lines = []
    spl_path = run_dir / "prob_pinball_scaled.csv"
    if spl_path.exists():
        spl = pd.read_csv(spl_path)
        mean_rows = spl[spl["quantile"].astype(str) == "mean"]
        if not mean_rows.empty:
            best = mean_rows.loc[mean_rows["scaled_pinball"].idxmin()]
            lines.append(
                f"  - best SPL mean: **{best['model']}** = {best['scaled_pinball']:.4f} "
                f"(n_models={mean_rows['model'].nunique()})"
            )
    cov_path = run_dir / "prob_metrics.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        if "pinball_mean" in cov.columns:
            best = cov.loc[cov["pinball_mean"].idxmin()]
            lines.append(f"  - best pinball mean: **{best['model']}** = {best['pinball_mean']:.4f}")
    return lines


def main() -> None:
    lines = [
        "# AISTATS 2027 Experiment Manifest",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Every table in `paper_v2/main.tex` maps to one run directory below.",
        "",
    ]
    for run_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        lines.append(f"## {run_dir.name}")
        present = [a for a in ARTIFACTS if (run_dir / a).exists()]
        missing_core = []
        if not any((run_dir / a).exists() for a in EXPECTED_POINT) and not (run_dir / "prob_metrics.csv").exists():
            missing_core.append("no metrics file — run incomplete?")
        lines.append(f"- artifacts: {', '.join(present) if present else '(none)'}")
        for warn in missing_core:
            lines.append(f"- **WARNING**: {warn}")
        for a in EXPECTED_POINT:
            if (run_dir / a).exists():
                lines.extend(summarize_point(run_dir / a))
        lines.extend(summarize_prob(run_dir))
        lines.append("")

    out_path = OUT / "MANIFEST.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
