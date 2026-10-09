"""W3: assemble the five-panel walk-forward evidence into one tidy table
and emit LaTeX rows for tab:wfprob and the appendix full tables.

Sources (each already scored by the same pipeline functions):
  online_retail  baselines  outputs/paper_rebuild/wf_online_retail.csv
                 ACI        outputs/paper_rebuild/adaptive_conformal_wf.csv
                 EBB        outputs/integrity_2026-08-07/wf_ebb_corrected.csv
  carparts       baselines  outputs/paper_rebuild/wf_carparts.csv
                 ACI        outputs/2026-09-05/wf_roster_carparts_aci.csv
                 EBB        outputs/integrity_2026-08-07/wf_ebb_corrected.csv
  auto, raf      full roster outputs/2026-09-05/wf_roster_<panel>.csv
  m5             plan c      outputs/2026-09-05/m5_walkforward.csv
  TweedieGP      outputs/<recent dates>/wf_tweediegp_<panel>.csv (W1)

Output: outputs/<date>/wf_all_panels.csv  + printed LaTeX rows.

Usage: py scripts/integrity/w3_assemble_wf_tables.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
RB = ROOT / "outputs" / "paper_rebuild"
I0807 = ROOT / "outputs" / "integrity_2026-08-07"
D0905 = ROOT / "outputs" / "2026-09-05"
PANELS = ["online_retail", "carparts", "auto", "raf", "m5"]
NAME = {"EB-Hurdle": "EBB", "CP-CrostonClassic": "CP-Croston", "CP-CrostonSBA": "CP-SBA",
        "Zero": "Zero", "TweedieGP": "TweedieGP"}
Q5 = ["0.1", "0.25", "0.5", "0.75", "0.9"]


def norm(df, panel):
    """-> columns panel, model, metric, quantile, value with unified names."""
    d = df.copy()
    if "dataset" not in d.columns:
        d["dataset"] = panel
    d = d.rename(columns={"dataset": "panel"})
    d["panel"] = panel
    d["quantile"] = d["quantile"].astype(str).replace({"NA": "", "nan": ""})
    d.loc[d["metric"] == "SPL_mean", "quantile"] = "mean"
    d["metric"] = d["metric"].replace({"SPL": "scaled_pinball", "SPL_mean": "scaled_pinball",
                                       "AIW80": "aiw80", "coverage80_marginal": "coverage80",
                                       "Coverage@80": "coverage80", "Cov+@80": "coverage80_positive",
                                       "AIW@80": "aiw80"})
    d["model"] = d["model"].map(lambda m: NAME.get(m, m))
    return d[["panel", "model", "metric", "quantile", "value"]]


def load_all():
    frames = []
    # --- Online Retail ---
    frames.append(norm(pd.read_csv(RB / "wf_online_retail.csv"), "online_retail"))
    frames.append(norm(pd.read_csv(RB / "adaptive_conformal_wf.csv"), "online_retail"))
    # --- Carparts ---
    frames.append(norm(pd.read_csv(RB / "wf_carparts.csv"), "carparts"))
    aci_cp = pd.read_csv(D0905 / "wf_roster_carparts_aci.csv")
    frames.append(norm(aci_cp[aci_cp["model"] == "ACI-ADIDA"], "carparts"))
    # --- EBB corrected rows for OR + Carparts (replace any EBB row above) ---
    ebb = pd.read_csv(I0807 / "wf_ebb_corrected.csv")
    ebb["model"] = "EBB"
    for p in ("online_retail", "carparts"):
        sub = ebb[ebb["dataset"] == p].copy()
        frames = [f[~((f["panel"] == p) & (f["model"] == "EBB"))] for f in frames]
        frames.append(norm(sub, p))
    # --- Auto / RAF full roster ---
    for p in ("auto", "raf"):
        frames.append(norm(pd.read_csv(D0905 / f"wf_roster_{p}.csv"), p))
    # --- M5 plan c ---
    frames.append(norm(pd.read_csv(D0905 / "m5_walkforward.csv"), "m5"))
    # --- TweedieGP walk-forward (search recent output dirs) ---
    for p in PANELS:
        for d in sorted((ROOT / "outputs").glob("2026-09-*"), reverse=True):
            f = d / f"wf_tweediegp_{p}.csv"
            if f.exists():
                frames.append(norm(pd.read_csv(f), p))
                break
    # --- ACI-EBB walk-forward (W8; keep only the ACI-EBB rows, EBB comes from above) ---
    for p in PANELS:
        for d in sorted((ROOT / "outputs").glob("2026-09-*"), reverse=True):
            f = d / f"wf_aci_ebb_{p}.csv"
            if f.exists():
                a = pd.read_csv(f)
                frames.append(norm(a[a["model"] == "ACI-EBB"], p))
                break
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["panel", "model", "metric", "quantile"], keep="last")
    # --- EBB-rule: calibration layer chosen per panel by the A1 validation-tail rule ---
    for d in sorted((ROOT / "outputs").glob("2026-09-*"), reverse=True):
        f = d / "aci_selector.csv"
        if f.exists():
            sel = pd.read_csv(f).set_index("panel")["apply_aci"].to_dict()
            parts = []
            for p, apply in sel.items():
                src = "ACI-EBB" if bool(apply) else "EBB"
                sub = all_df[(all_df["panel"] == p) & (all_df["model"] == src)].copy()
                sub["model"] = "EBB-rule"
                parts.append(sub)
            all_df = pd.concat([all_df] + parts, ignore_index=True)
            break
    return all_df


def main():
    df = load_all()
    df.to_csv(OUT / "wf_all_panels.csv", index=False)
    spl = df[df["metric"] == "scaled_pinball"]
    piv = spl.pivot_table(index=["panel", "model"], columns="quantile", values="value")
    other = df[df["metric"].isin(["coverage80", "coverage80_positive", "aiw80"])]
    piv2 = other.pivot_table(index=["panel", "model"], columns="metric", values="value")
    full = piv.join(piv2)
    cols = [c for c in Q5 + ["mean", "coverage80", "coverage80_positive", "aiw80"] if c in full.columns]
    full = full[cols]
    print("=== mean SPL by panel (sorted) ===")
    for p in PANELS:
        if p not in full.index.get_level_values(0):
            continue
        sub = full.loc[p].sort_values("mean")
        print(f"\n-- {p} --")
        print(sub.round(4).to_string())
    full.round(4).to_csv(OUT / "wf_all_panels_wide.csv")
    print(f"\nwrote wf_all_panels.csv / wf_all_panels_wide.csv")


if __name__ == "__main__":
    main()
