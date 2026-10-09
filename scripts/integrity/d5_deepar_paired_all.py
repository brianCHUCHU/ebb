"""D5: paired per-series tests of DeepAR against EBB and against TweedieGP under both protocols,
from stored quantiles (docs/DESIGN_deepar.md).

Per-series loss is the mean scaled pinball loss over the evaluation rows and the five quantiles,
with the scale taken from the initialization window (the same scale as the tables). Tests: paired
t-test and Wilcoxon signed-rank on the per-series differences.

Output: outputs/deepar/deepar_paired_all.csv
Usage : py scripts/integrity/d5_deepar_paired_all.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import fit_eb_hurdle, predict_eb_hurdle

spec = importlib.util.spec_from_file_location("c1", ROOT / "scripts" / "integrity" / "c1_coldstart.py")
c1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c1)

O = ROOT / "outputs"
DA = O / "deepar"
Q = c1.Q5
QC = c1.QCOLS
PANELS = ["online_retail", "m5", "auto", "carparts", "raf"]

WF_EBB = {"online_retail": O / "integrity_2026-08-07" / "wf_ebb_quantiles_online_retail.csv",
          "carparts": O / "integrity_2026-08-07" / "wf_ebb_quantiles_carparts.csv",
          "auto": O / "2026-09-05" / "wf_roster_quantiles_auto.csv",
          "raf": O / "2026-09-05" / "wf_roster_quantiles_raf.csv",
          "m5": O / "2026-09-05" / "m5_wf_quantiles.csv"}
WF_TG = {p: O / "2026-09-06" / f"wf_tweediegp_quantiles_{p}.csv" for p in ["online_retail", "auto", "carparts", "raf"]}
FX_TG = {"auto": O / "paper_rebuild" / "tweediegp_quantiles_auto.csv",
         "carparts": O / "paper_rebuild" / "tweediegp_quantiles_carparts.csv",
         "raf": O / "paper_rebuild" / "tweediegp_quantiles_raf.csv"}


def load_q(path: Path, model: str | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    head = pd.read_csv(path, nrows=1)
    use = [c for c in ["model", "unique_id", "ds"] + QC if c in head.columns]
    q = pd.read_csv(path, usecols=use, dtype={"unique_id": str}, low_memory=False)
    if model is not None and "model" in q.columns:
        q = q[q["model"] == model]
    q = q.drop(columns=[c for c in ["model"] if c in q.columns]).copy()
    q["ds"] = pd.to_datetime(q["ds"])
    for c in QC:
        q[c] = np.maximum(q[c].astype(float), 0.0)
    return c1._enforce_monotonic_quantiles(q, quantiles=Q)


def per_series(pred, ev, init):
    scale = init.groupby("unique_id")["y"].apply(lambda s: float(np.mean(np.abs(s)))).replace(0.0, np.nan)
    scale = scale.fillna(float(np.nanmedian(scale.to_numpy())))
    m = ev[["unique_id", "ds", "y"]].merge(pred[["unique_id", "ds"] + QC], on=["unique_id", "ds"], how="inner")
    loss = np.zeros(len(m))
    for q in Q:
        e = m["y"].to_numpy(float) - m[f"q_{q}"].to_numpy(float)
        loss += np.maximum(q * e, (q - 1) * e)
    m["l"] = loss / len(Q) / m["unique_id"].map(scale).to_numpy(float)
    return m.groupby("unique_id")["l"].mean(), float(m["l"].mean()), len(m)


def test(a, b):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = j["a"] - j["b"]
    t, pt = stats.ttest_1samp(d, 0.0)
    try:
        _, pw = stats.wilcoxon(d[d != 0])
    except ValueError:
        pw = np.nan
    return len(j), float(d.mean()), float(t), float(pt), float(pw)


def ebb_fixed(panel, init, ev):
    structure, w, _ = c1.CONFIG[panel]
    labels = mixture_group_labels(init, k=0, fit_discount=w).labels if structure == "mixture" else None
    if labels is not None:
        labels.index = labels.index.astype(str)
    prm = fit_eb_hurdle(init, group_labels=labels, item_variance_mode="conjugate", item_variance_shrink_strength=20.0,
                     fit_discount=w, bootstrap_draws=20, bootstrap_seed=42)
    e = predict_eb_hurdle(prm, ev, quantiles=Q, include_hyper_uncertainty=True)
    e["unique_id"] = e["unique_id"].astype(str); e["ds"] = pd.to_datetime(e["ds"])
    return c1._enforce_monotonic_quantiles(e, quantiles=Q)


def main() -> None:
    rows = []
    for p in PANELS:
        init = pd.read_csv(DA / f"panel_{p}_init.csv", dtype={"unique_id": str})
        ev = pd.read_csv(DA / f"panel_{p}_eval.csv", dtype={"unique_id": str})
        for d in (init, ev):
            d["ds"] = pd.to_datetime(d["ds"])
        for protocol in ("fixed", "wf"):
            dq = load_q(DA / f"deepar_{protocol}_{p}.csv")
            if dq is None:
                continue
            sd, md, nd = per_series(dq, ev, init)
            others = {}
            if protocol == "fixed":
                others["EBB"] = ebb_fixed(p, init, ev)
                if p in FX_TG:
                    others["TweedieGP"] = load_q(FX_TG[p])
            else:
                others["EBB"] = load_q(WF_EBB[p], "EB-Hurdle")
                if p in WF_TG:
                    others["TweedieGP"] = load_q(WF_TG[p], "TweedieGP")
            for name, oq in others.items():
                if oq is None or oq.empty:
                    continue
                so, mo, no = per_series(oq, ev, init)
                n, dm, t, pt, pw = test(sd, so)
                rows.append({"protocol": protocol, "panel": p, "other": name, "rows_deepar": nd, "rows_other": no,
                             "n_series": n, "deepar_mean_loss": round(md, 4), "other_mean_loss": round(mo, 4),
                             "mean_diff_deepar_minus_other": dm, "t": t, "p_t": pt, "p_wilcoxon": pw,
                             "deepar_better": dm < 0, "significant_5pct": pt < 0.05})
                print(f"[d5] {protocol:5s} {p:14s} DeepAR vs {name:9s}: diff {dm:+.5f} t={t:+.2f} p={pt:.3g} "
                      f"(rows {nd}/{no})", flush=True)
            pd.DataFrame(rows).to_csv(DA / "deepar_paired_all.csv", index=False)


if __name__ == "__main__":
    main()
