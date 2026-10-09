"""H5: redraw the two appendix point-forecast figures (horizon decomposition, average-rank
diagram) under the audited configurations and the current method name.

The published figures were drawn from per-timestamp predictions of the superseded selection
outcome and carry the legend "EBB". The audited external runs did not save per-timestamp
predictions, but the point forecast is horizon-invariant, y_hat_i = pi_i * exp(mu_i + sigma_i^2/2),
so it is recomputed here from one fit per panel at the audited (structure, discount) and checked
against the audited point MAE (outputs/integrity_2026-08-06/ebb_corrected_rows.csv) before use.
Baseline predictions are the saved ones (outputs/paper_runs/*_point_fixed*).

Outputs: paper_v2/v8_paper_runs/figs/horizon_decomp.pdf, cd_diagram.pdf;
         outputs/<date>/point_figs_audited_check.csv, point_rank_audited.csv
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import fit_eb_hurdle

spec = importlib.util.spec_from_file_location("c1", ROOT / "scripts" / "integrity" / "c1_coldstart.py")
c1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c1)
spec2 = importlib.util.spec_from_file_location("hcd", ROOT / "scripts" / "analysis" / "horizon_and_cd.py")
hcd = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(hcd)

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
FIGS = ROOT / "paper_v2" / "v8_paper_runs" / "figs"
OLD = ROOT / "outputs" / "paper_runs"
PANELS = {"Online Retail": "online_retail", "M5": "m5", "Auto": "auto", "Carparts": "carparts", "RAF": "raf"}
NAME = "EBB"


def ebb_point(panel: str) -> tuple[pd.Series, pd.DataFrame]:
    structure, w, _ = c1.CONFIG[panel]
    init, ev = c1.load_panel(panel)
    init = init.copy(); init["unique_id"] = init["unique_id"].astype(str)
    ev = ev.copy(); ev["unique_id"] = ev["unique_id"].astype(str); ev["ds"] = pd.to_datetime(ev["ds"])
    labels = mixture_group_labels(init, k=0, fit_discount=w).labels if structure == "mixture" else None
    if labels is not None:
        labels.index = labels.index.astype(str)
    p = fit_eb_hurdle(init, group_labels=labels, item_variance_mode="conjugate",
                   item_variance_shrink_strength=20.0, fit_discount=w, bootstrap_draws=0, bootstrap_seed=42)
    yhat = (p.p_posterior * np.exp(p.shrunk_mean_log + 0.5 * p.sigma_sq_process)).astype(float)
    yhat.index = yhat.index.astype(str)
    return yhat, ev


def main() -> None:
    audited = pd.read_csv(ROOT / "outputs" / "integrity_2026-08-06" / "ebb_corrected_rows.csv").set_index("panel")
    checks, rank_blocks, joints = [], [], {}
    for ds, panel in PANELS.items():
        yhat, ev = ebb_point(panel)
        m = ev[["unique_id", "ds", "y"]].copy()
        m[NAME] = m["unique_id"].map(yhat)
        mae = float((m[NAME] - m["y"]).abs().mean())
        ref = float(audited.loc[panel, "point_mae"])
        ok = abs(mae - ref) < 5e-4
        checks.append({"panel": panel, "mae_recomputed": round(mae, 4), "mae_audited": ref, "match": ok,
                       "n_missing": int(m[NAME].isna().sum())})
        print(f"[h5] {panel}: MAE recomputed {mae:.4f} vs audited {ref:.4f} -> {'OK' if ok else 'MISMATCH'}",
              flush=True)
        base_dir, _ = hcd.PRED_SOURCES[ds]
        bp = OLD / base_dir / "point_predictions.csv.gz"
        if not bp.exists() or not ok:
            continue
        base = pd.read_csv(bp, dtype={"unique_id": str})
        base["ds"] = pd.to_datetime(base["ds"])
        keep = ["unique_id", "ds", "y"] + [c for c in hcd.MODELS if c in base.columns]
        joint = base[keep].merge(m[["unique_id", "ds", NAME]], on=["unique_id", "ds"], how="inner")
        joints[ds] = joint
        denom = hcd.load_init_naive_mse(ds)
        denom.index = denom.index.astype(str)
        ps = hcd.per_series_scaled_sqerr(joint, [c for c in hcd.MODELS + [NAME] if c in joint.columns], denom)
        rank_blocks.append(ps.rank(axis=1))
        print(f"[h5] {ds}: ranked over {len(ps)} series", flush=True)
    pd.DataFrame(checks).to_csv(OUT / "point_figs_audited_check.csv", index=False)

    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "legend.frameon": False, "pdf.fonttype": 42})
    C = hcd.C
    # ---- horizon figure (Online Retail) ----
    if "Online Retail" in joints:
        j = joints["Online Retail"].sort_values(["unique_id", "ds"]).copy()
        j["h"] = j.groupby("unique_id").cumcount() + 1
        j["bucket"] = pd.cut(j["h"], bins=[0, 7, 14, 28, 56, 112, 10**9],
                             labels=["1-7", "8-14", "15-28", "29-56", "57-112", "113+"])
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        show = [(NAME, NAME, C[1], "-", 2.0), ("TSB", "TSB", C[0], "--", 1.2), ("ADIDA", "ADIDA", C[2], "--", 1.2),
                ("IMAPA", "IMAPA", C[5], ":", 1.2), ("AutoARIMA", "AutoARIMA", C[3], ":", 1.2)]
        rows = {}
        for col, label, color, ls, lw in show:
            g = j.groupby("bucket", observed=True).apply(
                lambda d: float(np.sqrt(((d[col] - d.y) ** 2).mean())), include_groups=False)
            rows[label] = g
            ax.plot(range(len(g)), g.values, ls, marker="o", ms=2.5, lw=lw, color=color, label=label)
            ax.set_xticks(range(len(g))); ax.set_xticklabels(list(g.index), fontsize=7)
        ax.set_xlabel("forecast horizon bucket (days ahead)"); ax.set_ylabel("RMSE")
        ax.legend(fontsize=7, ncol=2, handlelength=1.5)
        fig.tight_layout(); fig.savefig(FIGS / "horizon_decomp.pdf", bbox_inches="tight"); plt.close(fig)
        hz = pd.DataFrame(rows)
        hz.to_csv(OUT / "horizon_rmse_audited.csv")
        print(hz.round(3).to_string())
        print("[h5] EBB lowest in every bucket:", bool((hz.idxmin(axis=1) == NAME).all()))
        mae_b = j.groupby("bucket", observed=True).apply(
            lambda d: pd.Series({c: float((d[c] - d.y).abs().mean()) for c, *_ in show}), include_groups=False)
        print(mae_b.round(3).to_string())
    # ---- average-rank diagram ----
    if rank_blocks:
        ranks = pd.concat(rank_blocks, ignore_index=True).dropna()
        avg = ranks.mean(axis=0).sort_values()
        k, n = ranks.shape[1], len(ranks)
        _, fp = stats.friedmanchisquare(*[ranks[c].values for c in ranks.columns])
        q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}[k]
        cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
        fig, ax = plt.subplots(figsize=(3.6, 1.9))
        y = 0
        for mname, r in avg.items():
            color = C[1] if mname == NAME else "#444444"
            ax.plot([r], [y], "o", ms=5, color=color)
            ax.annotate(f" {mname} ({r:.2f})", (r, y), fontsize=7, va="center", color=color)
            y -= 1
        ax.errorbar([avg.min()], [y], xerr=[[0], [cd]], fmt="none", capsize=3, color="#000000")
        ax.annotate(f" CD={cd:.2f} (Nemenyi, $\\alpha$=0.05)", (avg.min(), y), fontsize=7, va="center")
        ax.set_yticks([])
        ax.set_xlabel(f"avg. rank of per-series scaled sq. error ({n} series, {len(rank_blocks)} datasets; "
                      f"Friedman p={fp:.2g})", fontsize=7)
        fig.tight_layout(); fig.savefig(FIGS / "cd_diagram.pdf", bbox_inches="tight"); plt.close(fig)
        avg.round(4).to_csv(OUT / "point_rank_audited.csv")
        print(avg.round(3).to_string()); print(f"[h5] n_series={n} datasets={len(rank_blocks)} CD={cd:.3f}")


if __name__ == "__main__":
    main()
