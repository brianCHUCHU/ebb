"""R2: held-out evaluation of the pre-fit pooling-room diagnostic on the
synthetic grid (docs/DESIGN_room_diagnostic.md).

Scores: R1 (closed form, no fitting) and R2 (logistic / linear on log1p
features, fitted on the training folds only). Folds: leave-one-level-out
over separation, discount, and T (each level held out in turn), plus random
5-fold. Metrics per fold family: Spearman, MAE (R2 regression), AUC and
precision/recall at Delta > 0.5% and > 1%.

Outputs (outputs/<date>/): room_eval_metrics.csv, room_eval_predictions.csv,
  fig15_room_diagnostic.{pdf,png}, r2_run_meta.json
Usage: py scripts/integrity/r2_room_eval.py [room_synthetic.csv path]
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
FIGS = ROOT / "paper_v2" / "figs_v3"
FEATS = ["lev_size", "het_size", "z_size", "lev_occ", "het_occ", "z_occ", "median_n_pos", "median_n_obs", "m_items"]
EPS = (0.5, 1.0)
INK, MUTED, ACCENT, BLUE = "#222222", "#9a9a9a", "#D55E00", "#0072B2"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def design(df):
    X = np.log1p(df[FEATS].clip(lower=0).to_numpy(float))
    return X


def folds(df):
    yield from ((f"sep={v:g}", df["separation"] != v, df["separation"] == v) for v in sorted(df["separation"].unique()))
    yield from ((f"w={v:g}", df["discount"] != v, df["discount"] == v) for v in sorted(df["discount"].unique()))
    yield from ((f"T={v}", df["T"] != v, df["T"] == v) for v in sorted(df["T"].unique()))
    rng = np.random.default_rng(0); k = rng.integers(0, 5, len(df))
    yield from ((f"random{i}", k != i, k == i) for i in range(5))


def pr_at_best_f1(y, s):
    p, r, _ = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f1))
    return float(p[i]), float(r[i]), float(f1[i])


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "room_synthetic.csv"
    subset = sys.argv[2] if len(sys.argv) > 2 else "all"      # "all" or "resolved" (global pool did not collapse)
    df = pd.read_csv(src).dropna(subset=["delta_oracle"]).reset_index(drop=True)
    if subset == "resolved":
        df = df[df["z_size"] > 0].reset_index(drop=True)
    tag = "" if subset == "all" else f"_{subset}"
    global FEATS
    if len(sys.argv) > 3 and sys.argv[3] == "sound":
        # amendment 2026-09-15: drop the unweighted S^2 / s-bar features (unstable under discounting)
        FEATS = ["lev_size", "tau2_global", "lev_occ", "phi_global", "median_n_pos", "median_n_obs", "m_items"]
        tag += "_sound"
    # R1b (amendment, 2026-09-15): the unweighted sampling-variance mean in R1 breaks under
    # discounting (items with n+ -> 0 dominate s-bar); the model's own weighted tau^2 is the
    # de-noised heterogeneity the design intended. Reported next to the pre-registered R1.
    df["R1b"] = df["lev_size"] ** 2 * df["tau2_global"]
    print(f"subset={subset}: {len(df)} panels")
    y = df["delta_oracle"].to_numpy(float)
    X = design(df)
    preds = pd.DataFrame({"idx": np.arange(len(df))})
    preds["R1"] = df["R1"].to_numpy(float)
    rows = []
    fam = lambda name: name.split("=")[0] if "=" in name else "random"
    store = {}
    for name, tr, te in folds(df):
        tr, te = np.asarray(tr, dtype=bool), np.asarray(te, dtype=bool)
        sc = StandardScaler().fit(X[tr])
        lin = LinearRegression().fit(sc.transform(X[tr]), y[tr])
        yhat = lin.predict(sc.transform(X[te]))
        rec = {"fold": name, "family": fam(name), "n_test": int(te.sum()),
               "spearman_R1": spearmanr(df.loc[te, "R1"], y[te]).correlation,
               "spearman_R1b": spearmanr(df.loc[te, "R1b"], y[te]).correlation,
               "spearman_R2": spearmanr(yhat, y[te]).correlation, "mae_R2": mean_absolute_error(y[te], yhat)}
        for eps in EPS:
            yb = (y[tr] > eps).astype(int); ybt = (y[te] > eps).astype(int)
            if ybt.min() == ybt.max() or yb.min() == yb.max():
                rec[f"auc_R1_{eps}"] = rec[f"auc_R2_{eps}"] = np.nan; continue
            logit = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), yb)
            pr = logit.predict_proba(sc.transform(X[te]))[:, 1]
            rec[f"auc_R1_{eps}"] = roc_auc_score(ybt, df.loc[te, "R1"])
            rec[f"auc_R1b_{eps}"] = roc_auc_score(ybt, df.loc[te, "R1b"])
            rec[f"auc_R2_{eps}"] = roc_auc_score(ybt, pr)
            rec[f"ap_R2_{eps}"] = average_precision_score(ybt, pr)
            rec[f"prec_R2_{eps}"], rec[f"rec_R2_{eps}"], rec[f"f1_R2_{eps}"] = pr_at_best_f1(ybt, pr)
            rec[f"base_rate_{eps}"] = float(ybt.mean())
            if fam(name) == "random":
                store.setdefault(("p", eps), np.full(len(df), np.nan))[te] = pr
        if fam(name) == "random":
            store.setdefault("yhat", np.full(len(df), np.nan))[te] = yhat
        rows.append(rec)
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"room_eval_metrics{tag}.csv", index=False)
    summary = met.groupby("family").agg({c: "mean" for c in met.columns if c not in ("fold", "family", "n_test")}).round(3)
    print(summary.T.to_string())
    for eps in EPS:
        preds[f"p_R2_{eps}"] = store.get(("p", eps))
    preds["yhat_R2"] = store.get("yhat"); preds["delta"] = y
    pd.concat([df[["separation", "discount", "T", "occ_mean", "items", "balance", "rep"]], preds], axis=1).to_csv(OUT / f"room_eval_predictions{tag}.csv", index=False)

    # ---- figure: R1 (closed form) vs oracle gain, colored by separation; inset: AUC by held-out family ----
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    wcol = {1.0: BLUE, 0.95: "#009E73", 0.9: ACCENT}
    res = df["z_size"] > 0
    for wv, c in wcol.items():
        d = df[(df["discount"] == wv) & res]
        ax.scatter(d["R1b"], d["delta_oracle"], s=9, color=c, alpha=0.55, label=f"$w={wv:g}$", edgecolor="none")
        d = df[(df["discount"] == wv) & ~res]
        ax.scatter(d["R1b"], d["delta_oracle"], s=9, facecolor="none", edgecolor=c, linewidth=0.5, alpha=0.5)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.axhline(1, color=INK, lw=0.6, ls=(0, (3, 3)))
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_ylim(-8, 8)
    ax.set_xlabel(r"closed-form room score $R_1'=\mathrm{med}(1-\lambda^{(+)})^2\,\hat\tau^2$ (single-pool fit)")
    ax.set_ylabel("oracle-partition gain (% SPL)")
    ax.legend(frameon=False, ncol=3, fontsize=6.5, loc="upper left", title="filled: single pool resolved; open: collapsed", title_fontsize=6.2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax = axes[1]
    fams = ["sep", "w", "T", "random"]
    xs = np.arange(len(fams))
    for j, (col, lab, c) in enumerate((("auc_R1b_1.0", r"$R_1'$, $\Delta>1\%$", MUTED), ("auc_R2_1.0", r"$R_2$, $\Delta>1\%$", ACCENT),
                                       ("auc_R2_0.5", r"$R_2$, $\Delta>0.5\%$", BLUE))):
        vals = [met[met["family"] == f][col].mean() for f in fams]
        ax.bar(xs + (j - 1) * 0.26, vals, width=0.26, color=c, label=lab)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.set_xticks(xs); ax.set_xticklabels(["hold out\nseparation", "hold out\n$w$", "hold out\n$T$", "random\n5-fold"])
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("held-out AUC")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"fig15_room_diagnostic{tag}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"fig15_room_diagnostic{tag}.png", bbox_inches="tight", dpi=200)
    for d in ("v5", "v5_paper", "v6_paper"):
        if (ROOT / "paper_v2" / d / "figs").exists():
            shutil.copy(FIGS / f"fig15_room_diagnostic{tag}.pdf", ROOT / "paper_v2" / d / "figs" / f"fig15_room_diagnostic{tag}.pdf")
    meta = {"task": "r2_room_eval", "design": "docs/DESIGN_room_diagnostic.md", "source": str(src), "n_panels": int(len(df)),
            "features": FEATS, "platform": platform.platform(), "python": sys.version.split()[0]}
    (OUT / "r2_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote room_eval_metrics.csv, room_eval_predictions.csv, fig15_room_diagnostic")


if __name__ == "__main__":
    main()
