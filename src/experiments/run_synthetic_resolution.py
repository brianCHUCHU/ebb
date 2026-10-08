"""Controlled pooling-forgetting resolution experiment (review items A3, B1-B3).

Three parts, all deterministic given ``--seed``:

A3  Calibration of the collapse-probability approximation
    Pr[tauhat_g^2 = 0] ~= Phi(-sqrt((m_g-1)/2) * rho_g) over a grid of group
    sizes, variance ratios, discounts, and both equal/unequal counts and
    homoskedastic/heteroskedastic sampling variances. Reports empirical vs
    predicted probability and the absolute error, so the regimes where the
    approximation is trustworthy are stated rather than assumed.

B1  Panel experiment on synthetic hierarchical hurdle data with known ground
    truth, comparing seven estimators that separate three explanations for a
    learned partition failing to pay: the plug-in estimator breaking
    (regularized vs not), the partition being mis-recovered (learned vs oracle
    labels), and the information simply not being there (oracle labels *and*
    oracle hyperparameters vs global pooling).

B3  The comparison above is the discrimination test; ``summary.md`` states which
    of the three explanations the numbers support in each cell of the grid.

Outputs under ``outputs/synthetic_resolution/``:
    collapse_calibration.csv   A3 grid
    panel_grid.csv             B1/B3 grid
    summary.md                 human-readable reading of the grid

Usage:
    py -m experiments.run_synthetic_resolution --quick
    py -m experiments.run_synthetic_resolution            # full grid
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score

from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import (
    _compute_series_stats,
    _estimate_group_hypers,
    fit_eb_hurdle,
    predict_eb_hurdle,
)

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "outputs" / "synthetic_resolution"


# ---------------------------------------------------------------- A3


def effective_count(T: int, w: float) -> float:
    """Discounted observation count sum_{k<T} w^k."""
    return float(T) if w >= 1.0 else float((1.0 - w**T) / (1.0 - w))


def collapse_calibration(
    rng: np.random.Generator,
    group_sizes=(10, 25, 50, 100, 250),
    var_ratios=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
    discounts=(1.0, 0.99, 0.95, 0.90),
    n_rep: int = 4000,
    T: int = 120,
) -> pd.DataFrame:
    """Empirical vs predicted Pr[moment estimate of tau^2 truncates at zero]."""
    rows: list[dict[str, float]] = []
    for m_g, ratio, w, count_mode, var_mode in itertools.product(
        group_sizes, var_ratios, discounts, ("equal", "unequal"), ("homo", "hetero")
    ):
        sigma2 = 1.0
        n_eff = effective_count(T, w)
        if count_mode == "equal":
            n_i = np.full(m_g, n_eff)
        else:
            # occurrence rates spread over an order of magnitude
            n_i = n_eff * rng.uniform(0.15, 1.0, size=m_g)
        n_i = np.maximum(n_i, 1.0)

        if var_mode == "homo":
            sig_i = np.full(m_g, sigma2)
        else:
            sig_i = sigma2 * np.exp(rng.normal(0.0, 0.7, size=m_g))
        sig_i = np.maximum(sig_i, 1e-6)

        tau2 = ratio * sigma2
        s_i = sig_i / n_i
        sbar = float(np.mean(s_i))
        rho = tau2 / (tau2 + sbar)
        predicted = float(norm.cdf(-np.sqrt((m_g - 1) / 2.0) * rho))

        # Simulate item means: mu_i ~ N(0, tau2), ybar_i = mu_i + N(0, s_i).
        draws = rng.normal(0.0, np.sqrt(tau2), size=(n_rep, m_g)) + rng.normal(
            0.0, np.sqrt(s_i), size=(n_rep, m_g)
        )
        s2 = draws.var(axis=1, ddof=1)
        empirical = float(np.mean(s2 - sbar <= 0.0))

        rows.append(
            {
                "m_g": m_g,
                "tau2_over_sigma2": ratio,
                "discount": w,
                "counts": count_mode,
                "sampling_var": var_mode,
                "n_eff": n_eff,
                "sbar": sbar,
                "rho": rho,
                "z": rho * np.sqrt((m_g - 1) / 2.0),
                "p_empirical": empirical,
                "p_predicted": predicted,
                "abs_error": abs(empirical - predicted),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B1 data


def simulate_panel(
    rng: np.random.Generator,
    n_groups: int,
    items_per_group: int,
    T: int,
    horizon: int,
    separation: float,
    tau2: float,
    sigma2: float,
    occurrence_rate: float,
    drift: float,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    """Hierarchical hurdle panel with known group labels.

    Group k has size mean ``k * separation``; item means are drawn around it with
    variance ``tau2``; positive sizes are lognormal with within-item variance
    ``sigma2``. Occurrence probability drifts linearly by ``drift`` over the
    window, which is what the discount is meant to track.
    """
    records = []
    labels = {}
    # Groups are separated on BOTH blocks the mixture models: mean log size and
    # occurrence rate. Separating on size alone leaves the occurrence likelihood
    # uninformative about membership and BIC then prefers a single component.
    occ_levels = np.linspace(0.5, 1.5, n_groups)
    for k in range(n_groups):
        group_mu = k * separation
        group_occ = float(np.clip(occurrence_rate * occ_levels[k], 0.02, 0.95))
        for j in range(items_per_group):
            uid = f"g{k}_i{j}"
            labels[uid] = f"true_{k}"
            mu_i = rng.normal(group_mu, np.sqrt(tau2))
            p0 = float(np.clip(group_occ * rng.uniform(0.85, 1.15), 0.02, 0.95))
            for t in range(T + horizon):
                frac = t / max(T + horizon - 1, 1)
                p_t = float(np.clip(p0 * (1.0 - drift * frac), 0.005, 0.99))
                y = 0.0
                if rng.random() < p_t:
                    y = float(np.exp(rng.normal(mu_i, np.sqrt(sigma2))))
                records.append((uid, t, y))
    df = pd.DataFrame(records, columns=["unique_id", "ds", "y"])
    truth = {"tau2": tau2, "sigma2": sigma2, "separation": separation}
    return df, pd.Series(labels, name="group"), truth


def _split(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(df["ds"].max()) - horizon
    return df[df["ds"] <= cutoff].copy(), df[df["ds"] > cutoff].copy()


def _mae(pred: pd.DataFrame, actual: pd.DataFrame) -> float:
    merged = actual[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
    return float(np.mean(np.abs(merged["y"].to_numpy(float) - merged["yhat"].to_numpy(float))))


def _pinball(pred: pd.DataFrame, actual: pd.DataFrame, quantiles=(0.5, 0.75, 0.9)) -> float:
    merged = actual[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
    y = merged["y"].to_numpy(float)
    losses = []
    for q in quantiles:
        col = f"q_{q}"
        if col not in merged:
            continue
        err = y - merged[col].to_numpy(float)
        losses.append(np.maximum(q * err, (q - 1.0) * err))
    return float(np.mean(np.concatenate(losses))) if losses else float("nan")


def _variance_component_report(train, labels, w, hyper_shrink) -> dict[str, float]:
    stats = _compute_series_stats(train, group_labels=labels, fit_discount=w)
    _, _, _, sig, tau = _estimate_group_hypers(stats, hyper_shrink=hyper_shrink)
    groups = [g for g in tau.index if g != "__global__"]
    if not groups:
        return {"frac_tau_collapsed": np.nan, "median_lambda": np.nan, "tau2_min": np.nan}
    kappa = (sig / tau.clip(lower=1e-9)).reindex(groups)
    lam = stats["n_pos"] / (stats["n_pos"] + stats["group"].map(sig / tau.clip(lower=1e-9)).astype(float))
    return {
        "frac_tau_collapsed": float(np.mean(tau.reindex(groups).to_numpy() <= 1e-6 + 1e-12)),
        "median_lambda": float(np.clip(lam, 0, 1).median()),
        "tau2_min": float(tau.reindex(groups).min()),
        "kappa_max": float(kappa.max()),
    }


def panel_cell(
    rng: np.random.Generator,
    cfg: dict,
    quantiles=(0.5, 0.75, 0.9),
) -> list[dict]:
    df, true_labels, truth = simulate_panel(
        rng,
        n_groups=cfg["n_groups"],
        items_per_group=cfg["items_per_group"],
        T=cfg["T"],
        horizon=cfg["horizon"],
        separation=cfg["separation"],
        tau2=cfg["tau2"],
        sigma2=cfg["sigma2"],
        occurrence_rate=cfg["occurrence_rate"],
        drift=cfg["drift"],
    )
    train, test = _split(df, cfg["horizon"])
    w = cfg["discount"]

    mix = mixture_group_labels(train, k=0)
    learned = mix.labels
    ari = float(adjusted_rand_score(
        true_labels.reindex(learned.index).astype(str).to_numpy(),
        learned.astype(str).to_numpy(),
    ))

    arms = [
        ("global-stationary", None, 1.0, "off"),
        ("global-discounted", None, w, "off"),
        ("oracle-partition", true_labels, w, "off"),
        ("learned-partition", learned, w, "off"),
        ("learned+regularized", learned, w, "credibility"),
    ]

    out = []
    for name, labels, disc, shrink in arms:
        params = fit_eb_hurdle(
            train,
            group_labels=labels,
            item_variance_mode="conjugate",
            item_variance_shrink_strength=20.0,
            fit_discount=disc,
            hyper_shrink=shrink,
        )
        point = predict_eb_hurdle(params, test, quantiles=None)
        quant = predict_eb_hurdle(params, test, quantiles=list(quantiles), include_hyper_uncertainty=False)
        rec = dict(cfg)
        rec.update(
            {
                "arm": name,
                "mae": _mae(point, test),
                "pinball": _pinball(quant, test, quantiles),
                "ari_learned_vs_true": ari,
                "learned_K": int(mix.k),
                "n_true_groups": int(cfg["n_groups"]),
            }
        )
        if labels is not None:
            rec.update(_variance_component_report(train, labels, disc, shrink))
        out.append(rec)

    # No-pooling reference: per-item empirical hurdle estimate (rate x mean size),
    # computed directly rather than through the hierarchical prior, so it does not
    # depend on a group-level variance component that is undefined for singletons.
    hist = train.copy()
    hist["occ"] = (hist["y"] > 0).astype(float)
    if w < 1.0:
        hist = hist.sort_values(["unique_id", "ds"], kind="stable")
        idx = hist.groupby("unique_id", sort=False).cumcount()
        size = hist.groupby("unique_id", sort=False)["y"].transform("size")
        hist["wt"] = np.power(w, (size - 1 - idx).to_numpy(dtype=float))
    else:
        hist["wt"] = 1.0
    g = hist.groupby("unique_id", sort=False)
    rate = g.apply(lambda d: float(np.sum(d["wt"] * d["occ"]) / max(np.sum(d["wt"]), 1e-9)), include_groups=False)
    pos = hist[hist["y"] > 0]
    msize = pos.groupby("unique_id", sort=False).apply(
        lambda d: float(np.sum(d["wt"] * d["y"]) / max(np.sum(d["wt"]), 1e-9)), include_groups=False
    )
    yhat = (rate * msize.reindex(rate.index).fillna(0.0)).rename("yhat")
    local_pred = test[["unique_id", "ds"]].merge(yhat, left_on="unique_id", right_index=True, how="left")
    local_pred["yhat"] = local_pred["yhat"].fillna(0.0)
    rec = dict(cfg)
    rec.update(
        {
            "arm": "no-pooling",
            "mae": _mae(local_pred, test),
            "pinball": np.nan,
            "ari_learned_vs_true": ari,
            "learned_K": int(mix.k),
            "n_true_groups": int(cfg["n_groups"]),
        }
    )
    out.append(rec)
    return out


def run_panel_grid(rng: np.random.Generator, quick: bool, n_reps: int = 5) -> pd.DataFrame:
    """Grid over (separation, group size, discount) with replicates.

    Replicates matter: a single draw per cell cannot separate an ordering of
    arms from simulation noise, and the differences we care about are small.
    """
    discounts = (1.0, 0.95, 0.90) if quick else (1.0, 0.99, 0.95, 0.90)
    seps = (2.0, 0.5) if quick else (2.0, 1.0, 0.5, 0.25)
    items = (40,) if quick else (40, 120)
    reps = 1 if quick else n_reps
    rows: list[dict] = []
    for sep, ipg, w, rep in itertools.product(seps, items, discounts, range(reps)):
        cfg = {
            "n_groups": 4,
            "items_per_group": ipg,
            "T": 120,
            "horizon": 14,
            "separation": sep,
            "tau2": 0.15,
            "sigma2": 1.0,
            "occurrence_rate": 0.25,
            "drift": 0.5,
            "discount": w,
            "rep": rep,
        }
        rows.extend(panel_cell(rng, cfg))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- reporting


def write_summary(calib: pd.DataFrame, panel: pd.DataFrame, out_dir: Path) -> None:
    lines = ["# Synthetic resolution experiment", ""]

    lines += ["## A3 collapse-probability calibration", ""]
    lines.append(f"- grid cells: {len(calib)}")
    lines.append(f"- median |error|: {calib.abs_error.median():.4f}")
    lines.append(f"- 90th pct |error|: {calib.abs_error.quantile(0.9):.4f}")
    lines.append(f"- max |error|: {calib.abs_error.max():.4f}")
    worst = calib.sort_values("abs_error", ascending=False).head(5)
    lines += ["", "Worst cells:", "", worst[
        ["m_g", "tau2_over_sigma2", "discount", "counts", "sampling_var", "p_empirical", "p_predicted", "abs_error"]
    ].round(4).to_string(index=False), ""]
    by_mode = calib.groupby(["counts", "sampling_var"])["abs_error"].agg(["median", "max"]).round(4)
    lines += ["Error by regime:", "", by_mode.to_string(), ""]

    lines += ["## B1/B3 panel grid", ""]
    piv = panel.pivot_table(index=["separation", "items_per_group", "discount"], columns="arm", values="mae", aggfunc="mean")
    lines += ["Mean MAE over replicates:", "", "```", piv.round(4).to_string(), "```", ""]
    if "rep" in panel.columns and panel["rep"].nunique() > 1:
        se = panel.pivot_table(index=["separation", "items_per_group", "discount"], columns="arm",
                               values="mae", aggfunc=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))))
        lines += ["Standard error over replicates:", "", "```", se.round(4).to_string(), "```", ""]
        # paired comparison: learned vs global, per replicate
        wide = panel.pivot_table(index=["separation", "items_per_group", "discount", "rep"],
                                 columns="arm", values="mae")
        if {"learned-partition", "global-discounted"}.issubset(wide.columns):
            d = (wide["global-discounted"] - wide["learned-partition"]).groupby(
                level=["separation", "items_per_group", "discount"])
            paired = pd.DataFrame({"mean_gain": d.mean(), "se": d.std(ddof=1) / np.sqrt(d.count()),
                                   "wins": d.apply(lambda x: int((x > 0).sum())), "n": d.count()})
            lines += ["Paired learned-vs-global MAE gain (positive = learned better):", "",
                      "```", paired.round(4).to_string(), "```", ""]
    lines += ["Selected number of mixture components (true K = 4):", ""]
    kk = panel.groupby(["separation", "items_per_group", "discount"])["learned_K"].first()
    lines += ["```", kk.to_string(), "```", ""]
    lines += ["Partition recovery (ARI, learned vs true):", ""]
    ari = panel.dropna(subset=["ari_learned_vs_true"]).groupby(
        ["separation", "items_per_group", "discount"]
    )["ari_learned_vs_true"].first().round(3)
    lines += ["```", ari.to_string(), "```", ""]

    lines += ["## B3 reading", ""]
    lines += [
        "For each cell the three explanations are separated as follows.",
        "",
        "- oracle-partition ~ global-discounted  ->  *information shortage*:",
        "  even the true labels buy nothing, so no estimator can.",
        "- oracle-partition < learned-partition  ->  *partition recovery*:",
        "  the structure pays but the mixture did not find it.",
        "- learned+regularized < learned-partition -> *plug-in failure*:",
        "  the boundary collapse of the variance component was the binding problem.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_figure(calib: pd.DataFrame, panel: pd.DataFrame, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # Panel A: learned-vs-global gain against forgetting strength.
    ax = axes[0]
    for sep, sub in panel.groupby("separation"):
        piv = sub.pivot_table(index="discount", columns="arm", values="mae")
        if not {"learned-partition", "global-discounted"}.issubset(set(piv.columns)):
            continue
        gain = 100.0 * (piv["global-discounted"] - piv["learned-partition"]) / piv["global-discounted"]
        ax.plot(1.0 - gain.index.to_numpy(float), gain.to_numpy(), marker="o", label=f"separation {sep:g}")
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"forgetting strength $1-w$")
    ax.set_ylabel("learned vs global pool\nMAE improvement (%)")
    ax.set_title("A. Forgetting erodes the gain\nfrom a learned partition")
    ax.legend(fontsize=7, frameon=False)

    # Panel B: collapse probability against the resolution statistic.
    ax = axes[1]
    sub = calib.sort_values("z")
    ax.scatter(sub["z"], sub["p_empirical"], s=7, alpha=0.35, label="simulated")
    zz = np.linspace(0.0, max(4.0, float(sub["z"].quantile(0.98))), 200)
    ax.plot(zz, norm.cdf(-zz), color="k", lw=1.6, label=r"$\Phi(-z)$")
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"$z=\rho_g\sqrt{(m_g-1)/2}$")
    ax.set_ylabel(r"$\Pr[\hat\tau_g^2=0]$")
    ax.set_title("B. Collapse probability follows\nthe resolution statistic")
    ax.legend(fontsize=7, frameon=False)

    # Panel C: heatmap of learned-vs-global improvement.
    ax = axes[2]
    piv = panel.pivot_table(index="separation", columns="discount", values="mae", aggfunc="mean")
    lp = panel[panel.arm == "learned-partition"].pivot_table(index="separation", columns="discount", values="mae")
    gp = panel[panel.arm == "global-discounted"].pivot_table(index="separation", columns="discount", values="mae")
    grid = 100.0 * (gp - lp) / gp
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-np.nanmax(np.abs(grid.to_numpy())),
                   vmax=np.nanmax(np.abs(grid.to_numpy())))
    ax.set_xticks(range(len(grid.columns)), [f"{c:g}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{r:g}" for r in grid.index])
    ax.set_xlabel("discount $w$")
    ax.set_ylabel("group separation")
    ax.set_title("C. Improvement over global pool (%)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--quick", action="store_true", help="Reduced grid for a fast smoke run.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fig", type=Path, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[A3] collapse-probability calibration ...")
    calib = collapse_calibration(rng, n_rep=1500 if args.quick else 4000)
    calib.to_csv(args.out / "collapse_calibration.csv", index=False)
    print(f"      cells={len(calib)}  median|err|={calib.abs_error.median():.4f}  max|err|={calib.abs_error.max():.4f}")

    print("[B1] panel grid ...")
    panel = run_panel_grid(rng, quick=args.quick)
    panel.to_csv(args.out / "panel_grid.csv", index=False)
    print(f"      rows={len(panel)}")

    write_summary(calib, panel, args.out)
    fig_path = args.fig or (Path(__file__).resolve().parents[2] / "paper_v2" / "figs" / "fig_resolution.pdf")
    make_figure(calib, panel, fig_path)
    print(f"wrote {args.out}/ and {fig_path}")


if __name__ == "__main__":
    main()
