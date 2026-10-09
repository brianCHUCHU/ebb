from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    set_seed,
    default_data_file,
    default_out_dir,
    default_m5_sales_file,
    default_m5_calendar_file,
)
from data_loading import (
    load_online_retail,
    preprocess_online_retail,
    train_eval_split_fixed_origin,
    load_m5_long,
    preprocess_m5,
    load_generic_long,
    train_eval_split_last_h,
)
from experiments.protocols import evaluate_point_models, iter_walk_forward_frames
from models.eb_hurdle import (
    EBHurdleOnlineState,
    EBHurdleParams,
    fit_eb_hurdle,
    initialize_online_eb_hurdle,
    predict_online_eb_hurdle,
    predict_eb_hurdle,
    update_online_eb_hurdle,
)
from models.baselines import fit_predict_baselines, fit_predict_single_baseline, POINT_BASELINE_KEYS
from models.hurdle_baselines import (
    fit_hurdle_global_lognormal,
    fit_hurdle_local_lognormal,
    predict_hurdle_global_lognormal,
    predict_hurdle_local_lognormal,
)
from models.tweedie_baseline import TWEEDIE_POINT_COL, fit_predict_tweedie_panel
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import (
    select_fit_discount,
    select_pooling_and_discount,
    split_for_hyper_estimation,
)
from metrics import rmsse, wrmsse, compute_adi_cv2, classify_adi_cv2
from plotting import plot_shrinkage_scatter


POINT_EBHURDLE_MODEL = "EB-Hurdle"


def _normalize_point_baseline_mode(baseline_mode: str | None) -> str:
    mode = str(baseline_mode or "paper").lower()
    if mode == "full":
        mode = "extended"
    valid = {"paper", "extended", "hurdle_only", "hb_only"}
    if mode not in valid:
        raise ValueError("baseline_mode must be one of: paper, extended, full, hurdle_only, hb_only.")
    return mode


def _include_statistical_point_baselines(baseline_mode: str) -> bool:
    return baseline_mode in {"paper", "extended"}


def _include_hurdle_point_baselines(baseline_mode: str) -> bool:
    return baseline_mode in {"extended", "hurdle_only"}


def _build_regime_group_labels(train_df: pd.DataFrame) -> pd.Series | None:
    feats = compute_adi_cv2(train_df)
    if feats.empty:
        return None
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)
    return feats.set_index("unique_id")["category"].astype(str)


def _build_m5_hierarchy_group_labels(df: pd.DataFrame) -> pd.Series | None:
    cols = set(df.columns)
    if {"dept_id", "store_id", "unique_id"}.issubset(cols):
        tmp = df[["unique_id", "dept_id", "store_id"]].drop_duplicates(subset=["unique_id"]).copy()
        tmp["group"] = tmp["dept_id"].astype(str) + "|" + tmp["store_id"].astype(str)
        return tmp.set_index("unique_id")["group"]
    if {"cat_id", "store_id", "unique_id"}.issubset(cols):
        tmp = df[["unique_id", "cat_id", "store_id"]].drop_duplicates(subset=["unique_id"]).copy()
        tmp["group"] = tmp["cat_id"].astype(str) + "|" + tmp["store_id"].astype(str)
        return tmp.set_index("unique_id")["group"]
    if {"store_id", "unique_id"}.issubset(cols):
        tmp = df[["unique_id", "store_id"]].drop_duplicates(subset=["unique_id"]).copy()
        tmp["group"] = tmp["store_id"].astype(str)
        return tmp.set_index("unique_id")["group"]
    return None


def _build_m5_taxonomy4_group_labels(df: pd.DataFrame) -> pd.Series | None:
    """Build 4-group taxonomy labels on M5 via ADI/CV^2 (Smooth/Erratic/Intermittent/Lumpy)."""
    return _build_regime_group_labels(df)


def _resolve_fit_discount(
    spec: str,
    init_set: pd.DataFrame,
    group_labels: pd.Series | None,
    item_variance_mode: str,
    variance_prior_df: float,
    hyper_split: str = "off",
    hyper_shrink: str = "off",
) -> tuple[float, pd.DataFrame | None]:
    """Resolve --hb-fit-discount: a float, or 'auto' for internal-split selection."""
    text = str(spec).strip().lower()
    if text in {"auto", "select"}:
        w, diag = select_fit_discount(
            init_set,
            group_labels=group_labels,
            item_variance_mode=item_variance_mode,
            item_variance_shrink_strength=variance_prior_df,
            hyper_split=hyper_split,
            hyper_shrink=hyper_shrink,
        )
        return w, diag
    return float(text), None


def _fit_ebhurdle_variant_frames(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    taxonomy_labels: pd.Series | None,
    hb_group_shrink_strength: float,
    hb_item_variance_mode: str,
    hb_variance_prior_df: float,
    mixture_k: int,
    discount_spec: str,
) -> tuple[list[pd.DataFrame], dict[str, float], list[dict[str, object]]]:
    """Fit the EB-Hurdle variant suite for the pooling/discount comparison:

    - EB-Hurdle-Global   : single global pool, no discount
    - EB-Hurdle-Taxonomy : fixed ADI/CV^2 pooling, no discount (paper model, when labels exist)
    - EB-Hurdle-Mixture  : learned mixture pooling (BIC-selected K), no discount
    - EB-Hurdle-Discount : base pooling + recency discount (auto-selected on init split)
    - EB-Hurdle-Mix-Disc : learned pooling on discounted stats + recency discount
    """
    frames: list[pd.DataFrame] = []
    timings: dict[str, float] = {}
    diag_rows: list[dict[str, object]] = []

    def _fit_one(name: str, labels: pd.Series | None, discount: float) -> None:
        t0 = time.perf_counter()
        params = fit_eb_hurdle(
            init_set,
            group_labels=labels,
            group_shrink_strength=hb_group_shrink_strength,
            item_variance_mode=hb_item_variance_mode,
            item_variance_shrink_strength=hb_variance_prior_df,
            fit_discount=discount,
        )
        preds = predict_eb_hurdle(params, eval_set, quantiles=None)
        timings[name] = time.perf_counter() - t0
        frames.append(preds.rename(columns={"yhat": name}))

    # 1) Global pool
    _fit_one("EB-Hurdle-Global", None, 1.0)
    diag_rows.append({"variant": "EB-Hurdle-Global", "grouping": "global", "fit_discount": 1.0})

    # 2) Fixed taxonomy (paper model)
    if taxonomy_labels is not None:
        _fit_one("EB-Hurdle-Taxonomy", taxonomy_labels, 1.0)
        diag_rows.append({"variant": "EB-Hurdle-Taxonomy", "grouping": "taxonomy", "fit_discount": 1.0})

    # 3) Learned mixture pooling
    t0 = time.perf_counter()
    mix_res = mixture_group_labels(init_set, k=mixture_k)
    mix_fit_seconds = time.perf_counter() - t0
    _fit_one("EB-Hurdle-Mixture", mix_res.labels, 1.0)
    timings["EB-Hurdle-Mixture"] += mix_fit_seconds
    diag_rows.append(
        {
            "variant": "EB-Hurdle-Mixture",
            "grouping": "mixture",
            "fit_discount": 1.0,
            "mixture_k": mix_res.k,
            "mixture_bic": mix_res.bic,
            "mixture_loglik": mix_res.log_likelihood,
        }
    )

    # 4) Recency discount on the base pooling
    base_labels = taxonomy_labels
    t0 = time.perf_counter()
    w_sel, w_diag = _resolve_fit_discount(
        discount_spec, init_set, base_labels, hb_item_variance_mode, hb_variance_prior_df
    )
    discount_select_seconds = time.perf_counter() - t0
    _fit_one("EB-Hurdle-Discount", base_labels, w_sel)
    timings["EB-Hurdle-Discount"] += discount_select_seconds
    diag_rows.append(
        {
            "variant": "EB-Hurdle-Discount",
            "grouping": "taxonomy" if base_labels is not None else "global",
            "fit_discount": w_sel,
            "discount_grid": None if w_diag is None else w_diag.to_dict("records"),
        }
    )

    # 5) Learned pooling on discounted stats + discount
    t0 = time.perf_counter()
    mix_disc_res = mixture_group_labels(init_set, k=mixture_k, fit_discount=w_sel)
    mix_disc_seconds = time.perf_counter() - t0
    _fit_one("EB-Hurdle-Mix-Disc", mix_disc_res.labels, w_sel)
    timings["EB-Hurdle-Mix-Disc"] += mix_disc_seconds
    diag_rows.append(
        {
            "variant": "EB-Hurdle-Mix-Disc",
            "grouping": "mixture",
            "fit_discount": w_sel,
            "mixture_k": mix_disc_res.k,
            "mixture_bic": mix_disc_res.bic,
        }
    )

    return frames, timings, diag_rows


def _run_m5_point(args: argparse.Namespace, out_dir: Path) -> None:
    sales_df, calendar_df = load_m5_long(args.m5_sales, args.m5_calendar)
    df = preprocess_m5(sales_df, calendar_df, sample_size=args.m5_sample_size)

    if df.empty:
        raise ValueError("Preprocessed M5 dataframe is empty; check input files and sample size.")

    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=2.0 / 3.0, min_len=1)
    if eval_set.empty:
        raise ValueError("Evaluation set for M5 is empty; verify dataset contents and split parameters.")

    eval_horizons = eval_set.groupby("unique_id").size()
    eval_horizon = eval_set[["unique_id", "ds"]]
    n_series = int(init_set["unique_id"].nunique())
    timings: dict[str, float] = {}
    baseline_mode = _normalize_point_baseline_mode(args.baseline_mode)

    hb_label_split = str(getattr(args, "hb_label_split", "off")).lower()
    hb_hyper_shrink = str(getattr(args, "hb_hyper_shrink", "off")).lower()
    if hb_label_split == "off":
        structure_set, hb_hyper_train_df = init_set, None
    else:
        structure_set, hb_hyper_train_df = split_for_hyper_estimation(init_set, mode=hb_label_split)
        print(
            f"Hyperparameter sample split: mode={hb_label_split}, "
            f"structure rows={len(structure_set)}, hyper rows={len(hb_hyper_train_df)}"
        )

    hierarchy_labels = _build_m5_hierarchy_group_labels(init_set)
    taxonomy4_labels = _build_m5_taxonomy4_group_labels(structure_set)
    mode = args.m5_hierarchy_mode
    if mode in {"on", "ablation"} and hierarchy_labels is None:
        print("Warning: hierarchy labels unavailable in M5 frame; falling back to non-hierarchical EB-Hurdle.")
        mode = "off"
    if mode in {"taxonomy4"} and taxonomy4_labels is None:
        print("Warning: taxonomy4 labels unavailable in M5 frame; falling back to non-hierarchical EB-Hurdle.")
        mode = "off"

    merged = eval_set[["unique_id", "ds", "y"]].copy()
    ebhurdle_frames: list[pd.DataFrame] = []

    if getattr(args, "hb_grouping", "legacy") == "select":
        # EBB: joint validation-split selection of pooling structure + discount.
        mix_res = mixture_group_labels(structure_set, k=args.hb_mixture_k)
        candidates = {"global": None, "taxonomy": taxonomy4_labels, "mixture": mix_res.labels}
        t0 = time.perf_counter()
        sel_name, sel_w, sel_diag = select_pooling_and_discount(
            init_set, candidates,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=float(max(args.hb_variance_prior_df, 2.1)),
            hyper_split=hb_label_split,
            hyper_shrink=hb_hyper_shrink,
        )
        params_sel = fit_eb_hurdle(
            init_set,
            group_labels=candidates[sel_name],
            group_shrink_strength=args.hb_group_shrink_strength,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=args.hb_variance_prior_df,
            fit_discount=sel_w,
            hyper_train_df=hb_hyper_train_df,
            hyper_shrink=hb_hyper_shrink,
        )
        ebhurdle_sel = predict_eb_hurdle(params_sel, eval_set, quantiles=None)
        timings[POINT_EBHURDLE_MODEL] = time.perf_counter() - t0
        ebhurdle_frames.append(ebhurdle_sel.rename(columns={"yhat": POINT_EBHURDLE_MODEL}))
        sel_diag.to_csv(out_dir / "ebb_selection_diagnostics.csv", index=False)
        print(f"EBB selection (M5): structure={sel_name} (mixture K={mix_res.k}), discount={sel_w}")
        mode = "select"

    if mode in {"off", "ablation"}:
        t0 = time.perf_counter()
        params_plain = fit_eb_hurdle(
            init_set,
            group_labels=None,
            group_shrink_strength=args.hb_group_shrink_strength,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=args.hb_variance_prior_df,
            hyper_train_df=hb_hyper_train_df,
            hyper_shrink=hb_hyper_shrink,
        )
        ebhurdle_plain = predict_eb_hurdle(params_plain, eval_set, quantiles=None)
        model_name = POINT_EBHURDLE_MODEL if mode == "off" else "EB-Hurdle-NoHierarchy"
        timings[model_name] = time.perf_counter() - t0
        ebhurdle_frames.append(ebhurdle_plain.rename(columns={"yhat": model_name}))
    if mode in {"on", "ablation"} and hierarchy_labels is not None:
        t0 = time.perf_counter()
        params_hier = fit_eb_hurdle(
            init_set,
            group_labels=hierarchy_labels,
            group_shrink_strength=args.hb_group_shrink_strength,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=args.hb_variance_prior_df,
            hyper_train_df=hb_hyper_train_df,
            hyper_shrink=hb_hyper_shrink,
        )
        ebhurdle_hier = predict_eb_hurdle(params_hier, eval_set, quantiles=None)
        model_name = POINT_EBHURDLE_MODEL if mode == "on" else "EB-Hurdle-Hierarchy"
        timings[model_name] = time.perf_counter() - t0
        ebhurdle_frames.append(ebhurdle_hier.rename(columns={"yhat": model_name}))
    if mode in {"taxonomy4"} and taxonomy4_labels is not None:
        t0 = time.perf_counter()
        params_tax4 = fit_eb_hurdle(
            init_set,
            group_labels=taxonomy4_labels,
            group_shrink_strength=args.hb_group_shrink_strength,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=args.hb_variance_prior_df,
            hyper_train_df=hb_hyper_train_df,
            hyper_shrink=hb_hyper_shrink,
        )
        ebhurdle_tax4 = predict_eb_hurdle(params_tax4, eval_set, quantiles=None)
        model_name = "EB-Hurdle-Taxonomy4"
        timings[model_name] = time.perf_counter() - t0
        ebhurdle_frames.append(ebhurdle_tax4.rename(columns={"yhat": model_name}))

    for f in ebhurdle_frames:
        merged = merged.merge(f, on=["unique_id", "ds"], how="left")

    baseline_subset = [b.strip() for b in str(getattr(args, "point_baselines", "") or "").split(",") if b.strip()]
    active_baseline_keys = [k for k in POINT_BASELINE_KEYS if not baseline_subset or k in baseline_subset]
    if _include_statistical_point_baselines(baseline_mode):
        for model_key in active_baseline_keys:
            t0 = time.perf_counter()
            pred = fit_predict_single_baseline(init_set, eval_horizons, model_key=model_key, freq="D")
            timings[model_key] = time.perf_counter() - t0
            if not pred.empty:
                pred = pred.drop(columns=["index"], errors="ignore")
                pred = pred.merge(eval_horizon, on=["unique_id", "ds"], how="inner")
                merged = merged.merge(pred, on=["unique_id", "ds"], how="left")

    if _include_hurdle_point_baselines(baseline_mode):
        t0 = time.perf_counter()
        local_params = fit_hurdle_local_lognormal(init_set)
        local_preds = predict_hurdle_local_lognormal(local_params, eval_set)[
            ["unique_id", "ds", "Hurdle-Local-LogNormal"]
        ]
        timings["Hurdle-Local-LogNormal"] = time.perf_counter() - t0
        merged = merged.merge(local_preds, on=["unique_id", "ds"], how="left")

        t0 = time.perf_counter()
        global_params = fit_hurdle_global_lognormal(init_set)
        global_preds = predict_hurdle_global_lognormal(global_params, eval_set)[
            ["unique_id", "ds", "Hurdle-Global-LogNormal"]
        ]
        timings["Hurdle-Global-LogNormal"] = time.perf_counter() - t0
        merged = merged.merge(global_preds, on=["unique_id", "ds"], how="left")

    if getattr(args, "with_iets", False):
        from models.iets_baseline import IETS_POINT_COL, fit_predict_iets_panel

        t0 = time.perf_counter()
        iets_pred = fit_predict_iets_panel(
            init_set,
            eval_set,
            rscript=str(args.iets_rscript),
            script_path=args.iets_script,
            seed=int(args.seed),
            occurrence=str(args.iets_occurrence),
            timeout_seconds=int(args.iets_timeout) if int(args.iets_timeout) > 0 else None,
            per_series_timeout_seconds=int(max(args.iets_per_series_timeout, 1)),
        )
        timings[IETS_POINT_COL] = time.perf_counter() - t0
        merged = merged.merge(iets_pred, on=["unique_id", "ds"], how="left")

    if getattr(args, "hb_variants", "none") == "all":
        var_frames, var_timings, var_diags = _fit_ebhurdle_variant_frames(
            init_set,
            eval_set,
            taxonomy_labels=_build_m5_taxonomy4_group_labels(init_set),
            hb_group_shrink_strength=args.hb_group_shrink_strength,
            hb_item_variance_mode=args.hb_item_variance_mode,
            hb_variance_prior_df=float(max(args.hb_variance_prior_df, 2.1)),
            mixture_k=args.hb_mixture_k,
            discount_spec="auto" if args.hb_fit_discount == "1.0" else args.hb_fit_discount,
        )
        for f in var_frames:
            merged = merged.merge(f, on=["unique_id", "ds"], how="left")
        timings.update(var_timings)
        pd.DataFrame(var_diags).to_csv(out_dir / "ebhurdle_variant_diagnostics.csv", index=False)

    if getattr(args, "save_predictions", False):
        merged.to_csv(out_dir / "point_predictions.csv.gz", index=False, compression="gzip")

    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y", "index"}]
    metrics_df = evaluate_point_models(init_set, merged, model_cols=model_cols)
    if n_series > 0:
        metrics_df["Efficiency"] = metrics_df["model"].map(
            lambda m: timings.get(m, 0.0) / n_series
        )
    metrics_path = out_dir / "point_metrics_m5.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if mode == "ablation":
        ablation = metrics_df[metrics_df["model"].isin(["EB-Hurdle-NoHierarchy", "EB-Hurdle-Hierarchy"])].copy()
        ablation.to_csv(out_dir / "m5_hierarchy_ablation.csv", index=False)

    if not metrics_df.empty:
        print("=== M5 Point Forecast Metrics ===")
        print(metrics_df.sort_values("WRMSSE"))


def _predict_online_models_once(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    hb_group_labels: pd.Series | None = None,
    hb_group_shrink_strength: float = 0.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    ebhurdle_override: pd.DataFrame | None = None,
    baseline_mode: str = "paper",
    with_tweedie: bool = False,
    tweedie_lags: int = 14,
    tweedie_power: float = 1.5,
    tweedie_alpha: float = 0.1,
    tweedie_max_iter: int = 1000,
    hb_fit_discount: float = 1.0,
    freq: str = "D",
    hb_label_split: str = "off",
    hb_hyper_shrink: str = "off",
) -> pd.DataFrame:
    baseline_mode = _normalize_point_baseline_mode(baseline_mode)

    if ebhurdle_override is None:
        hyper_train_df = (
            None if str(hb_label_split).lower() == "off"
            else split_for_hyper_estimation(train_df, mode=hb_label_split)[1]
        )
        params = fit_eb_hurdle(
            train_df,
            group_labels=hb_group_labels,
            group_shrink_strength=hb_group_shrink_strength,
            item_variance_mode=hb_item_variance_mode,
            item_variance_shrink_strength=hb_variance_prior_df,
            fit_discount=hb_fit_discount,
            hyper_train_df=hyper_train_df,
            hyper_shrink=hb_hyper_shrink,
        )
        ebhurdle_point = predict_eb_hurdle(params, eval_df, quantiles=None)
        ebhurdle_point = ebhurdle_point.rename(columns={"yhat": POINT_EBHURDLE_MODEL})
    else:
        ebhurdle_point = ebhurdle_override.copy()
        if POINT_EBHURDLE_MODEL not in ebhurdle_point.columns and "yhat" in ebhurdle_point.columns:
            ebhurdle_point = ebhurdle_point.rename(columns={"yhat": POINT_EBHURDLE_MODEL})

    merged = eval_df[["unique_id", "ds", "y"]].copy()
    merged = merged.merge(ebhurdle_point, on=["unique_id", "ds"], how="left")

    if _include_statistical_point_baselines(baseline_mode):
        eval_horizons = eval_df["unique_id"].value_counts()
        base_preds = fit_predict_baselines(train_df, horizons=eval_horizons, freq=freq, probabilistic=False)
        if not base_preds.empty:
            merged = merged.merge(base_preds, on=["unique_id", "ds"], how="left")

    if _include_hurdle_point_baselines(baseline_mode):
        local_params = fit_hurdle_local_lognormal(train_df)
        local_preds = predict_hurdle_local_lognormal(local_params, eval_df)
        local_preds = local_preds[["unique_id", "ds", "Hurdle-Local-LogNormal"]]

        global_params = fit_hurdle_global_lognormal(train_df)
        global_preds = predict_hurdle_global_lognormal(global_params, eval_df)
        global_preds = global_preds[["unique_id", "ds", "Hurdle-Global-LogNormal"]]

        merged = merged.merge(local_preds, on=["unique_id", "ds"], how="left")
        merged = merged.merge(global_preds, on=["unique_id", "ds"], how="left")
    if with_tweedie:
        tw_pred = fit_predict_tweedie_panel(
            train_df,
            eval_df,
            lags=tweedie_lags,
            power=tweedie_power,
            alpha=tweedie_alpha,
            max_iter=tweedie_max_iter,
        )
        merged = merged.merge(tw_pred, on=["unique_id", "ds"], how="left")
    return merged


def _run_online_point_fixed(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    hb_group_labels: pd.Series | None = None,
    hb_group_shrink_strength: float = 0.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    baseline_mode: str = "paper",
) -> pd.DataFrame:
    return _predict_online_models_once(
        init_set,
        eval_set,
        hb_group_labels=hb_group_labels,
        hb_group_shrink_strength=hb_group_shrink_strength,
        hb_item_variance_mode=hb_item_variance_mode,
        hb_variance_prior_df=hb_variance_prior_df,
        baseline_mode=baseline_mode,
    )


def _run_online_point_fixed_with_timing(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    hb_group_labels: pd.Series | None = None,
    hb_group_shrink_strength: float = 0.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    baseline_mode: str = "paper",
    *,
    seed: int = 42,
    with_iets: bool = False,
    iets_rscript: str = "Rscript",
    iets_script: Path | None = None,
    iets_occurrence: str = "auto",
    iets_timeout: int | None = 3600,
    iets_per_series_timeout: int = 10,
    with_tweedie: bool = False,
    tweedie_lags: int = 14,
    tweedie_power: float = 1.5,
    tweedie_alpha: float = 0.1,
    tweedie_max_iter: int = 1000,
    hb_fit_discount: float = 1.0,
    freq: str = "D",
    hb_hyper_train_df: pd.DataFrame | None = None,
    hb_hyper_shrink: str = "off",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run fixed-origin point forecasting and record per-model elapsed time (fit+predict) in seconds."""
    timings: dict[str, float] = {}
    merged = eval_set[["unique_id", "ds", "y"]].copy()

    # EB-Hurdle
    t0 = time.perf_counter()
    params = fit_eb_hurdle(
        init_set,
        group_labels=hb_group_labels,
        group_shrink_strength=hb_group_shrink_strength,
        item_variance_mode=hb_item_variance_mode,
        item_variance_shrink_strength=hb_variance_prior_df,
        fit_discount=hb_fit_discount,
        hyper_train_df=hb_hyper_train_df,
        hyper_shrink=hb_hyper_shrink,
    )
    ebhurdle_point = predict_eb_hurdle(params, eval_set, quantiles=None)
    ebhurdle_point = ebhurdle_point.rename(columns={"yhat": POINT_EBHURDLE_MODEL})
    timings[POINT_EBHURDLE_MODEL] = time.perf_counter() - t0
    merged = merged.merge(ebhurdle_point, on=["unique_id", "ds"], how="left")

    baseline_mode = _normalize_point_baseline_mode(baseline_mode)

    if _include_statistical_point_baselines(baseline_mode):
        eval_horizons = eval_set["unique_id"].value_counts()
        for model_key in POINT_BASELINE_KEYS:
            t0 = time.perf_counter()
            pred = fit_predict_single_baseline(
                init_set, eval_horizons, model_key=model_key, freq=freq
            )
            timings[model_key] = time.perf_counter() - t0
            if not pred.empty:
                pred = pred.drop(columns=["index"], errors="ignore")
                merged = merged.merge(pred, on=["unique_id", "ds"], how="left")

    if _include_hurdle_point_baselines(baseline_mode):
        t0 = time.perf_counter()
        local_params = fit_hurdle_local_lognormal(init_set)
        local_preds = predict_hurdle_local_lognormal(local_params, eval_set)
        local_preds = local_preds[["unique_id", "ds", "Hurdle-Local-LogNormal"]]
        timings["Hurdle-Local-LogNormal"] = time.perf_counter() - t0
        merged = merged.merge(local_preds, on=["unique_id", "ds"], how="left")

        t0 = time.perf_counter()
        global_params = fit_hurdle_global_lognormal(init_set)
        global_preds = predict_hurdle_global_lognormal(global_params, eval_set)
        global_preds = global_preds[["unique_id", "ds", "Hurdle-Global-LogNormal"]]
        timings["Hurdle-Global-LogNormal"] = time.perf_counter() - t0
        merged = merged.merge(global_preds, on=["unique_id", "ds"], how="left")

    if with_iets:
        from models.iets_baseline import IETS_POINT_COL, fit_predict_iets_panel

        t0 = time.perf_counter()
        iets_pred = fit_predict_iets_panel(
            init_set,
            eval_set,
            rscript=iets_rscript,
            script_path=iets_script,
            seed=seed,
            occurrence=iets_occurrence,
            timeout_seconds=iets_timeout,
            per_series_timeout_seconds=iets_per_series_timeout,
        )
        timings[IETS_POINT_COL] = time.perf_counter() - t0
        merged = merged.merge(iets_pred, on=["unique_id", "ds"], how="left")
    if with_tweedie:
        t0 = time.perf_counter()
        tw_pred = fit_predict_tweedie_panel(
            init_set,
            eval_set,
            lags=tweedie_lags,
            power=tweedie_power,
            alpha=tweedie_alpha,
            max_iter=tweedie_max_iter,
        )
        timings[TWEEDIE_POINT_COL] = time.perf_counter() - t0
        merged = merged.merge(tw_pred, on=["unique_id", "ds"], how="left")

    return merged, timings


def _run_online_point_walk_forward(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    walk_step: int,
    hb_group_labels: pd.Series | None = None,
    hb_group_shrink_strength: float = 0.0,
    hb_online_update: bool = True,
    hb_dynamic_occurrence: bool = False,
    hb_occ_discount: float = 1.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    baseline_mode: str = "paper",
    with_iets: bool = False,
    iets_rscript: str = "Rscript",
    iets_script: Path | None = None,
    iets_occurrence: str = "auto",
    iets_timeout: int | None = 3600,
    iets_per_series_timeout: int = 10,
    seed: int = 42,
    with_tweedie: bool = False,
    tweedie_lags: int = 14,
    tweedie_power: float = 1.5,
    tweedie_alpha: float = 0.1,
    tweedie_max_iter: int = 1000,
    hb_fit_discount: float = 1.0,
    hb_label_split: str = "off",
    hb_hyper_shrink: str = "off",
) -> pd.DataFrame:
    if str(hb_label_split).lower() != "off" and hb_online_update:
        print(
            "Warning: --hb-label-split is not applied by the online sufficient-statistics "
            "update; it only takes effect on per-block refits. Pass --no-hb-online-update "
            "for a split-consistent walk-forward run."
        )
    step_outputs: list[pd.DataFrame] = []
    total_frames = max(int(np.ceil(eval_set.groupby("unique_id").size().max() / walk_step)), 1)
    hb_state: EBHurdleOnlineState | None = None
    fit_predict_iets_panel = None
    iets_col: str | None = None
    if with_iets:
        from models.iets_baseline import IETS_POINT_COL, fit_predict_iets_panel as _fit_predict_iets_panel
        fit_predict_iets_panel = _fit_predict_iets_panel
        iets_col = IETS_POINT_COL
    if hb_online_update:
        hb_state = initialize_online_eb_hurdle(
            init_set,
            group_labels=hb_group_labels,
            group_shrink_strength=hb_group_shrink_strength,
            fit_discount=hb_fit_discount,
            dynamic_occurrence=hb_dynamic_occurrence,
            occurrence_discount=hb_occ_discount,
            item_variance_mode=hb_item_variance_mode,
            item_variance_shrink_strength=hb_variance_prior_df,
        )

    for frame_idx, frame in enumerate(iter_walk_forward_frames(init_set, eval_set, step_size=walk_step), start=1):
        if frame_idx == 1 or frame_idx % 10 == 0 or frame_idx == total_frames:
            print(
                f"[walk_forward] frame {frame_idx}/{total_frames} "
                f"(step={frame.step}, n_target={len(frame.target)})"
            )
        if hb_state is not None:
            ebhurdle_step = predict_online_eb_hurdle(
                hb_state,
                frame.target,
                quantiles=None,
                include_hyper_uncertainty=False,
            ).rename(columns={"yhat": POINT_EBHURDLE_MODEL})
            merged_step = _predict_online_models_once(
                frame.history,
                frame.target,
                hb_group_labels=hb_group_labels,
                hb_group_shrink_strength=hb_group_shrink_strength,
                hb_item_variance_mode=hb_item_variance_mode,
                hb_variance_prior_df=hb_variance_prior_df,
                ebhurdle_override=ebhurdle_step,
                baseline_mode=baseline_mode,
                with_tweedie=with_tweedie,
                tweedie_lags=tweedie_lags,
                tweedie_power=tweedie_power,
                tweedie_alpha=tweedie_alpha,
                tweedie_max_iter=tweedie_max_iter,
            )
            hb_state = update_online_eb_hurdle(hb_state, frame.target)
        else:
            merged_step = _predict_online_models_once(
                frame.history,
                frame.target,
                hb_group_labels=hb_group_labels,
                hb_group_shrink_strength=hb_group_shrink_strength,
                hb_item_variance_mode=hb_item_variance_mode,
                hb_variance_prior_df=hb_variance_prior_df,
                baseline_mode=baseline_mode,
                with_tweedie=with_tweedie,
                tweedie_lags=tweedie_lags,
                tweedie_power=tweedie_power,
                tweedie_alpha=tweedie_alpha,
                tweedie_max_iter=tweedie_max_iter,
                hb_label_split=hb_label_split,
                hb_hyper_shrink=hb_hyper_shrink,
            )
        if with_iets and fit_predict_iets_panel is not None and iets_col is not None:
            iets_step = fit_predict_iets_panel(
                frame.history,
                frame.target,
                rscript=iets_rscript,
                script_path=iets_script,
                seed=seed,
                occurrence=iets_occurrence,
                timeout_seconds=iets_timeout,
                per_series_timeout_seconds=iets_per_series_timeout,
            )
            # R output can cast ids/dates to different dtypes; align keys before merging.
            merged_step["unique_id"] = merged_step["unique_id"].astype(str)
            merged_step["ds"] = pd.to_datetime(merged_step["ds"])
            iets_step["unique_id"] = iets_step["unique_id"].astype(str)
            iets_step["ds"] = pd.to_datetime(iets_step["ds"])
            merged_step = merged_step.merge(
                iets_step[["unique_id", "ds", iets_col]],
                on=["unique_id", "ds"],
                how="left",
            )
        step_outputs.append(merged_step)

    if not step_outputs:
        empty = eval_set[["unique_id", "ds", "y"]].copy()
        return empty

    merged = pd.concat(step_outputs, ignore_index=True)
    merged = merged.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return merged


def _write_shrinkage_plots(init_set: pd.DataFrame, params: EBHurdleParams, out_dir: Path) -> None:
    init_set_local = init_set.copy()
    init_set_local["occ"] = (init_set_local["y"] > 0).astype(int)
    g_init = init_set_local.groupby("unique_id")
    s = g_init["occ"].sum()
    n = g_init["ds"].nunique()
    p_mle = (s / n).reindex(params.p_posterior.index).dropna()
    p_post = params.p_posterior.reindex(p_mle.index)
    plot_shrinkage_scatter(
        x=p_mle.values,
        y=p_post.values,
        xlabel="p MLE (per-item)",
        ylabel="p posterior mean (HB)",
        title="Shrinkage on Demand Probability (p)",
        out_path=out_dir / "fig_shrink_p.png",
    )

    init_set_local["size"] = np.where(init_set_local["occ"] == 1, init_set_local["y"].astype(float), np.nan)
    init_set_local["log_size"] = np.log(init_set_local["size"])
    size_mle = init_set_local.groupby("unique_id")["size"].mean()
    size_post = np.exp(params.shrunk_mean_log + params.sigma_sq_process / 2.0)
    idx = size_mle.index.intersection(size_post.index)
    plot_shrinkage_scatter(
        x=size_mle.reindex(idx).fillna(0).values,
        y=size_post.reindex(idx).fillna(0).values,
        xlabel="Size MLE (per-item avg size)",
        ylabel="Size posterior mean (HB)",
        title="Shrinkage on Demand Size",
        out_path=out_dir / "fig_shrink_size.png",
    )


def _write_segmentation_metrics(
    init_set: pd.DataFrame,
    merged_eval: pd.DataFrame,
    model_cols: list[str],
    out_dir: Path,
) -> None:
    feats = compute_adi_cv2(init_set)
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)

    eval_with_cats = merged_eval.merge(
        feats[["unique_id", "adi", "cv_sq", "category"]],
        on="unique_id",
        how="left",
    )
    categories = [
        c for c in ["Smooth", "Erratic", "Intermittent", "Lumpy"]
        if (eval_with_cats["category"] == c).any()
    ]

    seg_rows = []
    for model in model_cols:
        for cat in categories:
            subset = eval_with_cats[eval_with_cats["category"] == cat][["unique_id", "ds", "y", model]].dropna(subset=[model]).copy()
            if subset.empty:
                val_r = np.nan
                val_wr = np.nan
            else:
                tmp = subset.rename(columns={model: "y_pred"})
                val_r = rmsse(init_set, tmp)
                val_wr = wrmsse(init_set, tmp)
            seg_rows.append({"model": model, "category": cat, "Metric": "RMSSE", "Value": val_r})
            seg_rows.append({"model": model, "category": cat, "Metric": "WRMSSE", "Value": val_wr})

    seg_df = pd.DataFrame(seg_rows)
    if seg_df.empty:
        return

    pivot = seg_df.pivot_table(index="model", columns=["Metric", "category"], values="Value")
    pivot = pivot.reindex(model_cols)
    pivot.columns = [f"{cat}_{metric}" for metric, cat in pivot.columns]
    pivot = pivot.reindex(
        columns=[f"{cat}_RMSSE" for cat in categories]
        + [f"{cat}_WRMSSE" for cat in categories if f"{cat}_WRMSSE" in pivot.columns]
    )
    pivot.reset_index().to_csv(out_dir / "segmentation_rmsse.csv", index=False)


def _cold_start_bucket(init_pos: int) -> str:
    v = int(init_pos)
    if v <= 1:
        return "CS_0_1"
    if v <= 3:
        return "CS_2_3"
    if v <= 5:
        return "CS_4_5"
    if v <= 10:
        return "CS_6_10"
    return "CS_11_plus"


def _build_slice_features(init_set: pd.DataFrame) -> pd.DataFrame:
    tmp = init_set.copy()
    tmp["occ"] = (tmp["y"] > 0).astype(int)
    base = (
        tmp.groupby("unique_id", as_index=False)
        .agg(init_len=("ds", "nunique"), init_pos=("occ", "sum"))
    )
    base["init_pos"] = base["init_pos"].astype(int)
    base["cold_start_bin"] = base["init_pos"].map(_cold_start_bucket)

    feats = compute_adi_cv2(init_set)
    if not feats.empty:
        feats["category"] = feats.apply(classify_adi_cv2, axis=1)
        base = base.merge(feats[["unique_id", "adi", "cv_sq", "category"]], on="unique_id", how="left")
    else:
        base["adi"] = np.nan
        base["cv_sq"] = np.nan
        base["category"] = np.nan

    base["category"] = base["category"].fillna("NoPositiveInit")
    base["sparse_regime"] = np.where(
        base["category"].isin(["Intermittent", "Lumpy"]),
        "Sparse(Intermittent|Lumpy)",
        "NonSparse(Smooth|Erratic)",
    )
    return base


def _ordered_slice_values(slice_type: str, values: pd.Series) -> list[str]:
    observed = [str(v) for v in values.dropna().unique().tolist()]
    if slice_type == "regime":
        order = ["Smooth", "Erratic", "Intermittent", "Lumpy", "NoPositiveInit"]
        return [x for x in order if x in observed] + sorted([x for x in observed if x not in order])
    if slice_type == "cold_start":
        order = ["CS_0_1", "CS_2_3", "CS_4_5", "CS_6_10", "CS_11_plus"]
        return [x for x in order if x in observed] + sorted([x for x in observed if x not in order])
    if slice_type == "sparse":
        order = ["NonSparse(Smooth|Erratic)", "Sparse(Intermittent|Lumpy)"]
        return [x for x in order if x in observed] + sorted([x for x in observed if x not in order])
    return sorted(observed)


def _write_point_slice_metrics(
    init_set: pd.DataFrame,
    merged_eval: pd.DataFrame,
    model_cols: list[str],
    out_dir: Path,
) -> None:
    slice_feats = _build_slice_features(init_set)
    eval_with_slices = merged_eval.merge(
        slice_feats[["unique_id", "category", "sparse_regime", "cold_start_bin", "init_len", "init_pos"]],
        on="unique_id",
        how="left",
    )
    spec = [
        ("regime", "category"),
        ("sparse", "sparse_regime"),
        ("cold_start", "cold_start_bin"),
    ]

    rows: list[dict[str, float | int | str]] = []
    for slice_type, col in spec:
        for slice_value in _ordered_slice_values(slice_type, eval_with_slices[col]):
            block = eval_with_slices[eval_with_slices[col] == slice_value]
            if block.empty:
                continue
            ids = block["unique_id"].dropna().unique()
            init_sub = init_set[init_set["unique_id"].isin(ids)]
            for model in model_cols:
                subset = block[["unique_id", "ds", "y", model]].dropna(subset=[model]).copy()
                if subset.empty:
                    continue
                metric_df = evaluate_point_models(init_sub, subset, model_cols=[model])
                if metric_df.empty:
                    continue
                metric_row = metric_df.iloc[0].to_dict()
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice": slice_value,
                        "model": model,
                        "n_obs": int(len(subset)),
                        "n_series": int(subset["unique_id"].nunique()),
                        **metric_row,
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "point_slice_metrics.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["online_retail", "m5", "auto", "carparts", "raf"], default="online_retail")
    ap.add_argument("--holdout-h", type=int, default=None, help="Fixed evaluation horizon (last-h holdout) for auto/carparts/raf; defaults: auto=6, carparts=6, raf=12.")
    ap.add_argument("--protocol", choices=["fixed", "walk_forward"], default="fixed")
    ap.add_argument("--walk-step", type=int, default=7, help="Block size (steps) for walk-forward protocol. Larger values are faster and reduce repeated re-fitting cost.")
    ap.add_argument("--baseline-mode", choices=["paper", "extended", "full", "hurdle_only", "hb_only"], default=None, help="Baseline set for both fixed and walk-forward: paper=EB-Hurdle + classical baselines from the paper, extended=paper + hurdle baselines, full=legacy alias for extended, hurdle_only=EB-Hurdle + hurdle baselines, hb_only=EB-Hurdle only.")
    ap.add_argument("--hb-regime-aware", dest="hb_regime_aware", action="store_true", default=True, help="Use ADI/CV^2 regime-aware HB priors for Online Retail.")
    ap.add_argument("--no-hb-regime-aware", dest="hb_regime_aware", action="store_false", help="Disable regime-aware priors and use global HB priors.")
    ap.add_argument("--hb-group-shrink-strength", type=float, default=0.0, help="Extra shrink strength from group-level hyperparameters back to global hyperparameters (0 disables).")
    ap.add_argument("--hb-online-update", dest="hb_online_update", action="store_true", default=True, help="Use sufficient-statistics online update for EB-Hurdle in walk-forward.")
    ap.add_argument("--no-hb-online-update", dest="hb_online_update", action="store_false", help="Disable online update and re-fit EB-Hurdle each walk-forward step.")
    ap.add_argument("--hb-dynamic-occurrence", dest="hb_dynamic_occurrence", action="store_true", default=False, help="Enable dynamic discounted occurrence update for EB-Hurdle in online walk-forward.")
    ap.add_argument("--no-hb-dynamic-occurrence", dest="hb_dynamic_occurrence", action="store_false", help="Disable dynamic discounted occurrence update.")
    ap.add_argument("--hb-occ-discount", type=float, default=1.0, help="Discount factor for dynamic occurrence update (0<d<=1). Smaller means faster adaptation.")
    ap.add_argument("--hb-item-variance-mode", choices=["group", "conjugate"], default="conjugate", help="Process variance mode for size: group or conjugate.")
    ap.add_argument("--hb-variance-prior-df", type=float, default=20.0, help="Prior degrees of freedom for conjugate variance model (larger = stronger shrinkage).")
    ap.add_argument("--m5-hierarchy-mode", choices=["off", "on", "ablation", "taxonomy4"], default="off", help="M5 grouping for EB-Hurdle: off=single global pool (paper default), on=hier priors, ablation=off vs hierarchy, taxonomy4=ADI/CV^2 4-group pooling.")
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--m5-sales", type=Path, default=default_m5_sales_file())
    ap.add_argument("--m5-calendar", type=Path, default=default_m5_calendar_file())
    ap.add_argument("--m5-sample-size", type=int, default=5000)
    ap.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Optional cap on number of Online Retail series (after preprocess) for faster runs; omit for all series.",
    )
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--with-iets",
        action="store_true",
        help="Include iETS point forecasts via R smooth::adam (requires R on PATH and CRAN packages smooth, greybox). Supported for fixed and walk_forward on online retail, and fixed runs on M5.",
    )
    ap.add_argument("--with-tweedie", action="store_true", help="Include Tweedie autoregressive baseline.")
    ap.add_argument("--tweedie-lags", type=int, default=14, help="Number of lag features for Tweedie baseline.")
    ap.add_argument("--tweedie-power", type=float, default=1.5, help="Tweedie variance power (1<p<2 typical for intermittent demand).")
    ap.add_argument("--tweedie-alpha", type=float, default=0.1, help="L2 regularization strength for Tweedie baseline.")
    ap.add_argument("--tweedie-max-iter", type=int, default=1000, help="Max optimizer iterations for Tweedie baseline.")
    ap.add_argument("--iets-rscript", type=str, default="Rscript", help="Rscript executable for iETS.")
    ap.add_argument(
        "--iets-script",
        type=Path,
        default=None,
        help="Path to iets_panel_forecast.R (default: <repo>/scripts/iets_panel_forecast.R).",
    )
    ap.add_argument(
        "--iets-occurrence",
        type=str,
        default="auto",
        help="adam() occurrence= argument (e.g. auto, direct, fixed, odds-ratio).",
    )
    ap.add_argument("--iets-timeout", type=int, default=3600, help="Overall timeout in seconds for each iETS R subprocess.")
    ap.add_argument("--iets-per-series-timeout", type=int, default=10, help="Per-series elapsed-time limit passed to the iETS R script.")
    ap.add_argument("--hb-grouping", choices=["legacy", "global", "taxonomy", "mixture", "select"], default="legacy", help="Pooling structure for the main EB-Hurdle model: legacy=follow --hb-regime-aware, global=single pool, taxonomy=fixed ADI/CV^2 groups, mixture=learned mixture-of-priors pooling (EM, BIC-selected K), select=jointly select structure and discount on the internal validation split (EBB).")
    ap.add_argument("--hb-mixture-k", type=int, default=0, help="Number of mixture components for learned pooling (0 = select by BIC).")
    ap.add_argument("--hb-hyper-shrink", choices=["off", "credibility"], default="off", help="Regularize the group-level variance components by Buhlmann credibility toward the global value (data-determined weights, no tuned parameter). Prevents the collapse of tau^2 that drives the size credibility weight to zero under learned partitions and strong discounting. off reproduces the released model.")
    ap.add_argument("--hb-label-split", choices=["off", "parity", "chrono"], default="off", help="Sample-split the initialization window so the pooling structure is learned on one half and the group hyperparameters estimated on the other (per-item statistics still use the full window). Removes the post-selection collapse of the between-item variance under learned partitions. parity=interleaved (preserves recency in both halves), chrono=first/second half.")
    ap.add_argument("--hb-fit-discount", type=str, default="1.0", help="Recency discount for initialization-window sufficient statistics: a float in (0,1], or 'auto' to select on an internal chronological split.")
    ap.add_argument("--hb-variants", choices=["none", "all"], default="none", help="Additionally fit the EB-Hurdle pooling/discount variant suite (Global/Taxonomy/Mixture/Discount/Mix-Disc) as extra models in the fixed protocol.")
    ap.add_argument("--save-predictions", action="store_true", help="Persist the merged per-timestamp prediction frame (point_predictions.csv.gz) for paired significance testing.")
    ap.add_argument("--point-baselines", type=str, default="", help="Comma-separated subset of statistical point baselines to run (empty = all). E.g. 'CrostonClassic,CrostonSBA,TSB,ADIDA,IMAPA' to skip slow AutoARIMA/AutoTheta on full M5.")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "m5":
        if args.protocol != "fixed":
            raise ValueError("M5 currently supports only --protocol fixed in run_point.")
        _run_m5_point(args, out_dir)
        return

    # Load & preprocess
    freq = "D"
    monthly_h_defaults = {"auto": 6, "carparts": 6, "raf": 12}
    if args.dataset in monthly_h_defaults:
        from utils import find_repo_root

        freq = "MS"
        df = load_generic_long(find_repo_root() / "data" / f"{args.dataset}_long.csv")
    else:
        df_raw = load_online_retail(args.data)
        df = preprocess_online_retail(df_raw)
    if args.max_series is not None:
        max_series = max(int(args.max_series), 1)
        uids = df["unique_id"].drop_duplicates()
        if len(uids) > max_series:
            keep = np.random.choice(uids.to_numpy(), size=max_series, replace=False)
            df = df[df["unique_id"].isin(keep)].copy()

    # Split: fixed origin by ratio (Online Retail) or last-h holdout (monthly panels)
    if args.dataset in monthly_h_defaults:
        if args.protocol != "fixed":
            raise ValueError(f"{args.dataset} supports only --protocol fixed (last-h holdout).")
        holdout_h = int(args.holdout_h) if args.holdout_h else monthly_h_defaults[args.dataset]
        init_set, eval_set = train_eval_split_last_h(df, h=holdout_h)
    else:
        init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    baseline_mode = _normalize_point_baseline_mode(args.baseline_mode)
    hb_variance_prior_df = float(max(args.hb_variance_prior_df, 2.1))

    # Sample splitting: pooling structure is learned on one half of the
    # initialization window, group hyperparameters are estimated on the other.
    # Per-item sufficient statistics always come from the full window.
    hb_label_split = str(getattr(args, "hb_label_split", "off")).lower()
    hb_hyper_shrink = str(getattr(args, "hb_hyper_shrink", "off")).lower()
    if hb_label_split == "off":
        structure_set, hb_hyper_train_df = init_set, None
    else:
        structure_set, hb_hyper_train_df = split_for_hyper_estimation(init_set, mode=hb_label_split)
        print(
            f"Hyperparameter sample split: mode={hb_label_split}, "
            f"structure rows={len(structure_set)}, hyper rows={len(hb_hyper_train_df)}"
        )
    if hb_hyper_shrink != "off":
        print(f"Variance-component regularization: {hb_hyper_shrink}")

    taxonomy_labels = _build_regime_group_labels(structure_set)

    if args.hb_grouping == "select":
        # Joint validation-split selection of pooling structure and discount.
        mix_res = mixture_group_labels(structure_set, k=args.hb_mixture_k)
        candidates = {"global": None, "taxonomy": taxonomy_labels, "mixture": mix_res.labels}
        sel_name, hb_fit_discount, sel_diag = select_pooling_and_discount(
            init_set, candidates,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=hb_variance_prior_df,
            hyper_split=hb_label_split,
            hyper_shrink=hb_hyper_shrink,
        )
        hb_group_labels = candidates[sel_name]
        sel_diag.to_csv(out_dir / "ebb_selection_diagnostics.csv", index=False)
        print(f"EBB selection: structure={sel_name} (mixture K={mix_res.k}), discount={hb_fit_discount}")
    else:
        # Resolve recency discount for the main EB-Hurdle model.
        discount_base_labels = taxonomy_labels if (args.hb_grouping in {"legacy", "taxonomy"} and args.hb_regime_aware) else None
        hb_fit_discount, discount_diag = _resolve_fit_discount(
            args.hb_fit_discount, init_set, discount_base_labels, args.hb_item_variance_mode,
            hb_variance_prior_df, hyper_split=hb_label_split, hyper_shrink=hb_hyper_shrink,
        )

        # Resolve pooling structure for the main EB-Hurdle model.
        if args.hb_grouping == "legacy":
            hb_group_labels = taxonomy_labels if args.hb_regime_aware else None
        elif args.hb_grouping == "global":
            hb_group_labels = None
        elif args.hb_grouping == "taxonomy":
            hb_group_labels = taxonomy_labels
        else:  # mixture
            mix_res = mixture_group_labels(structure_set, k=args.hb_mixture_k, fit_discount=hb_fit_discount)
            hb_group_labels = mix_res.labels
            print(f"Mixture pooling: K={mix_res.k}, BIC={mix_res.bic:.1f}, iters={mix_res.n_iter}")
        if discount_diag is not None:
            print(f"Selected fit discount: {hb_fit_discount}")

    n_skus = int(init_set["unique_id"].nunique())
    if args.protocol == "fixed":
        merged, model_timings = _run_online_point_fixed_with_timing(
            init_set,
            eval_set,
            hb_group_labels=hb_group_labels,
            hb_group_shrink_strength=args.hb_group_shrink_strength,
            hb_item_variance_mode=args.hb_item_variance_mode,
            hb_variance_prior_df=hb_variance_prior_df,
            baseline_mode=baseline_mode,
            seed=int(args.seed),
            with_iets=bool(args.with_iets),
            iets_rscript=str(args.iets_rscript),
            iets_script=args.iets_script,
            iets_occurrence=str(args.iets_occurrence),
            iets_timeout=int(args.iets_timeout) if int(args.iets_timeout) > 0 else None,
            iets_per_series_timeout=int(max(args.iets_per_series_timeout, 1)),
            with_tweedie=bool(args.with_tweedie),
            tweedie_lags=int(max(args.tweedie_lags, 1)),
            tweedie_power=float(args.tweedie_power),
            tweedie_alpha=float(max(args.tweedie_alpha, 0.0)),
            tweedie_max_iter=int(max(args.tweedie_max_iter, 100)),
            hb_fit_discount=hb_fit_discount,
            freq=freq,
            hb_hyper_train_df=hb_hyper_train_df,
            hb_hyper_shrink=hb_hyper_shrink,
        )
        if args.hb_variants == "all":
            var_frames, var_timings, var_diags = _fit_ebhurdle_variant_frames(
                init_set,
                eval_set,
                taxonomy_labels=taxonomy_labels,
                hb_group_shrink_strength=args.hb_group_shrink_strength,
                hb_item_variance_mode=args.hb_item_variance_mode,
                hb_variance_prior_df=hb_variance_prior_df,
                mixture_k=args.hb_mixture_k,
                discount_spec="auto" if args.hb_fit_discount == "1.0" else args.hb_fit_discount,
            )
            for f in var_frames:
                merged = merged.merge(f, on=["unique_id", "ds"], how="left")
            model_timings.update(var_timings)
            pd.DataFrame(var_diags).to_csv(out_dir / "ebhurdle_variant_diagnostics.csv", index=False)
    else:
        model_timings = {}
        merged = _run_online_point_walk_forward(
            init_set,
            eval_set,
            walk_step=args.walk_step,
            hb_group_labels=hb_group_labels,
            hb_group_shrink_strength=args.hb_group_shrink_strength,
            hb_online_update=args.hb_online_update,
            hb_dynamic_occurrence=args.hb_dynamic_occurrence,
            hb_occ_discount=args.hb_occ_discount,
            hb_item_variance_mode=args.hb_item_variance_mode,
            hb_variance_prior_df=hb_variance_prior_df,
            baseline_mode=baseline_mode,
            with_iets=bool(args.with_iets),
            iets_rscript=str(args.iets_rscript),
            iets_script=args.iets_script,
            iets_occurrence=str(args.iets_occurrence),
            iets_timeout=int(args.iets_timeout) if int(args.iets_timeout) > 0 else None,
            iets_per_series_timeout=int(max(args.iets_per_series_timeout, 1)),
            seed=int(args.seed),
            with_tweedie=bool(args.with_tweedie),
            tweedie_lags=int(max(args.tweedie_lags, 1)),
            tweedie_power=float(args.tweedie_power),
            tweedie_alpha=float(max(args.tweedie_alpha, 0.0)),
            tweedie_max_iter=int(max(args.tweedie_max_iter, 100)),
            hb_fit_discount=hb_fit_discount,
            hb_label_split=hb_label_split,
            hb_hyper_shrink=hb_hyper_shrink,
        )

    if args.save_predictions:
        merged.to_csv(out_dir / "point_predictions.csv.gz", index=False, compression="gzip")

    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y"}]
    point_df = evaluate_point_models(init_set, merged, model_cols=model_cols)
    point_df.insert(0, "protocol", args.protocol)
    if model_timings and n_skus > 0:
        point_df["Efficiency"] = point_df["model"].map(
            lambda m: model_timings.get(m, 0.0) / n_skus
        )
    point_df.to_csv(out_dir / "point_metrics.csv", index=False)

    _write_shrinkage_plots(
        init_set,
        fit_eb_hurdle(
            init_set,
            group_labels=hb_group_labels,
            group_shrink_strength=args.hb_group_shrink_strength,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=hb_variance_prior_df,
        ),
        out_dir,
    )
    _write_segmentation_metrics(init_set, merged, model_cols, out_dir)
    _write_point_slice_metrics(init_set, merged, model_cols, out_dir)


if __name__ == "__main__":
    main()
