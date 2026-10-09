"""Smoothing constants for TSB selected on the same split EBB uses (item D1).

The released TSB baseline uses the textbook constants
``(alpha_d, alpha_p) = (0.5, 0.45)`` while EBB selects its discount on an
internal chronological split. Comparing a tuned method against an untuned one is
the fairness objection a reviewer will raise first, and by Proposition 1 the two
knobs are the same knob: ``alpha_p = 1 - w``, so EBB's selected
``w in [0.90, 0.997]`` corresponds to ``alpha_p in [0.003, 0.10]`` -- two orders
of magnitude away from ``0.45``.

This module gives TSB the same treatment: a grid search over ``(alpha_d,
alpha_p)`` scored on the *same* internal chronological split, under the *same*
criterion, with a candidate budget of the same order as EBB's 24
configurations. Nothing here touches out-of-sample data.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from models.baselines import _fit_predict_panel, _import_statsforecast, _prepare_horizons
from models.eb_hurdle import _split_init_head_tail

TSB_TUNED_COL = "TSB-tuned"

# 5 x 5 = 25 candidates, matching the order of EBB's 8 discounts x 3 structures.
DEFAULT_ALPHA_D: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5)
DEFAULT_ALPHA_P: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.45)


def _score(pred: pd.DataFrame, actual: pd.DataFrame, scale: pd.Series, criterion: str) -> float:
    merged = actual[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
    col = [c for c in merged.columns if c not in {"unique_id", "ds", "y", "index"}]
    if not col:
        return float("inf")
    yhat = merged[col[0]].to_numpy(dtype=float)
    y = merged["y"].to_numpy(dtype=float)
    err = np.abs(y - yhat)
    if criterion == "mae":
        return float(np.nanmean(err))
    # scaled absolute error, the point-forecast analogue of the scaled pinball
    # loss EBB selects on, so neither method is scored on its home turf.
    sc = merged["unique_id"].map(scale).to_numpy(dtype=float)
    return float(np.nanmean(err / np.maximum(sc, 1e-9)))


def tune_tsb_on_split(
    init_df: pd.DataFrame,
    freq: str = "D",
    alpha_d_grid: Sequence[float] = DEFAULT_ALPHA_D,
    alpha_p_grid: Sequence[float] = DEFAULT_ALPHA_P,
    val_ratio: float = 0.2,
    criterion: str = "scaled_mae",
) -> tuple[float, float, pd.DataFrame]:
    """Select (alpha_d, alpha_p) on the last ``val_ratio`` of the fitting window.

    Returns ``(alpha_d, alpha_p, diagnostics)``. Uses the identical
    :func:`_split_init_head_tail` helper as the EBB selector, so the two
    methods see the same validation block.
    """
    head_df, tail_df = _split_init_head_tail(init_df, val_ratio=val_ratio)
    if head_df.empty or tail_df.empty:
        return 0.5, 0.45, pd.DataFrame()

    _, M = _import_statsforecast()
    horizons = tail_df["unique_id"].value_counts()
    horizon_df = _prepare_horizons(horizons)

    scale = head_df.groupby("unique_id")["y"].apply(lambda v: float(np.mean(np.abs(v))))
    med = float(np.nanmedian(scale.to_numpy())) if len(scale) else 1.0
    scale = scale.replace(0.0, np.nan).fillna(max(med, 1e-9))

    rows: list[dict[str, float]] = []
    best = (0.5, 0.45, float("inf"))
    for a_d in alpha_d_grid:
        for a_p in alpha_p_grid:
            pred = _fit_predict_panel(
                train_df=head_df,
                horizon_df=horizon_df,
                models=[M["TSB"](alpha_d=float(a_d), alpha_p=float(a_p))],
                freq=freq,
                probabilistic=False,
                levels=None,
                n_jobs=1,
            )
            if pred.empty:
                continue
            val = _score(pred, tail_df, scale, criterion)
            rows.append({"alpha_d": float(a_d), "alpha_p": float(a_p), "score": val})
            if np.isfinite(val) and val < best[2] - 1e-12:
                best = (float(a_d), float(a_p), val)

    return best[0], best[1], pd.DataFrame(rows)


def fit_predict_tsb_tuned(
    init_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    freq: str = "D",
    val_ratio: float = 0.2,
    criterion: str = "scaled_mae",
    alpha_d_grid: Sequence[float] = DEFAULT_ALPHA_D,
    alpha_p_grid: Sequence[float] = DEFAULT_ALPHA_P,
) -> tuple[pd.DataFrame, dict[str, float], Optional[pd.DataFrame]]:
    """Tune on the internal split, then refit on the full window and predict."""
    a_d, a_p, diag = tune_tsb_on_split(
        init_df, freq=freq, alpha_d_grid=alpha_d_grid, alpha_p_grid=alpha_p_grid,
        val_ratio=val_ratio, criterion=criterion,
    )
    _, M = _import_statsforecast()
    horizon_df = _prepare_horizons(eval_df["unique_id"].value_counts())
    pred = _fit_predict_panel(
        train_df=init_df,
        horizon_df=horizon_df,
        models=[M["TSB"](alpha_d=a_d, alpha_p=a_p)],
        freq=freq,
        probabilistic=False,
        levels=None,
        n_jobs=1,
    )
    if pred.empty:
        return pd.DataFrame(), {"alpha_d": a_d, "alpha_p": a_p}, diag
    col = [c for c in pred.columns if c not in {"unique_id", "ds", "index"}]
    pred = pred.rename(columns={col[0]: TSB_TUNED_COL})[["unique_id", "ds", TSB_TUNED_COL]]
    return pred, {"alpha_d": a_d, "alpha_p": a_p}, diag
