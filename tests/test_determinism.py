"""Determinism test (paper claim: repeated runs are bit-identical).

Fits the full REMIX pipeline twice on a synthetic panel and asserts every
output (selection, posteriors, point and quantile forecasts) is exactly
identical. Run: py tests/test_determinism.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from models.mixture_pooling import mixture_group_labels
from models.tsb_hb import fit_tsb_hb, predict_tsb_hb, select_pooling_and_discount


def make_panel(n_series: int = 80, T: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for i in range(n_series):
        p = rng.uniform(0.05, 0.5)
        occ = rng.random(T) < p
        y = np.where(occ, np.round(np.exp(rng.normal(1.0, 0.7, T)), 3), 0.0)
        ds = pd.date_range("2022-01-01", periods=T, freq="D")
        rows.append(pd.DataFrame({"unique_id": f"s{i}", "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def run_once(df: pd.DataFrame):
    mix = mixture_group_labels(df, k=0)
    cands = {"global": None, "mixture": mix.labels}
    name, w, diag = select_pooling_and_discount(df, cands)
    params = fit_tsb_hb(df, group_labels=cands[name], item_variance_mode="conjugate",
                        item_variance_shrink_strength=20.0, fit_discount=w,
                        bootstrap_draws=20, bootstrap_seed=42)
    point = predict_tsb_hb(params, df[["unique_id", "ds"]].drop_duplicates(), quantiles=None)
    quant = predict_tsb_hb(params, df[["unique_id", "ds"]].drop_duplicates(),
                           quantiles=[0.1, 0.5, 0.9])
    return name, w, mix.labels, diag, point, quant


def test_bit_identical():
    df = make_panel()
    r1 = run_once(df)
    r2 = run_once(df)
    assert r1[0] == r2[0], "selected structure differs"
    assert r1[1] == r2[1], "selected discount differs"
    pd.testing.assert_series_equal(r1[2], r2[2])
    pd.testing.assert_frame_equal(r1[3], r2[3])
    pd.testing.assert_frame_equal(r1[4], r2[4], check_exact=True)
    pd.testing.assert_frame_equal(r1[5], r2[5], check_exact=True)


if __name__ == "__main__":
    test_bit_identical()
    print("determinism test passed: two runs bit-identical")
