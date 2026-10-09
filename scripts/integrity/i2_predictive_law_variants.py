"""Stage-1 experiment: alternative predictive laws on the SAME fitted EBB
posterior (no change to fitting, pooling, selection, or theory).

For each panel, fit EBB once at its audited configuration, take the item
posterior (pi_hat, mu_hat, s2 = sigma_hat^2 + V_mu) and emit quantiles under
several positive-size laws sharing that posterior:

  lognormal      current law (reference; single fit, no B=20 averaging)
  logt5/logt10   log-Student-t with nu=5 / 10 on the log scale (heavier tail)
  gamma_mm       Gamma moment-matched to the log-normal mean/variance (lighter)
  lognormal_x1.2 log-normal with the log-scale s inflated by 1.2 (wider)

All variants keep P(Y=0) = 1 - pi_hat. Scored with the pipeline's
_scaled_pinball_table at q in {0.1,...,0.9} plus far tail.

Output: outputs/<date>/predictive_law_variants.csv

Usage: py scripts/integrity/i2_predictive_law_variants.py [panel ...]
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from data_loading import (
    load_generic_long, load_m5_long, load_online_retail, preprocess_m5,
    preprocess_online_retail, train_eval_split_fixed_origin, train_eval_split_last_h,
)
from experiments.run_prob import _enforce_monotonic_quantiles, _scaled_pinball_table
from models.mixture_pooling import mixture_group_labels
from models.eb_hurdle import fit_eb_hurdle
from utils import default_m5_calendar_file, default_m5_sales_file, set_seed

OUT = ROOT / "outputs" / date.today().isoformat()
OUT.mkdir(parents=True, exist_ok=True)
Q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
CENTRAL = Q[:5]
CONFIG = {
    "online_retail": ("global", 0.95), "m5": ("mixture", 0.95),
    "auto": ("global", 0.99), "carparts": ("global", 0.90), "raf": ("mixture", 0.997),
}


def load_panel(name):
    if name == "online_retail":
        df = preprocess_online_retail(load_online_retail(ROOT / "data/online_retail.csv"))
        return train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    if name == "m5":
        set_seed(42)
        s, c = load_m5_long(default_m5_sales_file(), default_m5_calendar_file())
        return train_eval_split_fixed_origin(preprocess_m5(s, c, sample_size=5000),
                                             init_ratio=2 / 3, min_len=1)
    h = {"auto": 6, "carparts": 6, "raf": 12}[name]
    return train_eval_split_last_h(load_generic_long(ROOT / f"data/{name}_long.csv"), h=h)


def variant_quantiles(pi, mu, s2, law):
    """Return array [n_rows, len(Q)] of hurdle quantiles under `law`."""
    s = np.sqrt(np.maximum(s2, 1e-12))
    pi = np.clip(pi, 1e-9, 1 - 1e-9)
    out = np.zeros((len(pi), len(Q)))
    for k, q in enumerate(Q):
        adj = (q - (1.0 - pi)) / pi
        pos = adj > 0
        a = np.clip(adj[pos], 1e-9, 1 - 1e-9)
        if law == "lognormal":
            v = np.exp(mu[pos] + s[pos] * stats.norm.ppf(a))
        elif law.startswith("logt"):
            nu = float(law[4:])
            v = np.exp(mu[pos] + s[pos] * stats.t.ppf(a, df=nu))
        elif law == "gamma_mm":
            m = np.exp(mu[pos] + s2[pos] / 2.0)
            var = (np.exp(s2[pos]) - 1.0) * np.exp(2 * mu[pos] + s2[pos])
            shape = np.maximum(m ** 2 / np.maximum(var, 1e-12), 1e-3)
            scale = m / shape
            v = stats.gamma.ppf(a, a=shape, scale=scale)
        elif law == "lognormal_x1.2":
            v = np.exp(mu[pos] + 1.2 * s[pos] * stats.norm.ppf(a))
        else:
            raise ValueError(law)
        out[pos, k] = v
    return out


def main():
    panels = sys.argv[1:] or ["online_retail", "auto", "carparts", "raf", "m5"]
    rows = []
    for panel in panels:
        init, ev = load_panel(panel)
        structure, w = CONFIG[panel]
        labels = (mixture_group_labels(init, k=0, fit_discount=w).labels
                  if structure == "mixture" else None)
        params = fit_eb_hurdle(init, group_labels=labels, item_variance_mode="conjugate",
                            item_variance_shrink_strength=20.0, fit_discount=w)
        uid = ev["unique_id"].to_numpy()
        pi = np.nan_to_num(params.p_posterior.reindex(uid).to_numpy(float), nan=0.5)
        mu = np.nan_to_num(params.shrunk_mean_log.reindex(uid).to_numpy(float), nan=0.0)
        s2 = np.nan_to_num(params.sigma_sq_process.reindex(uid).to_numpy(float)
                           + params.posterior_var_mu.reindex(uid).to_numpy(float), nan=1.0)
        evv = ev[["unique_id", "ds", "y"]].copy()
        evv["unique_id"] = evv["unique_id"].astype(str)
        evv["ds"] = pd.to_datetime(evv["ds"])
        for law in ("lognormal", "logt10", "logt5", "gamma_mm", "lognormal_x1.2"):
            qa = variant_quantiles(pi, mu, s2, law)
            qd = evv[["unique_id", "ds"]].copy()
            for k, q in enumerate(Q):
                qd[f"q_{q}"] = np.maximum(qa[:, k], 0.0)
            qd["model"] = law
            qd = _enforce_monotonic_quantiles(qd, quantiles=Q)
            merged = evv.merge(qd, on=["unique_id", "ds"], how="inner")
            spl = _scaled_pinball_table(merged, init_set=init, quantiles=Q)
            spl = spl[spl["model"] == law]
            central = spl[spl["quantile"].isin(CENTRAL)]["scaled_pinball"]
            rec = {"panel": panel, "law": law, "spl_mean": round(float(central.mean()), 4)}
            for _, r in spl.iterrows():
                rec[f"q{r['quantile']}"] = round(float(r["scaled_pinball"]), 4)
            rows.append(rec)
            print(f"[{panel}] {law:15s} mean {rec['spl_mean']:.4f}  q90 {rec['q0.9']:.4f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT / "predictive_law_variants.csv", index=False)
    print("wrote predictive_law_variants.csv")


if __name__ == "__main__":
    main()
