# AISTATS 2027 Research Notes — TSB-HB v2

> ⚠️ **狀態橫幅（2026-07-29 加註）**：本檔是 2026-07-13～18 的**歷史研究計畫**，保留以供
> 對照 review 回應。**權威狀態請看 `paper_v2/STATUS.md`。** 以下兩點已被後續結果推翻或取代：
>
> 1. **模型已改名 REMIX**，工作標題定為 **"Learning to Pool and to Forget"**（非下方那個舊標題）。
>    pooling 結構改為與折扣 **聯合選擇**（`--hb-grouping select`），非固定用 mixture。
> 2. 下表「Learned mixture pooling 取代硬閾值」那一格暗示 learned partition 帶來準確度增益——
>    **這一點在 Round 7 被三次獨立證偽**（sample splitting、τ² credibility 正則化、群均值收縮），
>    詳見 `paper_v2/CHANGELOG_revision.md` Round 7 與 `STATUS.md` §2。
>    正確定位：learned partition 移除手設閾值且無總體代價，價值集中在最稀疏 cold-start 切片；
>    一階增益全部來自 forgetting 軸。

Working title direction (**superseded**, see banner): "Scalable Hierarchical
Empirical-Bayes Forecasting for Intermittent Demand with Learned Pooling and
Obsolescence Tracking"

## Why v2 (response map to AISTATS 2026 / UAI 2026 reviews)

| Review criticism | v2 answer |
|---|---|
| "No temporal adaptation / obsolescence tracking; fixed-origin gives a time-invariant forecast" (UAI bm76) | **Discounted sufficient statistics**: recency-weighted EB with discount `w` selected on an internal chronological split. Gives temporal forgetting inside the fixed-origin protocol and strengthens walk-forward. |
| "Taxonomy conditioning shows no benefit on M5" (UAI D9ks + own rebuttal) | **Learned mixture pooling**: EM mixture-of-priors with BIC-selected K replaces hard ADI/CV² thresholds. Taxonomy demoted to ablation baseline. |
| "Novelty incremental; HB intermittent models exist (Chapados 2014, Seeger 2016)" (UAI KGCf) | Positioning around **closed-form O(N) EB with learned structure**; direct comparison + Pareto (accuracy vs. compute) vs. sampling/VI-based HB. **TSB-as-limit theorem** (below) makes the bridge formal. |
| "Post-hoc calibration conflated with model" (UAI bm76) | Report **uncalibrated TSB-HB as the primary probabilistic result**; calibration as optional layer. |
| "Use scaled pinball; upper quantiles matter (Kolassa 2016)" (UAI KGCf) | `prob_pinball_scaled.csv` (M5-U SPL definition) now produced by `run_prob`; emphasis on q75/q90. |

## New model components

### 1. Discounted sufficient statistics (obsolescence tracking)

For item *i* with initialization window `t = 1..t0` and discount `w ∈ (0,1]`, define

```
n_i(w)  = Σ_t w^(t0−t)
s_i(w)  = Σ_t w^(t0−t) · 1{y_it > 0}
m_i(w)  = s_i(w)
ℓ̄_i(w) = Σ_{t: y>0} w^(t0−t) log y_it / m_i(w)
```

All EB blocks operate on these weighted pseudo-counts (weighted-likelihood /
power-prior EB). `w = 1` reproduces the released model exactly (verified by
regression test). Occurrence posterior mean becomes

```
p̂_i = (α_g + s_i(w)) / (α_g + β_g + n_i(w)).
```

**TSB-as-a-limit result (candidate Proposition 1 of v2).**
As `α_g, β_g → 0` (noninformative prior limit),

```
p̂_i → s_i(w) / n_i(w) = Σ w^(t0−t) p_it / Σ w^(t0−t),
```

which is exactly the (normalized) exponentially weighted moving average of the
occurrence indicators — i.e., **classical TSB's occurrence estimator with
smoothing parameter 1−w**. The same holds for the size block (EWMA of log-sizes
in the k_i → 0 limit). Hence classical TSB is the *no-pooling limit* of
discounted TSB-HB, and the two model axes are:

- **shrinkage strength** (φ_g = α_g+β_g; k_i): cross-sectional information sharing
- **discount w**: temporal information decay

TSB sits at (no pooling, w<1); released TSB-HB v1 sat at (pooling, w=1); v2
spans the full plane and *selects both coordinates from data*. This is the
unification claim the paper was missing.

**Discount selection.** Internal chronological split of the initialization
window (first 80% fit, last 20% validation; same no-leakage pattern as the
calibration layer). Score: per-series-scaled mean pinball over {0.5, 0.75, 0.9}.
Grid: {1.0, 0.999, 0.997, 0.995, 0.99, 0.98, 0.95, 0.90}. On Online Retail the
validation curve is convex with interior minimum at w = 0.95.

### 2. Learned mixture pooling (replaces fixed taxonomy)

K-component mixture over item-level sufficient statistics; each component k has
its own Beta-Binomial occurrence prior (α_k, β_k) and Normal random-effects
size prior (μ0_k, τ²_k, σ²_k), with mixing weights π_k.

- E-step: closed form (Beta-Binomial marginal × Normal marginal on ℓ̄_i).
- M-step: responsibility-weighted versions of exactly the EB objectives already
  in the released model.
- K chosen by BIC; deterministic quantile-based initialization (no seed
  sensitivity — preserves the "fully deterministic" property from the AISTATS
  rebuttal).
- Hard assignment feeds the existing group-conditioned pipeline unchanged, so
  taxonomy / global / mixture are strictly comparable ablations.
- Compatible with discounting: partition learned on the same discounted stats.

Synthetic check: 3 latent regimes → BIC selects K=3, purity 1.0, component
parameters recover truth.

## Experiment grid (fixed origin unless noted)

Variant column names produced by `--hb-variants all`:

| Model | Pooling | Discount |
|---|---|---|
| TSB-HB-Global | single pool | 1.0 |
| TSB-HB-Taxonomy | fixed ADI/CV² (paper v1) | 1.0 |
| TSB-HB-Mixture | learned (BIC K) | 1.0 |
| TSB-HB-Discount | taxonomy/base | auto |
| TSB-HB-Mix-Disc | learned on discounted stats | auto |

First results (fixed origin, point):

**Online Retail** (3,649 series):

- TSB-HB (v1 paper config) reproduces Table 1 exactly: MAE 5.7663 / RMSE 17.6930 / RMSSE 4.7875.
- **TSB-HB-Discount: MAE 5.5624** — best MAE overall, beats the TSB baseline's
  5.5736, removing v1's "loses MAE to TSB" weakness. RMSE slightly worse
  (17.84). All six TSB-HB variants beat every baseline on RMSE and RMSSE.
- Slices: Discount improves v1's MAE in every ADI/CV² regime and nearly every
  cold-start bin; best-in-class on Intermittent (1.2823 vs TSB 1.3219,
  ADIDA 1.3098).
- Discount auto-selected at 0.95; mixture BIC selects K=5 on OR.
- Global ≈ Taxonomy on OR too — further evidence the fixed taxonomy adds
  nothing; the pooling contribution must come from the learned partition.

**M5** (5,000-series sample, init ratio 2/3):

- **TSB-HB-Discount: RMSSE 2.2574** — best overall by a wide margin (next:
  AutoTheta 2.3143, v1 TSB-HB 2.3767). v1 *lost* RMSSE to AutoTheta/AutoARIMA
  on M5 (paper Table 5); obsolescence tracking flips that metric outright.
  ME also improves (−0.054 vs −0.148).
- All TSB-HB variants still beat all baselines on MAE and RMSE.
- Mixture ≈ Global on M5 point metrics (BIC selects K=6).

**Combined narrative:** v1 lost exactly one headline metric per dataset
(MAE to TSB on OR; RMSSE to AutoTheta on M5). The discount component wins both
back while keeping every other lead. "Hierarchical shrinkage x temporal
discounting" spans the model plane; classical TSB is the no-pooling corner.

## Probabilistic results (OR, fixed origin, released config B=20)

| Config | SPL q50 | q75 | q90 | SPL mean | Cov@80 | AIW@80 |
|---|---|---|---|---|---|---|
| v1 (taxonomy, w=1) | 0.9198 | 1.3533 | 1.5305 | 0.8900 | 0.8454 | 10.03 |
| Mixture (w=1) | 0.9198 | 1.3535 | 1.5353 | 0.8912 | 0.8388 | 9.17 |
| Discount (w=auto) | 0.9202 | 1.3560 | 1.5408 | 0.8927 | 0.8365 | **8.96** |
| Mix-Disc | 0.9207 | 1.3582 | 1.5477 | 0.8946 | 0.8571 | 10.16 |

TSB-HB (any config) beats every baseline at every scaled-pinball quantile
(best baseline: AutoARIMA SPL mean 1.016 vs 0.890). Variants are statistically
tied on SPL; **Discount delivers 11% narrower intervals (AIW 8.96 vs 10.03)
with coverage closer to nominal (0.8365 vs 0.8454)** — a strictly better
sharpness–calibration tradeoff (GoalScore@80: 9.29 vs 10.49).

## Paired significance tests (OR point, per-series, Holm-adjusted)

- TSB-HB-Discount vs TSB-HB v1 (MAE): mean diff −0.234, p_t < 1e-24,
  p_Wilcoxon < 1e-7 → the discount improvement is systematic, not incidental.
- TSB-HB-Discount vs every baseline (RMSE): all negative, all p < 0.005.
- vs TSB on MAE: mean diff −0.110 (p_t = 0.079, Wilcoxon < 1e-166 with positive
  median diff — TSB marginally better on median series, Discount much better in
  the tail; report honestly as "no longer loses MAE, wins on average").
- Script: `scripts/analysis/paired_tests.py`; predictions persisted via
  `run_point --save-predictions`.

## New CLI

- `run_point` / `run_prob`: `--hb-grouping {legacy,global,taxonomy,mixture}`,
  `--hb-mixture-k INT` (0=BIC), `--hb-fit-discount FLOAT|auto`,
  `--hb-variants {none,all}` (run_point, fixed protocol).
- `run_prob` now also writes `prob_pinball_scaled.csv` (SPL, M5-U definition).

## TODO / next

- [x] OR probabilistic runs (released config B=20): paper vs mixture vs discount vs mix-disc.
- [x] M5 point with variants.
- [x] Theory: `docs/theory_v2.md` — Prop 1 (TSB = self-consistently-initialized
      discounted-mean limit, exact identity + geometric bound), Prop 2
      (discounting keeps shrinkage active forever: λ ≤ 1/(1+(1−w)φ)), Prop 3
      (drift MSE bound) + Cor 3.1 (w* → 1 as drift → 0, O(δ^{2/3}) tracking
      risk). All numerically verified.
- [x] Baselines restored from git 592fb71: iETS (R smooth, cached),
      Tweedie AR-GLM, DRP wiring, taxonomy4 M5 mode, --max-series; re-wired
      into v2 run_point/run_prob (grouping/discount/variants/SPL preserved).
      iETS needs `--iets-rscript "C:\Program Files\R\R-4.5.3\bin\Rscript.exe"`.
      OR fixed prob cache: outputs/iets_cache_auto_50.csv etc.
- [ ] Walk-forward v1 vs discounted (running).
- [ ] Datasets to add (TweedieGP, IJF 2025, Table 1): Auto (3,000 monthly,
      T=18, h=6; gluon-ts intermittent-datasets branch), Carparts (2,503
      monthly, T=45, h=6; zenodo.org/records/4656022), RAF (5,000 monthly,
      T=72, h=12; gluon-ts branch). Needs monthly-frequency support in
      data_loading + protocols (freq="M").
- [ ] Full M5 (30,490), Seeger-2016-style VI hurdle baseline, DRP full run.
