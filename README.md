# REMIX (TSB-HB)

Reference implementation for **REMIX** (Recency-weighted Empirical-bayes MIXtures),
*"Learning to Pool and to Forget"* — the current paper, targeting AISTATS 2027.
Paper source: `paper_v2/main.tex`. **Project state, open questions and the
prioritized todo list live in `paper_v2/STATUS.md`** — read that first.

The code also reproduces the earlier submission (`Taxonomy-Conditioned Hierarchical
Bayesian TSB Models for Heterogeneous Intermittent Demand Forecasting`); see
`PAPER_REPRODUCTION.md`. The `--hb-*` defaults below reproduce that release
exactly, so the v1 numbers stay reachable.

- REMIX selects the pooling structure (single pool / ADI/CV^2 taxonomy /
  BIC-selected mixture) **jointly** with the recency discount `w`, on an internal
  chronological validation split of the initialization window
  (`--hb-grouping select`). No out-of-sample data is touched by selection.
- Fixed-origin is the main evaluation protocol; walk-forward is kept as a robustness check.
- Probabilistic forecasting uses the released configuration: conjugate variance with `nu=20`, bootstrap averaging with `B=20`, and location-scale calibration on the initialization window only. The paper reports the **uncalibrated** run as the primary probabilistic result.
- Generated experiment artifacts live under `outputs/` and are not committed.

## Setup

1. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create the project environment:

```bash
uv sync
```

3. Optional DeepAR extras:

```bash
uv sync --extra deepar
```

4. Inspect CLI help:

```bash
uv run python -m experiments.run_point --help
uv run python -m experiments.run_prob --help
```

## Data

### Online Retail

Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place it at `data/online_retail.csv` or `data/Online_Retail.csv`.

### M5

Download `sales_train_evaluation.csv` (or `sales_train_validation.csv`) and `calendar.csv` from the [Kaggle M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data), then place both files in `data/`.

Optional one-time conversion to long format:

```bash
uv run python -m tools.convert_m5_to_long \
  --input data/sales_train_evaluation.csv \
  --output data/m5_evaluation_long.csv
```

## Paper Reproduction

Run the canonical paper suite with:

```bash
bash scripts/experiments/run_paper_suite.sh scripts/experiments/config.paper.env
```

The generated outputs map directly to the paper-facing runs:

- `outputs/paper_reproduction/online_retail/point_fixed/point_metrics.csv`
- `outputs/paper_reproduction/online_retail/point_fixed/segmentation_rmsse.csv`
- `outputs/paper_reproduction/online_retail/prob_fixed/prob_metrics.csv`
- `outputs/paper_reproduction/online_retail/prob_fixed/coverage_summary.csv`
- `outputs/paper_reproduction/online_retail/prob_fixed/prob_pinball.csv`
- `outputs/paper_reproduction/online_retail/point_walk_forward/point_metrics.csv`
- `outputs/paper_reproduction/m5/point_fixed/point_metrics_m5.csv`

The exact commands and parameter choices are documented in [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md).

## Main Scripts

- `experiments.run_point`
  Default `--baseline-mode paper` matches the point-forecast baselines used in the paper. `--m5-hierarchy-mode off` is the paper default for M5.
- `experiments.run_prob`
  Default `--baseline-mode paper` plus default `--hb-calibration-mode location_scale` matches the released probabilistic configuration.

## REMIX model components (AISTATS 2027)

Both `run_point` and `run_prob` accept:

- `--hb-grouping {legacy,global,taxonomy,mixture,select}` — pooling structure.
  `mixture` learns the pooling partition with an EM mixture-of-priors
  (Beta-Binomial occurrence x Normal random-effects size per component);
  K is selected by BIC unless `--hb-mixture-k` is given.
  **`select` is the REMIX model**: it scores {global, taxonomy, mixture} x the
  discount grid on the internal validation split and reports the joint winner
  to `remix_selection_diagnostics.csv`.
- `--hb-fit-discount FLOAT|auto` — exponential recency weighting of the
  initialization-window sufficient statistics (obsolescence tracking).
  `auto` selects the discount on an internal 80/20 chronological split of the
  initialization window (no out-of-sample data touched). `1.0` reproduces the
  released model exactly.
- `--hb-label-split {off,parity,chrono}` — sample-split the initialization
  window so the pooling partition is learned on one half and the group
  hyperparameters are estimated on the other; per-item sufficient statistics
  still use the full window. Removes the post-selection dependence between
  partition and variance components. `parity` (interleaved) is the meaningful
  mode: both halves span the whole window, so the recency profile survives.
- `--hb-hyper-shrink {off,credibility}` — regularize the group-level variance
  components by Buhlmann credibility toward the global value; weights are
  data-determined, no tuned parameter, and the map is the identity for a single
  pool. Prevents the collapse of `tau^2` that otherwise drives the size
  credibility weight to zero under learned partitions plus strong discounting.
- `run_point --hb-variants all` (fixed protocol) additionally fits the legacy
  variant suite (`TSB-HB-Global/Taxonomy/Mixture/Discount/Mix-Disc`), with
  diagnostics in `tsbhb_variant_diagnostics.csv`. **Retired for the paper** —
  the ablation table is now built from independent per-cell CLI runs.
- `run_prob` also writes `prob_pinball_scaled.csv` (M5-Uncertainty-style
  scaled pinball loss, per quantile plus mean).

Both new flags default to `off`, which reproduces the released model
bit-identically (verified against `outputs/aistats2027/ablate_or_point_mixture_auto`).

Documentation map: `paper_v2/STATUS.md` (authoritative state, contributions,
todo), `paper_v2/CHANGELOG_revision.md` (what was done, append-only),
`paper_v2/QUESTIONS.md` (open author decisions),
`paper_v2/NUMBERS.md` + `NUMBERS_tables.md` (every paper number to its source
file), `outputs/aistats2027/MANIFEST.md` (run inventory),
`docs/theory_v2.md` (propositions), `docs/aistats2027_research_notes.md`
(historical research plan — partly superseded, see its banner).
- `experiments.run_coverage_pit`
  Thin wrapper around `run_prob` for the same probabilistic pipeline.
- `experiments.run_grid`, `experiments.run_ablation`, `experiments.run_deepar`
  Supplemental experiments that are not required for the canonical reproduction path.

Supplemental baseline modes are still available when needed:

- `--baseline-mode extended`: paper baselines plus hurdle baselines.
- `--baseline-mode hurdle_only`: TSB-HB plus hurdle baselines only.
- `--baseline-mode hb_only`: TSB-HB only.

## Repository Layout

- `src/models/tsb_hb.py`: hierarchical Bayesian TSB model and forecasting logic.
- `src/experiments/`: point, probabilistic, ablation, and benchmark entry points.
- `src/data_loading.py`, `src/metrics.py`, `src/plotting.py`: shared preprocessing, metrics, and figures.
- `scripts/experiments/run_paper_suite.sh`: canonical batch runner for the paper configuration.

## References

1. Azul Garza, Max Mergenthaler Canseco, Cristian Challu, and Kin G. Olivares.  
   **StatsForecast: Lightning fast forecasting with statistical and econometric models.**  
   [https://github.com/Nixtla/statsforecast](https://github.com/Nixtla/statsforecast)

2. Addison Howard, inversion, Spyros Makridakis, and Vangelis.  
   **M5 Forecasting - Accuracy.**  
   [https://kaggle.com/competitions/m5-forecasting-accuracy](https://kaggle.com/competitions/m5-forecasting-accuracy)

3. Daqing Chen.  
   **Online Retail.** UCI Machine Learning Repository, 2015.  
   [https://doi.org/10.24432/C5BW33](https://doi.org/10.24432/C5BW33)
