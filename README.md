# EBB

Research code for **When Does Pooling Pay? Credibility and Resolution under
Forgetting in Intermittent-Demand Forecasting**.

**Authors:** [Zong-Han Bai](https://github.com/HummerQAQ) and
[Po-Yen Chu](https://github.com/brianCHUCHU). The two authors contributed equally.

## Paper

- [arXiv:2511.12749v3](https://arxiv.org/abs/2511.12749v3)
- [DOI: 10.48550/arXiv.2511.12749](https://doi.org/10.48550/arXiv.2511.12749)

EBB is the forecaster described in the revised work, which develops the earlier
TSB-HB work around joint pooling and forgetting. Internal modules retain the
names `eb_hurdle` and `ebhurdle`, and some output columns use `EB-Hurdle`.
`EBB` / `ebb` denotes that model with the selected pooling structure and discount.

## Installation

Use **Python 3.10** (the supported range in `pyproject.toml`) and
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --extra test
export ARS_SF_NJOBS=1
uv run python -m experiments.run_point --help
uv run python -m experiments.run_prob --help
```

The distribution name is `ebb`; existing Python module names are retained.
The main environment uses the pinned Python dependencies in `pyproject.toml`.
The tests additionally require pytest and SymPy:

```bash
uv run python -m pytest tests/test_determinism.py tests/test_theory_symbolic.py
```

TweedieGP uses its authors' released implementation in a separate environment
and is not redistributed here. The DeepAR integrity workflow uses a separate
GluonTS 0.11.12 / MXNet 1.7 environment. iETS scripts require R and its forecasting
packages; some archived runners contain a Windows R executable path that must
be explicitly configured for the local installation. These separate environments
are required for those baselines, rather than substitutes in the main environment.

## Data

All five panels use public data, downloaded separately into `data/`:

- **Online Retail:** [UCI dataset](https://archive.ics.uci.edu/dataset/352/online+retail).
- **M5:** [competition files](https://www.kaggle.com/c/m5-forecasting-accuracy/data),
  including sales and calendar files. `preprocess_m5` draws the seed-42 sample
  of 5,000 series.
- **Auto, Carparts, RAF:** monthly panels distributed with the TweedieGP work;
  convert them to long format using `src/tools/convert_intermittent_datasets.py`.

Raw datasets, full prediction archives, external baseline implementations and
internal design notes are not included. Input requirements and run settings are
specified in the individual scripts.

## Repository layout

- `src/models/eb_hurdle.py`: hierarchical empirical-Bayes hurdle model,
  discounted sufficient statistics, joint selection and online updates.
- `src/models/mixture_pooling.py`: learned partition using an EM mixture of priors
  and BIC.
- `src/models/conformal.py`, `baselines.py`, and other model modules: comparators.
- `src/experiments/`: fixed-origin and walk-forward drivers and scoring.
- `scripts/integrity/`, `scripts/rebuild/`, `scripts/analysis/`: experiment,
  rebuilding and reporting scripts.
- `results/`: small, saved result files grouped by topic.
- `tests/`: symbolic and determinism checks.

## Reproduction boundaries

`results/` preserves the supplied result CSVs. Each experiment script writes to
its configured folder under `outputs/`; the result CSVs are selected copies,
not the full intermediate output archive. Some reporting scripts require earlier
run outputs or manuscript source folders that are not included. Run their
prerequisite experiments and provide the required inputs before using them;
installing the package alone does not reproduce every table and figure.

Run-folder names use the neutral prefixes `outputs/paper_runs/`,
`outputs/paper_rebuild/`, and `paper_v2/v*_paper*/`. If using existing local
artifacts, place them at the paths specified by each script. The migration
preserves numerical code and results while renaming these path strings.

## Paper item -> script -> result file
| Paper item | Script | Result file |
|---|---|---|
| Selection (24 candidates), credibility by panel | `scripts/integrity/t1_leakage_safe_selection.py` | `results/selection/` |
| Table 1, fixed origin | `scripts/integrity/p0_ebb_external_corrected.py`, `f57_significance_rescore.py` | `results/fixed_origin/rescored_tab_prob.csv`, `ebb_corrected_rows.csv` |
| Table 1, walk-forward | `f1_wf_ebb_corrected.py`, `w2_wf_monthly_roster.py`, `h4_m5_wf_plan_c.py`, `w1_wf_tweediegp.py`, `w8_wf_aci_ebb.py`, `a1_aci_selector.py`, `w3_assemble_wf_tables.py` | `results/walk_forward/wf_all_panels_wide.csv`, `wf_ebb_corrected.csv` |
| Table 1, layout | `scripts/integrity/v8_tab_main.py` | (LaTeX) |
| DeepAR | `d1_deepar_export.py`, `d2_deepar_runner.py` (GluonTS 0.11.12 / MXNet 1.7 environment), `d3_deepar_score.py`, `d5_deepar_paired_all.py`, `d6_deepar_tex.py` | `results/deepar/` |
| Paired tests | `f57_significance_rescore.py`, `w9_wf_significance.py` | `results/fixed_origin/spl_significance_corrected.csv`, `results/walk_forward/spl_significance_wf.csv` |
| Short-history experiment | `c1_coldstart.py`, `c2_coldstart_report.py` | `results/short_history/coldstart_wide.csv`, `coldstart_all.csv` |
| Prior on / off, by block | `c3_coldstart_nopool.py`, `c5_coldstart_blocks.py`, `c4_coldstart_nopool_tex.py` | `results/short_history/coldstart_nopool_wide.csv`, `coldstart_blocks_wide.csv` |
| Controlled surface | `s1b_separation_surface_v2.py`, `s2_real_separation.py`, `s3_separation_report.py` | `results/surface/` |
| Figure with both panels; appendix figures on the prior gain and on DeepAR | `scripts/integrity/v8_figs.py` | (figures), `results/deepar/deepar_vs_credibility.csv` |
| Pre-fit screen | `r1_room_synthetic.py` ... `r5_room_rows.py`, `room_features.py` | (regenerated; seeds in the scripts) |
| No-pooling candidate | `t2_selection_with_none.py`, `t3_or_none_walkforward.py`, `t4_none_candidate_tex.py` | `results/none_candidate/` |
| Selection weighting check | `t5_selection_weighting_check.py` | `results/none_candidate/selection_weighting_check.csv` |
| Appendix point figures | `h5_point_figs_audited.py` | `results/fixed_origin/point_figs_audited_check.csv` |

Each script writes its output to a run folder under `outputs/`, named in the script; the files
under `results/` are copies of those outputs, grouped by topic.
EBB is deterministic given the data: rerunning a script reproduces its file to numerical precision.

## Citation

See `CITATION.bib` for the citation below.

```bibtex
@misc{bai2025ebb,
  title = {When Does Pooling Pay? Credibility and Resolution under
           Forgetting in Intermittent-Demand Forecasting},
  author = {Zong-Han Bai and Po-Yen Chu},
  year = {2025},
  eprint = {2511.12749},
  archivePrefix = {arXiv},
  primaryClass = {stat.ML},
  url = {https://arxiv.org/abs/2511.12749v3},
  note = {Version 3}
}
```
