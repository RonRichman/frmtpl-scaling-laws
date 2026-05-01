# freMTPL2 Scaling Laws for Actuarial Ratemaking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RonRichman/frmtpl-scaling-laws/blob/main/notebooks/01_colab_frmtpl_scaling.ipynb)

This repository is a small, open-source companion to the paper:

Richman, R. *Scaling Laws, Tabular Data and Actuarial Ratemaking Models*.
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6073948

The goal is not to reproduce the proprietary multi-million-row experiment
numerically. Instead, the package reproduces the main experimental workflow on
Mario Wüthrich's corrected public freMTPL2 frequency data:

- exposure-scaled Poisson frequency modeling;
- nested training fractions;
- three random seeds per model/fraction;
- seed-averaged ensemble scoring;
- GLM, FFN, MultiCLS Transformer, swap-SSL Transformer, and TabM-mini models;
- descriptive data-scaling fits of `L(N) = L_inf + A N^-alpha`;
- business-facing GLM versus TabM-mini outcome diagnostics.

The intended reader is an actuary who wants to inspect the mechanics behind the
paper in a runnable, smaller-scale setting.

## Quick Start

```bash
git clone https://github.com/RonRichman/frmtpl-scaling-laws.git
cd frmtpl-scaling-laws
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_experiment.py --smoke
python scripts/make_figures.py
```

The smoke command runs GLM and FFN on two small training fractions, one seed,
and one epoch. That is enough to populate `results/scaling_fits.csv` while
keeping the run short.

If `data/FRMTPL.csv` is not present, regenerate it from the corrected RDA file:

```bash
Rscript scripts/prepare_wuthrich_data.R
```

The full default run uses six training fractions and three seeds:

```bash
python scripts/run_experiment.py
python scripts/make_figures.py
```

To add business-facing GLM versus TabM-mini outcome diagnostics:

```bash
python scripts/create_outcome_diagnostics.py
```

To rebuild only the diagnostic plots from existing grouped CSV summaries:

```bash
python scripts/create_outcome_diagnostics.py --plots-only
```

On Colab, open `notebooks/01_colab_frmtpl_scaling.ipynb` from GitHub. The first
code cell clones `https://github.com/RonRichman/frmtpl-scaling-laws.git` into
`/content/frmtpl-scaling-laws` if the package files are not already present and
installs the local package with `pip install -e ".[dev]"`. The public Colab
badge works without additional GitHub credentials. The notebook requires a GPU
runtime by default and stops before training if TensorFlow cannot see one.

If Colab prints `Could not find cuda drivers` or `GPU will not be used`, the
session is running on CPU. Choose `Runtime > Change runtime type > Hardware
accelerator > GPU`, restart the session, and run the notebook again. For a
short CPU-only smoke/debug run, set `REQUIRE_GPU = False` in the notebook or
omit `--require-gpu` when using `scripts/run_experiment.py`.

## Data

`data/FRMTPL.csv` is generated from Mario Wüthrich's corrected
`freMTPL2freq.rda`:

https://people.math.ethz.ch/~wueth/Lecture/freMTPL2freq.rda

The CSV includes the learning/test split from Wüthrich and Merz, Listing 5.2:
`RNGversion("3.5.0")`, `set.seed(500)`, and a 90% learning sample. The resulting
split has 610,206 learning rows and 67,801 test rows. The extra `sample_unif`
column is used only to make nested learning subsets for the scaling-law sweep.

The modeling target is `ClaimNb`, and `Exposure` is used as an offset. The
workflow excludes `ClaimTotal`, so the public demo focuses on claim frequency
rather than severity or pure premium. For most dataset questions, especially
the cleaning rationale and variable definitions, see Appendix B of Wüthrich and
Merz, *Statistical Foundations of Actuarial Learning and its Applications*:

https://link.springer.com/book/10.1007/978-3-031-12409-9

See `DATA_LICENSE.md` for data attribution and license notes.

## Outputs

The experiment writes:

- `results/run_scores.csv`: one row per seed/model/fraction;
- `results/ensemble_scores.csv`: seed-averaged predictions scored by model/fraction;
- `results/scaling_fits.csv`: fitted scaling-law summaries;
- `figures/data_scaling.png`;
- `figures/reducible_loss_fits.png`;
- `figures/parameter_performance.png`;
- `figures/glm_lift.png`;
- `figures/best_model_lift.png`.

The diagnostic script writes:

- `results/outcome_diagnostics/full_data_glm_tabm_scores.csv`;
- `results/outcome_diagnostics/portfolio_outcome_summary.csv`;
- `results/outcome_diagnostics/driver_age_band_summary.csv`;
- `results/outcome_diagnostics/driver_age_bonusmalus_interaction_summary.csv`;
- `results/outcome_diagnostics/driver_age_vehicle_age_interaction_summary.csv`;
- `results/outcome_diagnostics/top_age_bonusmalus_differences.csv`;
- `figures/age_band_frequency_comparison.png`;
- `figures/age_bonusmalus_interaction_heatmap.png`.

It can also write
`results/outcome_diagnostics/full_data_test_predictions_glm_tabm.csv`, a
row-level held-out prediction extract. That file is intentionally ignored by
git because it is regenerated and much larger than the grouped summaries.

The bundled reference run was produced with the default six fractions and three
seeds per model. Full-data ensemble scores are:

| Model | Train Poisson deviance | Test Poisson deviance |
|---|---:|---:|
| GLM | 0.240812 | 0.241440 |
| FFN small | 0.238796 | 0.239948 |
| MultiCLS Transformer small | 0.238617 | 0.240024 |
| MultiCLS+SSL Transformer small | 0.238979 | 0.240128 |
| TabM-mini small | 0.238485 | 0.239787 |

## Notes for Actuaries

Each model predicts a claim rate per exposure-year and then multiplies by
exposure:

```text
predicted count = Exposure * exp(model log-rate)
```

This is the same offset logic used in a Poisson GLM. Poisson deviance is
reported because it is the standard likelihood-based metric for count frequency
models. A lower deviance means the model assigns higher likelihood to the
observed claim counts.

The scaling exponent `alpha` summarizes how quickly the model approaches its
fitted loss floor as more training rows are added. Larger `alpha` means stronger
data scaling within this experiment. Because freMTPL2 is much smaller than the
portfolio studied in the paper, these exponents should be read as descriptive
summaries of the public run, not as production scaling laws.

## Release Hygiene

This repository is designed to be pushed publicly. Source code, tests, the small
summary CSVs, and figures are suitable for version control. Python cache files,
model weights, and the row-level prediction extract are ignored. The proprietary
portfolio used in the paper is not included.
