# Reproducing the Public freMTPL2 Run

These commands reproduce the open-source companion workflow from a fresh clone.
They assume Python 3.10 to 3.12 and a TensorFlow-backed Keras install.

## 1. Install

```bash
cd open_source_frmtpl_scaling
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Prepare the Data

If `data/FRMTPL.csv` is missing, regenerate it from Mario Wüthrich's corrected
RDA file:

```bash
Rscript scripts/prepare_wuthrich_data.R
```

The expected split is 610,206 learning rows and 67,801 test rows.

## 3. Smoke Test

```bash
python scripts/run_experiment.py --smoke --results-dir /tmp/frmtpl_smoke_results
python scripts/make_figures.py \
  --ensemble-scores /tmp/frmtpl_smoke_results/ensemble_scores.csv \
  --run-scores /tmp/frmtpl_smoke_results/run_scores.csv \
  --scaling-fits /tmp/frmtpl_smoke_results/scaling_fits.csv \
  --figures-dir /tmp/frmtpl_smoke_figures
pytest
```

The smoke run trains only GLM and FFN for one fraction, one seed, and one epoch.
It is intended to catch installation or data-shape problems, not to reproduce
the reference metrics.

## 4. Full Scaling Sweep

```bash
python scripts/run_experiment.py
python scripts/make_figures.py
```

The full run trains five compact model families across six nested training
fractions with three seeds per model/fraction.

## 5. Business Outcome Diagnostics

```bash
python scripts/create_outcome_diagnostics.py
```

This retrains the full-data GLM and TabM-mini recipes, writes grouped outcome
summaries, and refreshes the age-band and age-by-bonus-malus figures. To avoid
writing the row-level prediction extract:

```bash
python scripts/create_outcome_diagnostics.py --no-row-predictions
```

To rebuild only the figures from existing grouped summaries:

```bash
python scripts/create_outcome_diagnostics.py --plots-only
```

## 6. Supplementary Material

From the repository root:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode supplementary_material_open_source_frmtpl_standalone.tex
```

The generated PDF is
`paper/supplementary_material_open_source_frmtpl_standalone.pdf`.
