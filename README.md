# Toy Watchtower MMM

This is a small standalone teaching repo for a Watchtower-style marketing mix model.

It intentionally mirrors the production flow without production complexity:

```text
simulate raw panel data
-> normalize and write HDF5
-> train one 1024-epoch multiplicative Keras MMM
-> export artifact dataframes
-> inspect Plotly decomposition, convergence, intermediaries, and hierarchy parameters
```

The model equation is:

```text
Y = Baseline * multiplicative_impacts
  + (multiplicative_impacts ** alpha) * ROI_impacts
```

Price is handled by a dedicated `pricing.py` layer instead of a generic linear feature.
The walkthrough uses one training stage only; there is no calibration pretraining or calibration loss.

The main training knobs live in `config/toy_mmm_hyperparameters.yml`:

- `training.learning_rate`
- `training.epochs`
- `training.snapshot_interval`
- embedding-wise regularization blocks for `baseline`, `pricing`, `sigmoid_curve`, and `beta_gamma`
- separate multiplicative feature embedding blocks for `feature_beta.distribution`, `feature_beta.temperature`, and `feature_beta.macro`
- ROI bounds under `roi.sigmoid_curve` and `roi.beta_gamma`
- alpha multiplier range under `roi.alpha_mult`
- separate ROI embedding blocks for `embeddings.sigmoid_curve` and `embeddings.beta_gamma`
- axis weights such as `reg_region`, `reg_brand`, `reg_channel`, and `reg_vehicle`

## Quick Start

From the repo root:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests
MPLCONFIGDIR=/private/tmp/watchtower_mpl_cache .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/toy_watchtower_mmm_walkthrough.ipynb --ExecutePreprocessor.timeout=900
```

The notebook writes its tiny HDF5 bundle under `data/generated/`.
