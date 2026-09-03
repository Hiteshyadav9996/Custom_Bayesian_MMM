# Custom Bayesian MMM (Toy Watchtower)

A standalone, teaching-oriented **Marketing Mix Model (MMM)** that mirrors the Watchtower-style multiplicative architecture. The model is implemented in **TensorFlow/Keras** with hierarchical parameter embeddings, sigmoid media response curves, beta–gamma adstock carryover, and a dedicated price-elasticity layer.

This repository is designed for readers with a **mathematics and statistics background** who want to understand *what* is being estimated, *why* the structure looks this way, and *how* to run or adapt the code on their own data.

---

## Table of contents

1. [What this model does](#what-this-model-does)
2. [Mathematical specification](#mathematical-specification)
3. [Repository layout](#repository-layout)
4. [Getting started with uv](#getting-started-with-uv)
5. [End-to-end workflow](#end-to-end-workflow)
6. [Configuring the model for your own data](#configuring-the-model-for-your-own-data)
7. [Integrating into your system](#integrating-into-your-system)
8. [Hyperparameter reference](#hyperparameter-reference)
9. [Diagnostics and artifacts](#diagnostics-and-artifacts)
10. [Limitations and extensions](#limitations-and-extensions)

---

## What this model does

Given weekly panel data for many **series** (each series is a unique combination of region × brand × channel), the model predicts a normalized sales/volume target from:

| Input type | Examples in the default toy setup | Role |
|---|---|---|
| **Generic features** | distribution, temperature, macro index | Multiplicative demand shifters |
| **Price** | average selling price | Dedicated elasticity curve (not a plain linear beta) |
| **Media vehicles** | search, social, TV, promo, sponsorship | Sigmoid ROI curves + carryover dynamics |

The core idea is **not** a standard additive regression:

```text
Y ≠ β₀ + β₁X₁ + β₂X₂ + …
```

Instead, the model separates **baseline level**, **multiplicative context**, and **media ROI**:

```text
Y = Baseline × MultiplicativeImpacts + MultiplicativeImpacts^α × ROIImpacts
```

This structure keeps interpretability for business users (baseline, price, distribution) while allowing media to interact with the demand environment through the exponent α.

---

## Mathematical specification

### Indexing and data tensors

Let:

- `s ∈ {1,…,S}` index **series** (region × brand × channel combinations)
- `t ∈ {1,…,T}` index **time** (weeks)
- `k ∈ {1,…,K}` index **generic features**
- `v ∈ {1,…,V}` index **media vehicles**

Observed inputs (after normalization — see [Normalization](#normalization)):

| Symbol | Shape | Meaning |
|---|---|---|
| `f_{s,t,k}` | `(S, T, K)` | Generic feature k for series s at week t |
| `p_{s,t}` | `(S, T)` | Price (normalized so series-average price ≈ 1 on training data) |
| `x_{s,t,v}` | `(S, T, V)` | Media spend for vehicle v |
| `y_{s,t}` | `(S, T)` | Target (normalized volume/sales) |

Each series carries categorical indices `(r(s), b(s), c(s))` for region, brand, and channel.

---

### Top-level prediction equation

The model prediction is:

$$
\hat{y}_{s,t} = B_s \cdot M_{s,t} + M_{s,t}^{\alpha} \cdot R_{s,t}
$$

where:

- **`B_s`** — baseline level for series s (strictly positive after offset initialization)
- **`M_{s,t}`** — multiplicative impact from generic features and price
- **`R_{s,t}`** — total media ROI impact (instant + carryover, summed over vehicles)
- **`α`** — global interaction exponent, constrained to a configurable interval (default ≈ [0.4, 1.2])

**Intuition:**

- When `M_{s,t} = 1`, the model reduces to `B_s + R_{s,t}` (baseline plus media).
- When demand conditions are strong (`M_{s,t} > 1`), media effects are amplified by `M_{s,t}^α`.
- α < 1 dampens interaction; α → 1 recovers a more linear additivity between the multiplicative environment and media.

---

### Hierarchical embeddings (partial pooling)

Most learned parameters follow an **additive hierarchical decomposition** with mean-centered deviations for identifiability:

$$
\theta_s = \theta_{\text{global}} + \delta_{r(s)} + \delta_{b(s)} + \delta_{c(s)}
$$

For media-curve parameters, an additional **vehicle deviation** is included:

$$
\theta_{s,v} = \theta_{\text{global}} + \delta_{r(s)} + \delta_{b(s)} + \delta_{c(s)} + \delta_v
$$

Each deviation axis is **mean-centered** (e.g. region deviations sum to zero across regions), which prevents confounding between global and group effects.

**Statistical interpretation:** This is a form of **partial pooling** / **mixed-effects-style shrinkage**. When L2 penalties on deviations are enabled, the model borrows strength across regions, brands, and channels — similar in spirit to hierarchical Bayesian MMM, but here implemented as penalized maximum likelihood in TensorFlow.

Regularization penalty (when enabled):

$$
\mathcal{L}_{\text{reg}} = \sum_{\text{layers}} \sum_{\text{axes}} \lambda_{\text{layer}} \cdot w_{\text{axis}} \cdot \|\delta\|_2^2
$$

Effective axis weight: `effective_lambda(axis) = layer_reg_lambda × reg_<axis>` (see `config/toy_mmm_hyperparameters.yml`).

---

### Multiplicative generic features

Each feature k has its own hierarchical beta `β_{s,k}`.

Per-feature log-contribution (clipped for numerical stability):

$$
\ell_{s,t,k} = \text{clip}(f_{s,t,k} \cdot \beta_{s,k},\,-1,\,1)
$$

Multiplicative factor:

$$
\exp(\ell_{s,t,k})
$$

Combined generic-feature multiplier:

$$
G_{s,t} = \prod_{k=1}^{K} \exp(\ell_{s,t,k})
$$

**Why multiplicative?** A 10% improvement in distribution and a 10% improvement in a macro index combine as `(1.1 × 1.1)` rather than `1.1 + 1.1`, which is usually more realistic for volume drivers.

---

### Price elasticity layer

Price is **not** modeled as a linear term. Instead, each series learns a **constant-elasticity-style** multiplier with hierarchical `(offset, exponent)` parameters.

Learned parameters (after transforms):

$$
\text{offset}_s = 0.01 + \text{softplus}(\text{offset}_s^{\text{raw}}) > 0
$$

$$
\text{exponent}_s = 0.5 + 4.5 \cdot \sigma(\text{exponent}_s^{\text{raw}}) \in (0.5,\,5.0)
$$

Price multiplier:

$$
P_{s,t} = \frac{(1 + \text{offset}_s)^{\text{exponent}_s}}{(p_{s,t} + \text{offset}_s)^{\text{exponent}_s}}
$$

**Key property:** After normalization, `p_{s,t} = 1` corresponds to the series' average training price, and **`P_{s,t} = 1` at that point** — i.e. average price is a **neutral** reference (verified in tests).

Higher price → lower multiplier (holding exponent fixed), consistent with downward-sloping demand.

Total multiplicative impact:

$$
M_{s,t} = G_{s,t} \cdot P_{s,t}
$$

---

### Media response: sigmoid curve (instant impact)

For each vehicle v, spend is scaled by a vehicle-specific axis scale `a_v` (80th percentile of positive spend on training data):

$$
\tilde{x}_{s,t,v} = \frac{x_{s,t,v}}{a_v}
$$

Each series–vehicle pair learns sigmoid parameters `(A_{s,v}, S_{s,v})` (asymptote and slope), constrained to bounds from the hyperparameter file.

Instantaneous media impact uses a **normalized sigmoid** `σ̃(·)` that passes through zero at zero spend and saturates at asymptote `A_{s,v}`:

$$
I_{s,t,v} = \underbrace{\frac{N_{\text{vol}}}{N_{\text{spend}}}}_{\text{roi\_unit\_scale}} \cdot a_v \cdot \tilde{\sigma}(\tilde{x}_{s,t,v};\, A_{s,v},\, S_{s,v})
$$

where `roi_unit_scale = normalization_factor_spend / normalization_factor` converts normalized spend units back to target units.

**Interpretation:**

- **`A_{s,v}`** — long-run saturation level (maximum incremental impact per week at high spend)
- **`S_{s,v}`** — steepness of the curve (how quickly impact rises with spend)

This is analogous to a **Hill/saturation function** commonly used in MMM, but with a fixed functional form chosen for stability.

---

### Media carryover: beta–gamma adstock

Instant impact is propagated forward with a **geometric decay** structure parameterized by `(β_{s,v}, γ_{s,v})`:

$$
C_{s,t,v} = \sum_{\ell=1}^{L} I_{s,t-\ell,v} \cdot \beta_{s,v} \cdot \gamma_{s,v}^{\,\ell-1}
$$

where `L = decay_length` (default 16 weeks).

Total vehicle impact:

$$
T_{s,t,v} = I_{s,t,v} + C_{s,t,v}
$$

Total ROI impact (summed across vehicles):

$$
R_{s,t} = \sum_{v=1}^{V} T_{s,t,v}
$$

**Parameter roles:**

| Parameter | Typical interpretation |
|---|---|
| `β_{s,v}` | Carryover intensity (fraction of instant effect that persists) |
| `γ_{s,v}` | Decay rate ( closer to 1 → longer memory ) |
| `L` | Maximum lag window |

This is related to **adstock** and **distributed lag** models in econometrics, and to beta-binomial / Weibull adstock variants used in industry MMM.

---

### Full decomposition

Putting it all together:

$$
\hat{y}_{s,t} = \underbrace{B_s}_{\text{baseline}} \cdot \underbrace{M_{s,t}}_{\text{features + price}} + \underbrace{M_{s,t}^{\alpha}}_{\text{interaction}} \cdot \underbrace{\sum_v (I_{s,t,v} + C_{s,t,v})}_{\text{media ROI}}
$$

The training objective is **masked MSE** on training weeks plus optional regularization:

$$
\mathcal{L} = \frac{1}{|\Omega_{\text{train}}|} \sum_{(s,t) \in \Omega_{\text{train}}} (y_{s,t} - \hat{y}_{s,t})^2 + \mathcal{L}_{\text{reg}}
$$

Validation weeks are held out via `train_week_mask` / `val_week_mask`.

---

### Normalization

All inputs are scaled using **training-set statistics only** (no leakage):

| Quantity | Normalization |
|---|---|
| Target `y` | Divide by median training target |
| Spends `x` | Divide by 80th percentile of positive spends (global) |
| Features `f` | Z-score using training mean/std (pooled across series and time) |
| Price `p` | Divide by series-specific mean price on training weeks |
| Vehicle axis scale `a_v` | 80th percentile of positive spend per vehicle (then divided by global spend factor) |

Metadata stores all factors in `metadata.json` so predictions and artifacts can be mapped back to raw units.

---

## Repository layout

```text
.
├── README.md
├── pyproject.toml              # Project metadata and dependencies (managed by uv)
├── config/
│   └── toy_mmm_hyperparameters.yml
├── data/
│   └── generated/              # Example HDF5 bundle from the walkthrough
├── notebooks/
│   └── toy_watchtower_mmm_walkthrough.ipynb
├── src/
│   └── toy_mmm/
│       ├── model.py            # Top-level ToyMMMModel
│       ├── layers.py           # Hierarchical embeddings, sigmoid, carryover math
│       ├── pricing.py          # Price elasticity layer
│       ├── sigmoid_curve.py    # Media saturation curves
│       ├── beta_gamma.py       # Adstock carryover
│       ├── normalization.py    # Train-only scaling
│       ├── simulate.py         # Synthetic data generator (for teaching)
│       ├── train.py            # Training loop
│       ├── hdf5_io.py          # Data persistence
│       ├── artifacts.py        # Contribution decomposition dataframes
│       ├── diagnostics.py      # Convergence and hierarchy diagnostics
│       └── config.py           # ToyConfig defaults
└── tests/
    └── test_toy_mmm.py
```

---

## Getting started with uv

This project uses **[uv](https://docs.astral.sh/uv/)** for environment and dependency management. Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clone and enter the repository

```bash
git clone git@github.com:Hiteshyadav9996/Custom_Bayesian_MMM.git
cd Custom_Bayesian_MMM
```

### 2. Create the virtual environment and install dependencies

```bash
uv sync --all-groups
```

This reads `pyproject.toml`, creates `.venv/`, and installs runtime + dev dependencies (including `pytest` and `jupyter`).

### 3. Run the test suite

```bash
uv run pytest
```

Expected: all tests pass (HDF5 round-trip, pricing neutrality, short training smoke test).

### 4. Run the walkthrough notebook

```bash
uv run jupyter lab notebooks/toy_watchtower_mmm_walkthrough.ipynb
```

Or execute headlessly:

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/toy_watchtower_mmm_walkthrough.ipynb \
  --ExecutePreprocessor.timeout=900
```

The notebook simulates data, normalizes, writes `data/generated/toy_mmm_data.h5`, trains the model, and produces diagnostic plots.

### 5. Quick Python smoke test

```bash
uv run python -c "
from toy_mmm import default_config, simulate_raw_data, normalize_raw_data, train_model
from toy_mmm.hyperparameters import load_hyperparameters

config = default_config()
hparams = load_hyperparameters('config/toy_mmm_hyperparameters.yml')
data = normalize_raw_data(simulate_raw_data(config, hparams), config, hparams)
result = train_model(data, epochs=10, hyperparameters=hparams, verbose=True)
print('Final train MSE:', result.history.iloc[-1]['train_mse'])
"
```

---

## End-to-end workflow

```text
Raw panel arrays (or simulate_raw_data)
        │
        ▼
normalize_raw_data(config, hyperparameters)
        │
        ├──► metadata.json  (encodings, norm factors, priors)
        └──► arrays: features, price, vehicle_spends, target, masks
        │
        ▼
write_hdf5_bundle(data, output_dir)   [optional persistence]
        │
        ▼
train_model(data, hyperparameters=...)
        │
        ├──► Trained ToyMMMModel
        ├──► checkpoints (predictions + learned params over epochs)
        └──► history DataFrame (train/val MSE, regularization)
        │
        ▼
build_all_artifacts(data, checkpoint)  [contributions, ROI curves, etc.]
```

Minimal training script:

```python
from pathlib import Path

from toy_mmm import default_config, normalize_raw_data, simulate_raw_data, train_model
from toy_mmm.hyperparameters import load_hyperparameters

config = default_config()
hparams = load_hyperparameters("config/toy_mmm_hyperparameters.yml")

raw = simulate_raw_data(config, hparams)
data = normalize_raw_data(raw, config, hparams)

result = train_model(
    data,
    hyperparameters=hparams,
    epochs=hparams["training"]["epochs"],
    verbose=True,
)

final = result.checkpoints[-1]
print("Alpha:", float(final["parts"]["alpha"]))
print("Val MSE:", result.history.iloc[-1]["val_mse"])
```

Run with:

```bash
uv run python train_example.py
```

---

## Configuring the model for your own data

You do **not** need the simulator for real data. The integration point is `normalize_raw_data`, which expects a `RawToyData` container.

### Step 1 — Define your hierarchy in `ToyConfig`

Edit `src/toy_mmm/config.py` or construct a custom instance:

```python
from toy_mmm.config import ToyConfig

config = ToyConfig(
    seed=42,
    n_weeks=156,                    # total weeks in your panel
    train_weeks=130,                # weeks used for training (rest = validation)
    decay_length=13,                # max adstock lag in weeks
    regions=("us_northeast", "us_west"),
    brands=("brand_x", "brand_y"),
    channels=("retail", "ecom"),
    generic_features=("distribution", "competitor_index", "seasonality_index"),
    vehicles=("tv", "search", "social", "promo"),
    # Priors initialize the hierarchical embeddings (rough expected values)
    baseline_prior=100.0,
    feature_beta_prior=(0.05, 0.03, 0.02),
    pricing_offset_prior=0.6,
    pricing_exponent_prior=2.0,
    alpha_prior=0.8,
    roi_vehicle_prior={
        "asymptote": (1.0, 0.9, 1.1, 0.8),   # one per vehicle
        "slope":     (1.2, 1.5, 1.0, 1.8),
        "beta":      (0.2, 0.3, 0.25, 0.15),
        "gamma":     (0.5, 0.6, 0.55, 0.4),
    },
)
```

### Step 2 — Build `RawToyData` from your arrays

Your raw arrays must match these shapes:

| Field | Shape | Description |
|---|---|---|
| `features_raw` | `(S, T, K)` | Unscaled generic features |
| `price_raw` | `(S, T)` | Price in natural units |
| `spends_raw` | `(S, T, V)` | Media spend (0 allowed) |
| `target_raw` | `(S, T)` | Volume/sales/revenue |
| `train_week_mask` | `(T,)` bool | True for training weeks |
| `val_week_mask` | `(T,)` bool | True for validation weeks |

```python
import numpy as np
import pandas as pd
from toy_mmm.simulate import RawToyData

# Example: build series index table
rows = []
for region in config.regions:
    for brand in config.brands:
        for channel in config.channels:
            rows.append({"region": region, "brand": brand, "channel": channel})
series = pd.DataFrame(rows)
series["series_id"] = np.arange(len(series), dtype=np.int32)
series["region_idx"] = series["region"].map({v: i for i, v in enumerate(config.regions)})
series["brand_idx"] = series["brand"].map({v: i for i, v in enumerate(config.brands)})
series["channel_idx"] = series["channel"].map({v: i for i, v in enumerate(config.channels)})

raw = RawToyData(
    dates=pd.DatetimeIndex(your_weekly_dates, freq="W-MON"),
    series=series,
    features_raw=your_features_array,      # float32, shape (S, T, K)
    price_raw=your_price_array,            # float32, shape (S, T)
    spends_raw=your_spend_array,           # float32, shape (S, T, V)
    target_raw=your_target_array,          # float32, shape (S, T)
    train_week_mask=train_mask,            # bool, shape (T,)
    val_week_mask=val_mask,                # bool, shape (T,)
    true_params={},                        # leave empty for real data
    true_parts={},                         # leave empty for real data
)
```

### Step 3 — Normalize, persist, and train

```python
from pathlib import Path

from toy_mmm import normalize_raw_data, train_model
from toy_mmm.hdf5_io import write_hdf5_bundle
from toy_mmm.hyperparameters import load_hyperparameters

hparams = load_hyperparameters("config/toy_mmm_hyperparameters.yml")
data = normalize_raw_data(raw, config, hparams)

# Save for reproducibility
write_hdf5_bundle(data, Path("data/my_run"))

result = train_model(data, hyperparameters=hparams)
```

### Step 4 — Tune hyperparameters

Copy and edit `config/toy_mmm_hyperparameters.yml`:

| Section | What to adjust |
|---|---|
| `training.learning_rate` | Adam step size (default 0.012) |
| `training.epochs` | Training length (default 1024) |
| `training.fallback_l2_strength` | Base L2 when layer lambda is 0 |
| `embeddings.*.layer_reg_lambda` | Enable partial pooling (try `0.00005`–`0.001`) |
| `embeddings.*.reg_region/brand/channel/vehicle` | Relative shrinkage per axis |
| `roi.sigmoid_curve.*` | Bounds on saturation (`asymptote`) and steepness (`slope`) |
| `roi.beta_gamma.*` | Bounds on carryover (`beta`, `gamma`) |
| `roi.alpha_mult.*` | Allowed range for interaction exponent α |

**Tip:** Start with `layer_reg_lambda: 0.0` (no shrinkage) to verify the pipeline, then increase regularization if hierarchy parameters look overfit.

---

## Integrating into your system

### Option A — Import as a Python library (recommended)

Install in editable mode via uv:

```bash
uv sync --all-groups
```

The `src/` layout is on the Python path via `[tool.pytest.ini_options] pythonpath = ["src"]`. When using `uv run`, imports work automatically:

```python
from toy_mmm import train_model, normalize_raw_data, load_hdf5_bundle
from toy_mmm.model import ToyMMMModel
from toy_mmm.artifacts import build_all_artifacts
```

**Typical production wrapper:**

1. **ETL** → build `RawToyData` from your warehouse (BigQuery, Snowflake, etc.)
2. **Normalize** → `normalize_raw_data` (store `metadata.json` alongside model artifacts)
3. **Train** → `train_model` on a schedule or on demand
4. **Serve** → load checkpoint weights into `ToyMMMModel`, call `model(tensors, return_parts=True)` for decomposition
5. **Report** → `build_all_artifacts` for contribution tables compatible with Plotly dashboards

### Option B — HDF5 bundle as the contract

The HDF5 + JSON metadata bundle is the **stable IO contract** between data engineering and modeling:

```python
from toy_mmm.hdf5_io import load_hdf5_bundle, write_hdf5_bundle

# Write (modeling side)
write_hdf5_bundle(normalized_data, "s3://bucket/mmm/run_id/")

# Read (training or scoring side)
tensors, metadata = load_hdf5_bundle("toy_mmm_data.h5", "metadata.json")
```

HDF5 datasets: `features`, `price`, `vehicle_spends`, `target`, `investment_axis_scale`, `region_idx`, `brand_idx`, `channel_idx`, `train_week_mask`, `val_week_mask`.

### Option C — Embed individual layers

Layers are modular and can be reused independently:

| Module | Use case |
|---|---|
| `SimplePricingLayer` | Standalone price elasticity curve |
| `SimpleSigmoidCurveLayer` | Saturation curves for a single media channel |
| `SimpleBetaGammaLayer` | Adstock on any instant impact series |
| `SimpleHierEmbedding` | Generic partial-pooling embeddings |

Example — pricing layer only:

```python
import tensorflow as tf
from toy_mmm.pricing import PricingInput, SimplePricingLayer

layer = SimplePricingLayer(n_regions=5, n_brands=3, n_channels=2)
result = layer.from_input(PricingInput(
    prices=tf.constant([[0.9, 1.0, 1.1]]),
    region_idx=tf.constant([0]),
    brand_idx=tf.constant([0]),
    channel_idx=tf.constant([0]),
))
print(result.impact.numpy())  # >1 at low price, 1 at average, <1 at high price
```

### Option D — Export trained parameters

Each training checkpoint contains:

```python
checkpoint = result.checkpoints[-1]
prediction = checkpoint["prediction"]          # normalized units
parts = checkpoint["parts"]                    # all intermediate tensors
weights = checkpoint["weights"]                # raw Keras variable values
deviation_norms = checkpoint["deviation_norms"]  # hierarchy deviation magnitudes
```

Use `diagnostics.layer_to_df(model, metadata)` to flatten hierarchy parameters into tidy DataFrames for BI tools.

---

## Hyperparameter reference

See `config/toy_mmm_hyperparameters.yml` for the full file. Key structures:

```yaml
training:
  optimizer: adam
  learning_rate: 0.012
  epochs: 1024
  snapshot_interval: 50

roi:
  sigmoid_curve:
    asymptote_lower_bound: 0.4
    asymptote_upper_bound: 2.4
    initial_slope_lower_bound: 0.4
    initial_slope_upper_bound: 1.9
  beta_gamma:
    beta_min: 0.01
    beta_max: 1.0
    gamma_min: 0.01
    gamma_max: 0.99
  alpha_mult:
    center: 0.8
    alpha_mult_range: 0.8   # α ∈ [center - range/2, center + range/2]

embeddings:
  sigmoid_curve:
    layer_reg_lambda: 0.0    # increase to shrink hierarchy deviations
    reg_region: 1.0
    reg_brand: 1.0
    reg_channel: 1.0
    reg_vehicle: 1.0
```

Load in code:

```python
from toy_mmm.hyperparameters import load_hyperparameters

hparams = load_hyperparameters("config/toy_mmm_hyperparameters.yml")
```

---

## Diagnostics and artifacts

After training:

```python
from toy_mmm.artifacts import build_all_artifacts
from toy_mmm.diagnostics import build_impact_convergence_df, layer_to_df

artifacts = build_all_artifacts(data, result.checkpoints[-1], selected_series=0)
# artifacts["y_df"]       — observed vs predicted (raw units)
# artifacts["impact_df"]  — contribution decomposition by layer/signal
# artifacts["roi_df"]     — ROI curves by vehicle

layer_frames = layer_to_df(result.model, data.metadata)
# Hierarchy parameters by region/brand/channel/vehicle

conv = build_impact_convergence_df({ckpt["epoch"]: build_all_artifacts(data, ckpt)["impact_df"]
                                    for ckpt in result.checkpoints})
```

The walkthrough notebook demonstrates Plotly charts for decomposition, convergence, pricing curves, and ROI saturation.

---

## Limitations and extensions

This is a **teaching implementation**, not a full production MMM platform:

| Topic | Status in this repo |
|---|---|
| Bayesian inference (MCMC, VI) | Not implemented — point estimation via Adam |
| Uncertainty quantification | Not implemented |
| Multi-stage calibration | Not implemented (single training stage) |
| Automated feature selection | Not implemented |
| Custom priors / constraints | Priors via initialization only |

**Natural extensions** for practitioners:

- Wrap hierarchy penalties in a fully Bayesian framework (PyMC, NumPyro)
- Add holdout-based early stopping or learning-rate schedules
- Replace sigmoid curves with Hill functions or splines
- Add cross-series correlation or dynamic baselines
- Export to ONNX / TF Serving for production scoring

---

## Quick reference commands (uv)

```bash
# Install everything
uv sync --all-groups

# Run tests
uv run pytest

# Run tests with output
uv run pytest -v

# Open notebook
uv run jupyter lab notebooks/toy_watchtower_mmm_walkthrough.ipynb

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --group dev <package-name>
```

---

## License

See repository settings on GitHub for license information.
