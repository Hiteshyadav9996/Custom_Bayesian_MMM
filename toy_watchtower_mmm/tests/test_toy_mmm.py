from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toy_mmm.artifacts import build_all_artifacts
from toy_mmm.config import ToyConfig
from toy_mmm.diagnostics import build_impact_convergence_df, build_stats_df, layer_to_df
from toy_mmm.hdf5_io import load_hdf5_bundle, write_hdf5_bundle
from toy_mmm.hyperparameters import (
    alpha_bounds_from_hyperparameters,
    embedding_axis_l2,
    load_hyperparameters,
    roi_bounds_from_hyperparameters,
)
from toy_mmm.normalization import normalize_raw_data, target_to_raw
from toy_mmm.pricing import PricingInput, SimplePricingLayer
from toy_mmm.simulate import simulate_raw_data
from toy_mmm.train import train_model


def test_hdf5_round_trip_and_denormalization(tmp_path):
    config = ToyConfig()
    hyperparameters = load_hyperparameters(ROOT / "config" / "toy_mmm_hyperparameters.yml")
    data = normalize_raw_data(simulate_raw_data(config, hyperparameters), config, hyperparameters)
    hdf5_path, metadata_path = write_hdf5_bundle(data, tmp_path)
    tensors, metadata = load_hdf5_bundle(hdf5_path, metadata_path)

    assert tensors["features"].shape == (50, 260, 3)
    assert tensors["vehicle_spends"].shape == (50, 260, 5)
    assert tensors["price"].shape == (50, 260)
    assert "roi" in metadata["hyperparameters"]
    np.testing.assert_allclose(target_to_raw(tensors["target"], metadata), data.raw.target_raw, rtol=1e-5, atol=1e-4)


def test_pricing_layer_average_price_is_neutral_and_decreasing():
    layer = SimplePricingLayer(n_regions=1, n_brands=1, n_channels=1)
    region = tf.constant([0], dtype=tf.int32)
    brand = tf.constant([0], dtype=tf.int32)
    channel = tf.constant([0], dtype=tf.int32)
    prices = tf.constant([[0.9, 1.0, 1.1]], dtype=tf.float32)
    result = layer.from_input(PricingInput(prices, region, brand, channel))
    values = result.impact.numpy()[0]
    assert np.isclose(values[1], 1.0, atol=1e-6)
    assert values[0] > values[1] > values[2]


def test_training_artifacts_and_diagnostics_smoke():
    config = ToyConfig()
    hyperparameters = load_hyperparameters(ROOT / "config" / "toy_mmm_hyperparameters.yml")
    data = normalize_raw_data(simulate_raw_data(config, hyperparameters), config, hyperparameters)
    sigmoid_axis_l2 = embedding_axis_l2(hyperparameters, "sigmoid_curve")
    beta_gamma_axis_l2 = embedding_axis_l2(hyperparameters, "beta_gamma")
    assert {"global", "region", "brand", "channel", "vehicle"}.issubset(sigmoid_axis_l2)
    assert {"global", "region", "brand", "channel", "vehicle"}.issubset(beta_gamma_axis_l2)

    result = train_model(data, epochs=60, snapshot_interval=30, hyperparameters=hyperparameters, verbose=False)
    final = result.checkpoints[-1]
    artifacts = build_all_artifacts(data, final, selected_series=0)

    assert np.isfinite(final["prediction"]).all()
    roi_bounds = roi_bounds_from_hyperparameters(hyperparameters)
    for name in ["asymptote", "slope", "beta", "gamma"]:
        lower, upper = roi_bounds[name]
        values = final["parts"][name]
        assert np.all(values >= lower - 1e-5)
        assert np.all(values <= upper + 1e-5)
    alpha = float(final["parts"]["alpha"])
    alpha_lower, alpha_upper = alpha_bounds_from_hyperparameters(hyperparameters)
    assert alpha_lower <= alpha <= alpha_upper

    y_df = artifacts["y_df"]
    impact_sum = artifacts["impact_df"].groupby(["series_id", "date"])["contribution"].sum().reset_index()
    merged = y_df.merge(impact_sum, on=["series_id", "date"])
    np.testing.assert_allclose(merged["predicted"], merged["contribution"], rtol=1e-4, atol=1e-3)

    layer_frames = layer_to_df(result.model, data.metadata)
    expected = {
        "feature_beta/distribution/region",
        "feature_beta/temperature/region",
        "feature_beta/macro/region",
        "pricing/global",
        "pricing/region",
        "sigmoid_curve/vehicle",
        "beta_gamma/vehicle",
        "roi_bounded/series_vehicle",
        "alpha/global",
    }
    assert expected.issubset(layer_frames)
    stats_df = build_stats_df(layer_frames)
    assert {"layer", "axis", "parameter", "overfitting_rel_avg"}.issubset(stats_df.columns)
    assert np.isfinite(stats_df["overfitting_rel_avg"]).all()

    epoch_dfs = {ckpt["epoch"]: artifacts["impact_df"] if ckpt is final else build_all_artifacts(data, ckpt)["impact_df"] for ckpt in result.checkpoints}
    conv = build_impact_convergence_df(epoch_dfs)
    final_conv = conv[conv["epoch"] == max(epoch_dfs)]
    assert np.allclose(final_conv["mse_to_final"], 0.0)
