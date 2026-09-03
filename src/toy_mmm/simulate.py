from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from toy_mmm.config import ToyConfig, default_config
from toy_mmm.hyperparameters import resolve_hyperparameters, roi_bounds_from_hyperparameters
from toy_mmm.layers import (
    beta_gamma_carryover_np,
    bounded_np,
    centered,
    centered_normal,
    normalized_sigmoid_curve_np,
    raw_from_bounded_np,
)


@dataclass
class RawToyData:
    dates: pd.DatetimeIndex
    series: pd.DataFrame
    features_raw: np.ndarray
    price_raw: np.ndarray
    spends_raw: np.ndarray
    target_raw: np.ndarray
    train_week_mask: np.ndarray
    val_week_mask: np.ndarray
    true_params: dict[str, np.ndarray | float]
    true_parts: dict[str, np.ndarray]


def _series_frame(config: ToyConfig) -> pd.DataFrame:
    rows = []
    for region in config.regions:
        for brand in config.brands:
            for channel in config.channels:
                rows.append({"region": region, "brand": brand, "channel": channel})
    df = pd.DataFrame(rows)
    df["series_id"] = np.arange(len(df), dtype=np.int32)
    df["region_idx"] = df["region"].map({v: i for i, v in enumerate(config.regions)}).astype(np.int32)
    df["brand_idx"] = df["brand"].map({v: i for i, v in enumerate(config.brands)}).astype(np.int32)
    df["channel_idx"] = df["channel"].map({v: i for i, v in enumerate(config.channels)}).astype(np.int32)
    return df


def _true_hier(
    rng: np.random.Generator,
    global_values: np.ndarray,
    series: pd.DataFrame,
    n_regions: int,
    n_brands: int,
    n_channels: int,
    region_sd,
    brand_sd,
    channel_sd,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    global_values = np.asarray(global_values, dtype=np.float32)
    region = centered_normal(rng, (n_regions, len(global_values)), region_sd)
    brand = centered_normal(rng, (n_brands, len(global_values)), brand_sd)
    channel = centered_normal(rng, (n_channels, len(global_values)), channel_sd)
    values = (
        global_values[None, :]
        + region[series["region_idx"].to_numpy()]
        + brand[series["brand_idx"].to_numpy()]
        + channel[series["channel_idx"].to_numpy()]
    ).astype(np.float32)
    return values, {"global": global_values, "region": region, "brand": brand, "channel": channel}


def simulate_raw_data(config: ToyConfig | None = None, hyperparameters: dict | str | None = None) -> RawToyData:
    config = config or default_config()
    hparams = resolve_hyperparameters(hyperparameters)
    roi_bounds = roi_bounds_from_hyperparameters(hparams)
    rng = np.random.default_rng(config.seed)
    dates = pd.date_range("2019-01-07", periods=config.n_weeks, freq="W-MON")
    series = _series_frame(config)
    n_series = len(series)
    n_features = len(config.generic_features)
    n_vehicles = len(config.vehicles)
    week = np.arange(config.n_weeks)
    season = np.sin(2 * np.pi * week / 52.0)
    cos_season = np.cos(2 * np.pi * week / 52.0)
    trend = np.linspace(-1.0, 1.0, config.n_weeks)

    region_idx = series["region_idx"].to_numpy()
    brand_idx = series["brand_idx"].to_numpy()
    channel_idx = series["channel_idx"].to_numpy()
    region_size = np.array([1.15, 0.95, 1.04, 0.88, 1.08], dtype=np.float32)
    brand_size = np.array([1.28, 1.12, 0.98, 0.86, 0.76], dtype=np.float32)
    channel_size = np.array([0.88, 1.18], dtype=np.float32)
    series_scale = region_size[region_idx] * brand_size[brand_idx] * channel_size[channel_idx]
    channel_sign = np.where(channel_idx == 0, -1.0, 1.0).astype(np.float32)

    distribution = (
        0.38 * rng.normal(size=(n_series, config.n_weeks))
        + 0.26 * series_scale[:, None]
        + 0.14 * channel_sign[:, None]
        + 0.15 * season[None, :]
    )
    temperature = (
        season[None, :]
        + 0.16 * rng.normal(size=(n_series, config.n_weeks))
        + 0.06 * region_idx[:, None]
        + 0.05 * channel_sign[:, None]
    )
    macro = trend[None, :] + 0.08 * rng.normal(size=(n_series, config.n_weeks)) + 0.04 * region_idx[:, None]
    features_raw = np.stack([distribution, temperature, macro], axis=-1).astype(np.float32)

    base_price = 95.0 + 5.0 * brand_idx[:, None] + 3.0 * channel_idx[:, None] + 2.0 * region_idx[:, None]
    price_phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_series, 1))
    price_wave = (
        0.12 * np.sin(2.0 * np.pi * week[None, :] / 31.0 + price_phase)
        + 0.07 * np.sin(2.0 * np.pi * week[None, :] / 17.0 + price_phase / 3.0)
    )
    price_discount = np.zeros((n_series, config.n_weeks), dtype=np.float32)
    for s in range(n_series):
        starts = rng.choice(np.arange(12, config.n_weeks - 8), size=7, replace=False)
        for start in starts:
            length = int(rng.integers(2, 5))
            price_discount[s, start : start + length] -= rng.uniform(0.08, 0.18)
    price_multiplier_raw = np.clip(
        1.0 + price_wave + price_discount + rng.normal(0, 0.012, size=(n_series, config.n_weeks)),
        0.70,
        1.32,
    )
    price_raw = (base_price * price_multiplier_raw).astype(np.float32)
    price_raw = np.maximum(price_raw, 30.0)

    spends = np.zeros((n_series, config.n_weeks, n_vehicles), dtype=np.float32)
    spends[:, :, 0] = np.maximum(
        0,
        (18 + 20 * rng.random(size=(n_series, config.n_weeks)) + 4 * season[None, :])
        * series_scale[:, None]
        * (0.92 + 0.12 * channel_idx[:, None]),
    )
    spends[:, :, 1] = np.maximum(
        0,
        (12 + 18 * rng.random(size=(n_series, config.n_weeks)) + 3 * cos_season[None, :])
        * (0.8 + 0.3 * series_scale[:, None])
        * (1.10 - 0.10 * channel_idx[:, None]),
    )
    for s in range(n_series):
        starts = rng.choice(np.arange(8, config.n_weeks - 8), size=13, replace=False)
        for start in starts:
            spends[s, start : start + int(rng.integers(1, 4)), 2] += rng.uniform(42, 125) * series_scale[s]
    promo_probability = 0.070 + 0.030 * channel_idx[:, None]
    promo_mask = rng.random((n_series, config.n_weeks)) < promo_probability
    spends[:, :, 3] = promo_mask * rng.uniform(24, 92, size=(n_series, config.n_weeks)) * series_scale[:, None]

    no_investment_weeks = np.array(config.no_investment_weeks, dtype=np.int32)
    allowed_event_weeks = np.setdiff1d(np.arange(35, config.train_weeks - 20), no_investment_weeks)
    sponsorship_weeks = rng.choice(allowed_event_weeks, size=n_series, replace=True)
    for s, event_week in enumerate(sponsorship_weeks):
        spends[s, event_week, 4] = rng.uniform(220, 360) * series_scale[s]
    spends[:, no_investment_weeks, :] = 0.0

    train_week_mask = np.zeros(config.n_weeks, dtype=bool)
    train_week_mask[: config.train_weeks] = True
    val_week_mask = ~train_week_mask

    feature_mean = features_raw[:, train_week_mask].mean(axis=(0, 1), keepdims=True)
    feature_std = features_raw[:, train_week_mask].std(axis=(0, 1), keepdims=True)
    features_norm_for_truth = (features_raw - feature_mean) / np.maximum(feature_std, 1e-6)
    price_mean_by_series = price_raw[:, train_week_mask].mean(axis=1, keepdims=True)
    price_norm_for_truth = price_raw / price_mean_by_series

    baseline_values, baseline_parts = _true_hier(
        rng,
        np.array([config.baseline_prior], dtype=np.float32),
        series,
        len(config.regions),
        len(config.brands),
        len(config.channels),
        region_sd=5.0,
        brand_sd=4.5,
        channel_sd=4.0,
    )
    baseline_true = baseline_values[:, 0]

    feature_values, feature_parts = _true_hier(
        rng,
        np.array(config.feature_beta_prior, dtype=np.float32),
        series,
        len(config.regions),
        len(config.brands),
        len(config.channels),
        region_sd=np.array([0.007, 0.006, 0.006], dtype=np.float32),
        brand_sd=np.array([0.006, 0.005, 0.005], dtype=np.float32),
        channel_sd=np.array([0.005, 0.004, 0.004], dtype=np.float32),
    )
    generic_feature_mult_by_signal = np.exp(features_norm_for_truth * feature_values[:, None, :]).astype(np.float32)

    pricing_values, pricing_parts = _true_hier(
        rng,
        np.array([config.pricing_offset_prior, config.pricing_exponent_prior], dtype=np.float32),
        series,
        len(config.regions),
        len(config.brands),
        len(config.channels),
        region_sd=np.array([0.040, 0.16], dtype=np.float32),
        brand_sd=np.array([0.035, 0.14], dtype=np.float32),
        channel_sd=np.array([0.030, 0.12], dtype=np.float32),
    )
    pricing_offset_true = np.maximum(pricing_values[:, 0], 0.05)
    pricing_exponent_true = np.clip(pricing_values[:, 1], 0.7, 4.2)
    price_multiplier_true = ((1.0 + pricing_offset_true[:, None]) ** pricing_exponent_true[:, None]) / (
        (price_norm_for_truth + pricing_offset_true[:, None]) ** pricing_exponent_true[:, None]
    )
    multiplicative_impact_true = generic_feature_mult_by_signal.prod(axis=-1) * price_multiplier_true

    vehicle_values = {
        name: np.array(values, dtype=np.float32) for name, values in config.roi_vehicle_prior.items()
    }
    raw_center = np.array(
        [
            raw_from_bounded_np(np.mean(vehicle_values["asymptote"]), *roi_bounds["asymptote"]),
            raw_from_bounded_np(np.mean(vehicle_values["slope"]), *roi_bounds["slope"]),
            raw_from_bounded_np(np.mean(vehicle_values["beta"]), *roi_bounds["beta"]),
            raw_from_bounded_np(np.mean(vehicle_values["gamma"]), *roi_bounds["gamma"]),
        ],
        dtype=np.float32,
    )
    vehicle_raw = np.stack(
        [
            raw_from_bounded_np(vehicle_values["asymptote"], *roi_bounds["asymptote"]),
            raw_from_bounded_np(vehicle_values["slope"], *roi_bounds["slope"]),
            raw_from_bounded_np(vehicle_values["beta"], *roi_bounds["beta"]),
            raw_from_bounded_np(vehicle_values["gamma"], *roi_bounds["gamma"]),
        ],
        axis=-1,
    ).astype(np.float32)
    vehicle_raw = centered(vehicle_raw - raw_center[None, :], axis=0)
    roi_base, roi_parts = _true_hier(
        rng,
        raw_center,
        series,
        len(config.regions),
        len(config.brands),
        len(config.channels),
        region_sd=0.055,
        brand_sd=0.050,
        channel_sd=0.045,
    )
    roi_param_raw_true = roi_base[:, None, :] + vehicle_raw[None, :, :]
    asymptote_true = bounded_np(roi_param_raw_true[:, :, 0], *roi_bounds["asymptote"]).astype(np.float32)
    slope_true = bounded_np(roi_param_raw_true[:, :, 1], *roi_bounds["slope"]).astype(np.float32)
    beta_true = bounded_np(roi_param_raw_true[:, :, 2], *roi_bounds["beta"]).astype(np.float32)
    gamma_true = bounded_np(roi_param_raw_true[:, :, 3], *roi_bounds["gamma"]).astype(np.float32)

    spend_scale = np.array(
        [np.percentile(spends[:, :, v][spends[:, :, v] > 0], 80) for v in range(n_vehicles)], dtype=np.float32
    )
    instant = spend_scale[None, None, :] * normalized_sigmoid_curve_np(
        spends / spend_scale[None, None, :], asymptote_true[:, None, :], slope_true[:, None, :]
    )
    carryover = beta_gamma_carryover_np(instant.astype(np.float32), beta_true, gamma_true, config.decay_length)
    total_vehicle_impact = instant + carryover
    roi_impact_true = total_vehicle_impact.sum(axis=-1)
    alpha_true = config.alpha_prior
    noise = rng.normal(0, 10, size=(n_series, config.n_weeks)).astype(np.float32)
    target = baseline_true[:, None] * multiplicative_impact_true + (multiplicative_impact_true**alpha_true) * roi_impact_true
    target = (target + noise).astype(np.float32)

    true_params: dict[str, np.ndarray | float] = {
        "baseline": baseline_true.astype(np.float32),
        "feature_beta": feature_values.astype(np.float32),
        "pricing_offset": pricing_offset_true.astype(np.float32),
        "pricing_exponent": pricing_exponent_true.astype(np.float32),
        "asymptote": asymptote_true,
        "slope": slope_true,
        "beta": beta_true,
        "gamma": gamma_true,
        "alpha": float(alpha_true),
        "vehicle_raw_deviation": vehicle_raw,
    }
    true_parts = {
        "baseline_parts_global": baseline_parts["global"],
        "baseline_parts_region": baseline_parts["region"],
        "baseline_parts_brand": baseline_parts["brand"],
        "baseline_parts_channel": baseline_parts["channel"],
        "feature_parts_global": feature_parts["global"],
        "pricing_parts_global": pricing_parts["global"],
        "roi_parts_global": roi_parts["global"],
        "generic_feature_mult_by_signal": generic_feature_mult_by_signal.astype(np.float32),
        "price_multiplier": price_multiplier_true.astype(np.float32),
        "multiplicative_impact": multiplicative_impact_true.astype(np.float32),
        "instant_impact": instant.astype(np.float32),
        "carryover_impact": carryover.astype(np.float32),
        "total_vehicle_impact": total_vehicle_impact.astype(np.float32),
    }

    return RawToyData(
        dates=dates,
        series=series,
        features_raw=features_raw,
        price_raw=price_raw,
        spends_raw=spends,
        target_raw=target,
        train_week_mask=train_week_mask,
        val_week_mask=val_week_mask,
        true_params=true_params,
        true_parts=true_parts,
    )
