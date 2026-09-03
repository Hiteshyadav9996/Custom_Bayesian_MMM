from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from toy_mmm.config import ToyConfig, default_config
from toy_mmm.hyperparameters import resolve_hyperparameters
from toy_mmm.simulate import RawToyData


@dataclass
class NormalizedToyData:
    features: np.ndarray
    price: np.ndarray
    vehicle_spends: np.ndarray
    target: np.ndarray
    investment_axis_scale: np.ndarray
    region_idx: np.ndarray
    brand_idx: np.ndarray
    channel_idx: np.ndarray
    train_week_mask: np.ndarray
    val_week_mask: np.ndarray
    metadata: dict[str, Any]
    raw: RawToyData


def normalize_raw_data(
    raw: RawToyData,
    config: ToyConfig | None = None,
    hyperparameters: dict[str, Any] | str | None = None,
) -> NormalizedToyData:
    config = config or default_config()
    hparams = resolve_hyperparameters(hyperparameters)
    train_mask = raw.train_week_mask
    target_factor = float(np.percentile(raw.target_raw[:, train_mask], 50))
    spend_positive = raw.spends_raw[raw.spends_raw > 0]
    spend_factor = float(np.percentile(spend_positive, 80))
    price_mean_by_series = raw.price_raw[:, train_mask].mean(axis=1).astype(np.float32)
    feature_mean = raw.features_raw[:, train_mask].mean(axis=(0, 1)).astype(np.float32)
    feature_std = raw.features_raw[:, train_mask].std(axis=(0, 1)).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-6)

    vehicle_axis_scale_raw = np.array(
        [
            np.percentile(raw.spends_raw[:, :, v][raw.spends_raw[:, :, v] > 0], 80)
            for v in range(len(config.vehicles))
        ],
        dtype=np.float32,
    )
    features = ((raw.features_raw - feature_mean[None, None, :]) / feature_std[None, None, :]).astype(np.float32)
    price = (raw.price_raw / price_mean_by_series[:, None]).astype(np.float32)
    vehicle_spends = (raw.spends_raw / spend_factor).astype(np.float32)
    target = (raw.target_raw / target_factor).astype(np.float32)
    investment_axis_scale = (vehicle_axis_scale_raw / spend_factor).astype(np.float32)

    metadata: dict[str, Any] = {
        "encodings": {
            "region": {v: i for i, v in enumerate(config.regions)},
            "brand": {v: i for i, v in enumerate(config.brands)},
            "channel": {v: i for i, v in enumerate(config.channels)},
            "generic_feature": {v: i for i, v in enumerate(config.generic_features)},
            "vehicle": {v: i for i, v in enumerate(config.vehicles)},
        },
        "dates": [str(d.date()) for d in raw.dates],
        "series": raw.series.to_dict(orient="records"),
        "normalizations": {
            "norm_factors": {
                "normalization_factor": target_factor,
                "normalization_factor_volume": target_factor,
                "normalization_factor_nr": target_factor,
                "normalization_factor_spend": spend_factor,
                "normalization_factor_price": float(price_mean_by_series.mean()),
            },
            "feature_stats": {
                "mean": feature_mean.tolist(),
                "std": feature_std.tolist(),
                "names": list(config.generic_features),
            },
            "price_mean_by_series": price_mean_by_series.tolist(),
            "vehicle_axis_scale_raw": vehicle_axis_scale_raw.tolist(),
        },
        "config": {
            "seed": config.seed,
            "n_weeks": config.n_weeks,
            "decay_length": config.decay_length,
            "train_weeks": config.train_weeks,
            "training_epochs": config.training_epochs,
            "snapshot_interval": config.snapshot_interval,
            "pricing_exponent_bounds": config.pricing_exponent_bounds,
            "baseline_prior": config.baseline_prior,
            "feature_beta_prior": list(config.feature_beta_prior),
            "pricing_offset_prior": config.pricing_offset_prior,
            "pricing_exponent_prior": config.pricing_exponent_prior,
            "alpha_prior": config.alpha_prior,
            "roi_vehicle_prior": {key: list(value) for key, value in config.roi_vehicle_prior.items()},
        },
        "hyperparameters": hparams,
        "true_params": {
            key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in raw.true_params.items()
        },
    }
    return NormalizedToyData(
        features=features,
        price=price,
        vehicle_spends=vehicle_spends,
        target=target,
        investment_axis_scale=investment_axis_scale,
        region_idx=raw.series["region_idx"].to_numpy(dtype=np.int32),
        brand_idx=raw.series["brand_idx"].to_numpy(dtype=np.int32),
        channel_idx=raw.series["channel_idx"].to_numpy(dtype=np.int32),
        train_week_mask=raw.train_week_mask,
        val_week_mask=raw.val_week_mask,
        metadata=metadata,
        raw=raw,
    )


def target_to_raw(values: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    return values * metadata["normalizations"]["norm_factors"]["normalization_factor"]


def spend_to_raw(values: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    return values * metadata["normalizations"]["norm_factors"]["normalization_factor_spend"]
