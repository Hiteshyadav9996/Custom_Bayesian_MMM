from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "training": {
        "optimizer": "adam",
        "learning_rate": 0.012,
        "epochs": 1024,
        "snapshot_interval": 50,
        "history_interval": 5,
        "fallback_l2_strength": 0.0,
    },
    "roi": {
        "sigmoid_curve": {
            "asymptote_lower_bound": 0.4,
            "asymptote_upper_bound": 2.4,
            "initial_slope_lower_bound": 0.4,
            "initial_slope_upper_bound": 1.9,
        },
        "beta_gamma": {
            "beta_min": 0.01,
            "beta_max": 1.0,
            "gamma_min": 0.01,
            "gamma_max": 0.99,
        },
        "alpha_mult": {
            "center": 0.8,
            "alpha_mult_range": 0.8,
        },
    },
    "embeddings": {
        "baseline": {
            "regularization_strategy": "l2",
            "layer_reg_lambda": 0.0,
            "reg_bias": 0.0,
            "reg_region": 1.0,
            "reg_brand": 1.0,
            "reg_channel": 1.0,
        },
        "feature_beta": {
            "distribution": {
                "regularization_strategy": "l2",
                "layer_reg_lambda": 0.0,
                "reg_bias": 0.0,
                "reg_region": 1.0,
                "reg_brand": 1.0,
                "reg_channel": 1.0,
            },
            "temperature": {
                "regularization_strategy": "l2",
                "layer_reg_lambda": 0.0,
                "reg_bias": 0.0,
                "reg_region": 1.0,
                "reg_brand": 1.0,
                "reg_channel": 1.0,
            },
            "macro": {
                "regularization_strategy": "l2",
                "layer_reg_lambda": 0.0,
                "reg_bias": 0.0,
                "reg_region": 1.0,
                "reg_brand": 1.0,
                "reg_channel": 1.0,
            },
        },
        "pricing": {
            "regularization_strategy": "l2",
            "layer_reg_lambda": 0.0,
            "reg_bias": 0.0,
            "reg_region": 1.0,
            "reg_brand": 1.0,
            "reg_channel": 1.0,
        },
        "sigmoid_curve": {
            "regularization_strategy": "l2",
            "layer_reg_lambda": 0.0,
            "reg_bias": 0.0,
            "reg_region": 1.0,
            "reg_brand": 1.0,
            "reg_channel": 1.0,
            "reg_vehicle": 1.0,
        },
        "beta_gamma": {
            "regularization_strategy": "l2",
            "layer_reg_lambda": 0.0,
            "reg_bias": 0.0,
            "reg_region": 1.0,
            "reg_brand": 1.0,
            "reg_channel": 1.0,
            "reg_vehicle": 1.0,
        },
    },
}


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_hyperparameters(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_HYPERPARAMETERS)

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return deep_update(DEFAULT_HYPERPARAMETERS, loaded)


def resolve_hyperparameters(hyperparameters: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if hyperparameters is None:
        return load_hyperparameters()
    if isinstance(hyperparameters, (str, Path)):
        return load_hyperparameters(hyperparameters)
    return deep_update(DEFAULT_HYPERPARAMETERS, hyperparameters)


def get_embedding_block(hyperparameters: dict[str, Any], embedding_name: str) -> dict[str, Any]:
    block: dict[str, Any] = hyperparameters.get("embeddings", {})
    for piece in embedding_name.split("."):
        value = block.get(piece, {})
        if not isinstance(value, dict):
            return {}
        block = value
    return block


def embedding_axis_l2(
    hyperparameters: dict[str, Any] | None,
    embedding_name: str,
    fallback_l2_strength: float = 0.0,
) -> dict[str, float]:
    if not hyperparameters:
        return {
            "global": 0.0,
            "region": fallback_l2_strength,
            "brand": fallback_l2_strength,
            "channel": fallback_l2_strength,
            "vehicle": fallback_l2_strength,
        }

    block = get_embedding_block(hyperparameters, embedding_name)
    if block.get("regularization_strategy", "l2") != "l2":
        raise ValueError(f"Only l2 regularization is implemented in the toy repo: {embedding_name}")

    layer_lambda = float(block.get("layer_reg_lambda", fallback_l2_strength))
    return {
        "global": layer_lambda * float(block.get("reg_bias", 0.0)),
        "region": layer_lambda * float(block.get("reg_region", 1.0)),
        "brand": layer_lambda * float(block.get("reg_brand", 1.0)),
        "channel": layer_lambda * float(block.get("reg_channel", 1.0)),
        "vehicle": layer_lambda * float(block.get("reg_vehicle", 1.0)),
    }


def roi_bounds_from_hyperparameters(hyperparameters: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
    hparams = resolve_hyperparameters(hyperparameters)
    sigmoid = hparams["roi"]["sigmoid_curve"]
    beta_gamma = hparams["roi"]["beta_gamma"]
    return {
        "asymptote": (float(sigmoid["asymptote_lower_bound"]), float(sigmoid["asymptote_upper_bound"])),
        "slope": (float(sigmoid["initial_slope_lower_bound"]), float(sigmoid["initial_slope_upper_bound"])),
        "beta": (float(beta_gamma["beta_min"]), float(beta_gamma["beta_max"])),
        "gamma": (float(beta_gamma["gamma_min"]), float(beta_gamma["gamma_max"])),
    }


def alpha_bounds_from_hyperparameters(hyperparameters: dict[str, Any] | None) -> tuple[float, float]:
    hparams = resolve_hyperparameters(hyperparameters)
    alpha_mult = hparams["roi"]["alpha_mult"]
    center = float(alpha_mult["center"])
    alpha_range = float(alpha_mult["alpha_mult_range"])
    return center - alpha_range / 2.0, center + alpha_range / 2.0
