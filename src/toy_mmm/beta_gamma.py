from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras

from toy_mmm.layers import SimpleHierEmbedding, beta_gamma_carryover_tf, bounded_tf


@dataclass
class BetaGammaIntermediaries:
    raw: tf.Tensor
    beta_raw: tf.Tensor
    gamma_raw: tf.Tensor
    beta: tf.Tensor
    gamma: tf.Tensor
    carryover_impact: tf.Tensor


class SimpleBetaGammaLayer(keras.layers.Layer):
    """Beta-gamma carryover with its own hierarchy embedding."""

    def __init__(
        self,
        n_regions: int,
        n_brands: int,
        n_channels: int,
        n_vehicles: int,
        bounds: dict[str, tuple[float, float]],
        decay_length: int,
        l2_strength: float = 0.0,
        axis_l2: dict[str, float] | None = None,
        global_initializer: np.ndarray | None = None,
        vehicle_initializer: np.ndarray | None = None,
        name: str | None = "beta_gamma",
    ):
        super().__init__(name=name)
        self.bounds = bounds
        self.decay_length = int(decay_length)
        self.params = SimpleHierEmbedding(
            2,
            n_regions=n_regions,
            n_brands=n_brands,
            n_channels=n_channels,
            n_vehicles=n_vehicles,
            l2_strength=l2_strength,
            axis_l2=axis_l2,
            global_initializer=global_initializer,
            vehicle_initializer=vehicle_initializer,
            name="beta_gamma_params",
        )

    def call(
        self,
        instant_impact: tf.Tensor,
        region_idx: tf.Tensor,
        brand_idx: tf.Tensor,
        channel_idx: tf.Tensor,
    ) -> BetaGammaIntermediaries:
        raw = self.params(region_idx, brand_idx, channel_idx)
        beta_raw = raw[:, :, 0]
        gamma_raw = raw[:, :, 1]
        beta = bounded_tf(beta_raw, *self.bounds["beta"])
        gamma = bounded_tf(gamma_raw, *self.bounds["gamma"])
        carryover_impact = beta_gamma_carryover_tf(instant_impact, beta, gamma, self.decay_length)
        return BetaGammaIntermediaries(
            raw=raw,
            beta_raw=beta_raw,
            gamma_raw=gamma_raw,
            beta=beta,
            gamma=gamma,
            carryover_impact=carryover_impact,
        )

    def deviation_l2(self) -> tf.Tensor:
        return self.params.deviation_l2()
