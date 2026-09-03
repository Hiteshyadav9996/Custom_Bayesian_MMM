from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras

from toy_mmm.layers import SimpleHierEmbedding


@dataclass
class PricingInput:
    prices: tf.Tensor
    region_idx: tf.Tensor
    brand_idx: tf.Tensor
    channel_idx: tf.Tensor
    signal_names: tuple[str, ...] = ("price",)


@dataclass
class PricingIntermediaries:
    offset_raw: tf.Tensor
    exponent_raw: tf.Tensor
    offset: tf.Tensor
    exponent: tf.Tensor
    volume_multiplier: tf.Tensor
    impact_by_signal: tf.Tensor
    impact: tf.Tensor


class SimplePricingLayer(keras.layers.Layer):
    """Standalone version of Watchtower-style multiplicative pricing elasticity."""

    def __init__(
        self,
        n_regions: int,
        n_brands: int,
        n_channels: int,
        l2_strength: float = 0.0,
        axis_l2: dict[str, float] | None = None,
        global_initializer: np.ndarray | None = None,
        name: str | None = "pricing",
    ):
        super().__init__(name=name)
        self.params = SimpleHierEmbedding(
            2,
            n_regions=n_regions,
            n_brands=n_brands,
            n_channels=n_channels,
            l2_strength=l2_strength,
            axis_l2=axis_l2,
            global_initializer=global_initializer,
            name="pricing_params",
        )

    def from_input(self, batch: PricingInput, training: bool = False) -> PricingIntermediaries:
        return self(batch.prices, batch.region_idx, batch.brand_idx, batch.channel_idx, training=training)

    def call(
        self,
        prices: tf.Tensor,
        region_idx: tf.Tensor,
        brand_idx: tf.Tensor,
        channel_idx: tf.Tensor,
        training: bool = False,
    ) -> PricingIntermediaries:
        raw = self.params(region_idx, brand_idx, channel_idx)
        offset_raw = raw[:, 0]
        exponent_raw = raw[:, 1]
        offset = 0.01 + tf.nn.softplus(offset_raw)
        exponent = 0.5 + 4.5 * tf.math.sigmoid(exponent_raw)
        prices = tf.where(prices > 0, prices, tf.ones_like(prices))
        volume_multiplier = tf.pow(1.0 + offset[:, None], exponent[:, None]) / tf.pow(
            prices + offset[:, None], exponent[:, None]
        )
        impact_by_signal = volume_multiplier[:, :, None]
        return PricingIntermediaries(
            offset_raw=offset_raw,
            exponent_raw=exponent_raw,
            offset=offset,
            exponent=exponent,
            volume_multiplier=volume_multiplier,
            impact_by_signal=impact_by_signal,
            impact=volume_multiplier,
        )

    def deviation_l2(self) -> tf.Tensor:
        return self.params.deviation_l2()

    def numpy_components(self) -> dict[str, object]:
        return self.params.numpy_components()
