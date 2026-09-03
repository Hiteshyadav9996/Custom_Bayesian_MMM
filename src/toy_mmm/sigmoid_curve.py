from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras

from toy_mmm.layers import SimpleHierEmbedding, bounded_tf, normalized_sigmoid_curve_tf


@dataclass
class SigmoidCurveIntermediaries:
    raw: tf.Tensor
    asymptote_raw: tf.Tensor
    slope_raw: tf.Tensor
    asymptote: tf.Tensor
    slope: tf.Tensor
    instant_impact: tf.Tensor


class SimpleSigmoidCurveLayer(keras.layers.Layer):
    """Vehicle-level sigmoid ROI curve with its own hierarchy embedding."""

    def __init__(
        self,
        n_regions: int,
        n_brands: int,
        n_channels: int,
        n_vehicles: int,
        bounds: dict[str, tuple[float, float]],
        l2_strength: float = 0.0,
        axis_l2: dict[str, float] | None = None,
        global_initializer: np.ndarray | None = None,
        vehicle_initializer: np.ndarray | None = None,
        name: str | None = "sigmoid_curve",
    ):
        super().__init__(name=name)
        self.bounds = bounds
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
            name="sigmoid_curve_params",
        )

    def call(
        self,
        normalized_spend: tf.Tensor,
        investment_axis_scale: tf.Tensor,
        roi_unit_scale: tf.Tensor,
        region_idx: tf.Tensor,
        brand_idx: tf.Tensor,
        channel_idx: tf.Tensor,
    ) -> SigmoidCurveIntermediaries:
        raw = self.params(region_idx, brand_idx, channel_idx)
        asymptote_raw = raw[:, :, 0]
        slope_raw = raw[:, :, 1]
        asymptote = bounded_tf(asymptote_raw, *self.bounds["asymptote"])
        slope = bounded_tf(slope_raw, *self.bounds["slope"])
        instant_impact = roi_unit_scale * investment_axis_scale[None, None, :] * normalized_sigmoid_curve_tf(
            normalized_spend,
            asymptote[:, None, :],
            slope[:, None, :],
        )
        return SigmoidCurveIntermediaries(
            raw=raw,
            asymptote_raw=asymptote_raw,
            slope_raw=slope_raw,
            asymptote=asymptote,
            slope=slope,
            instant_impact=instant_impact,
        )

    def deviation_l2(self) -> tf.Tensor:
        return self.params.deviation_l2()
