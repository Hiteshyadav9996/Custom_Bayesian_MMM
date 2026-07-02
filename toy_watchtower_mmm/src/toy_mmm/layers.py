from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras


def bounded_np(raw: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return lower + (upper - lower) / (1.0 + np.exp(-raw))


def raw_from_bounded_np(value: np.ndarray | float, lower: float, upper: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    p = np.clip((value - lower) / (upper - lower), 1e-5, 1 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def bounded_tf(raw: tf.Tensor, lower: float, upper: float) -> tf.Tensor:
    return tf.cast(lower, tf.float32) + tf.cast(upper - lower, tf.float32) * tf.math.sigmoid(raw)


def normalized_sigmoid_curve_np(x: np.ndarray, asymptote: np.ndarray, slope: np.ndarray) -> np.ndarray:
    x_offset = 0.7
    y_offset = 1.0 / (1.0 + np.exp(-x_offset))
    return asymptote * ((1.0 / (1.0 + np.exp(-(x * slope / (y_offset * asymptote) + x_offset)))) - y_offset) / (
        1.0 - y_offset
    )


def normalized_sigmoid_curve_tf(x: tf.Tensor, asymptote: tf.Tensor, slope: tf.Tensor) -> tf.Tensor:
    x_offset = tf.constant(0.7, tf.float32)
    y_offset = tf.math.sigmoid(x_offset)
    return asymptote * (tf.math.sigmoid(x * slope / (y_offset * asymptote) + x_offset) - y_offset) / (
        1.0 - y_offset
    )


def beta_gamma_carryover_np(
    instant_impact: np.ndarray, beta: np.ndarray, gamma: np.ndarray, decay_length: int
) -> np.ndarray:
    carryover = np.zeros_like(instant_impact, dtype=np.float32)
    for lag in range(1, decay_length + 1):
        shifted = np.zeros_like(instant_impact, dtype=np.float32)
        shifted[:, lag:, :] = instant_impact[:, :-lag, :]
        carryover += shifted * beta[:, None, :] * (gamma[:, None, :] ** (lag - 1))
    return carryover.astype(np.float32)


def beta_gamma_carryover_tf(
    instant_impact: tf.Tensor, beta: tf.Tensor, gamma: tf.Tensor, decay_length: int
) -> tf.Tensor:
    carryover = tf.zeros_like(instant_impact)
    for lag in range(1, decay_length + 1):
        zeros = tf.zeros_like(instant_impact[:, :lag, :])
        shifted = tf.concat([zeros, instant_impact[:, :-lag, :]], axis=1)
        carryover = carryover + shifted * beta[:, None, :] * tf.pow(gamma[:, None, :], lag - 1)
    return carryover


def centered(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return (values - values.mean(axis=axis, keepdims=True)).astype(np.float32)


def centered_normal(rng: np.random.Generator, shape: tuple[int, ...], scale: Any) -> np.ndarray:
    return centered(rng.normal(0, scale, size=shape).astype(np.float32), axis=0)


class SimpleHierEmbedding(keras.layers.Layer):
    """Additive hierarchy: global + region + brand + channel + optional vehicle."""

    def __init__(
        self,
        output_dim: int,
        n_regions: int,
        n_brands: int,
        n_channels: int,
        n_vehicles: int | None = None,
        l2_strength: float = 0.0,
        axis_l2: dict[str, float] | None = None,
        global_initializer: np.ndarray | None = None,
        region_initializer: np.ndarray | None = None,
        brand_initializer: np.ndarray | None = None,
        channel_initializer: np.ndarray | None = None,
        vehicle_initializer: np.ndarray | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.output_dim = int(output_dim)
        self.n_regions = int(n_regions)
        self.n_brands = int(n_brands)
        self.n_channels = int(n_channels)
        self.n_vehicles = None if n_vehicles is None else int(n_vehicles)
        self.l2_strength = float(l2_strength)
        self.axis_l2 = {
            "global": 0.0,
            "region": self.l2_strength,
            "brand": self.l2_strength,
            "channel": self.l2_strength,
            "vehicle": self.l2_strength,
        }
        if axis_l2 is not None:
            self.axis_l2.update({key: float(value) for key, value in axis_l2.items()})
        self.global_initializer = None if global_initializer is None else np.asarray(global_initializer, dtype=np.float32)
        self.region_initializer = None if region_initializer is None else np.asarray(region_initializer, dtype=np.float32)
        self.brand_initializer = None if brand_initializer is None else np.asarray(brand_initializer, dtype=np.float32)
        self.channel_initializer = None if channel_initializer is None else np.asarray(channel_initializer, dtype=np.float32)
        self.vehicle_initializer = None if vehicle_initializer is None else np.asarray(vehicle_initializer, dtype=np.float32)

    def build(self, input_shape):
        def initializer(value):
            if value is None:
                return "zeros"
            return keras.initializers.Constant(value)

        self.global_value = self.add_weight(
            name="global", shape=(self.output_dim,), initializer=initializer(self.global_initializer), trainable=True
        )
        self.region_dev = self.add_weight(
            name="region_dev",
            shape=(self.n_regions, self.output_dim),
            initializer=initializer(self.region_initializer),
            trainable=True,
        )
        self.brand_dev = self.add_weight(
            name="brand_dev",
            shape=(self.n_brands, self.output_dim),
            initializer=initializer(self.brand_initializer),
            trainable=True,
        )
        self.channel_dev = self.add_weight(
            name="channel_dev",
            shape=(self.n_channels, self.output_dim),
            initializer=initializer(self.channel_initializer),
            trainable=True,
        )
        self.vehicle_dev = None
        if self.n_vehicles is not None:
            self.vehicle_dev = self.add_weight(
                name="vehicle_dev",
                shape=(self.n_vehicles, self.output_dim),
                initializer=initializer(self.vehicle_initializer),
                trainable=True,
            )

    def centered_deviations(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor | None]:
        region = self.region_dev - tf.reduce_mean(self.region_dev, axis=0, keepdims=True)
        brand = self.brand_dev - tf.reduce_mean(self.brand_dev, axis=0, keepdims=True)
        channel = self.channel_dev - tf.reduce_mean(self.channel_dev, axis=0, keepdims=True)
        vehicle = None
        if self.vehicle_dev is not None:
            vehicle = self.vehicle_dev - tf.reduce_mean(self.vehicle_dev, axis=0, keepdims=True)
        return region, brand, channel, vehicle

    def call(self, region_index: tf.Tensor, brand_index: tf.Tensor, channel_index: tf.Tensor) -> tf.Tensor:
        region, brand, channel, vehicle = self.centered_deviations()
        base = (
            self.global_value[None, :]
            + tf.gather(region, region_index)
            + tf.gather(brand, brand_index)
            + tf.gather(channel, channel_index)
        )
        if vehicle is None:
            return base
        return base[:, None, :] + vehicle[None, :, :]

    def deviation_l2(self) -> tf.Tensor:
        region, brand, channel, vehicle = self.centered_deviations()
        pieces = []
        for axis_name, values in [
            ("global", self.global_value),
            ("region", region),
            ("brand", brand),
            ("channel", channel),
            ("vehicle", vehicle),
        ]:
            strength = float(self.axis_l2.get(axis_name, 0.0))
            if values is not None and strength != 0.0:
                pieces.append(tf.cast(strength, tf.float32) * tf.reduce_sum(tf.square(values)))
        if not pieces:
            return tf.constant(0.0, tf.float32)
        return tf.add_n(pieces)

    def numpy_components(self) -> dict[str, np.ndarray]:
        region, brand, channel, vehicle = self.centered_deviations()
        out = {
            "global": self.global_value.numpy().copy(),
            "region": region.numpy().copy(),
            "brand": brand.numpy().copy(),
            "channel": channel.numpy().copy(),
        }
        if vehicle is not None:
            out["vehicle"] = vehicle.numpy().copy()
        return out


class FeatureBetaEmbedding(keras.layers.Layer):
    """One hierarchy embedding per multiplicative feature signal."""

    def __init__(
        self,
        feature_names: tuple[str, ...],
        n_regions: int,
        n_brands: int,
        n_channels: int,
        l2_strength: float = 0.0,
        axis_l2_by_feature: dict[str, dict[str, float]] | None = None,
        global_initializer: np.ndarray | None = None,
        name: str | None = "feature_beta",
    ):
        super().__init__(name=name)
        self.feature_names = tuple(feature_names)
        axis_l2_by_feature = axis_l2_by_feature or {}
        global_initializer = None if global_initializer is None else np.asarray(global_initializer, dtype=np.float32)
        self.feature_embeddings = []
        for feature_idx, feature_name in enumerate(self.feature_names):
            safe_name = feature_name.replace("-", "_").replace(" ", "_")
            feature_initializer = None
            if global_initializer is not None:
                feature_initializer = np.array([global_initializer[feature_idx]], dtype=np.float32)
            self.feature_embeddings.append(
                SimpleHierEmbedding(
                    1,
                    n_regions=n_regions,
                    n_brands=n_brands,
                    n_channels=n_channels,
                    l2_strength=l2_strength,
                    axis_l2=axis_l2_by_feature.get(feature_name),
                    global_initializer=feature_initializer,
                    name=f"{safe_name}_beta",
                )
            )

    def call(self, region_index: tf.Tensor, brand_index: tf.Tensor, channel_index: tf.Tensor) -> tf.Tensor:
        values = [layer(region_index, brand_index, channel_index) for layer in self.feature_embeddings]
        return tf.concat(values, axis=1)

    def deviation_l2(self) -> tf.Tensor:
        pieces = [layer.deviation_l2() for layer in self.feature_embeddings]
        if not pieces:
            return tf.constant(0.0, tf.float32)
        return tf.add_n(pieces)

    def numpy_components_by_feature(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            feature_name: layer.numpy_components()
            for feature_name, layer in zip(self.feature_names, self.feature_embeddings, strict=True)
        }

    def numpy_components(self) -> dict[str, np.ndarray]:
        by_feature = self.numpy_components_by_feature()
        axis_names = ["global", "region", "brand", "channel"]
        return {
            axis: np.concatenate([by_feature[feature][axis] for feature in self.feature_names], axis=-1)
            for axis in axis_names
        }
