from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from toy_mmm.beta_gamma import SimpleBetaGammaLayer
from toy_mmm.hyperparameters import (
    alpha_bounds_from_hyperparameters,
    embedding_axis_l2,
    resolve_hyperparameters,
    roi_bounds_from_hyperparameters,
)
from toy_mmm.layers import (
    FeatureBetaEmbedding,
    SimpleHierEmbedding,
    bounded_tf,
    raw_from_bounded_np,
)
from toy_mmm.pricing import SimplePricingLayer
from toy_mmm.sigmoid_curve import SimpleSigmoidCurveLayer


class ToyMMMModel(keras.Model):
    def __init__(
        self,
        metadata: dict[str, Any],
        investment_axis_scale: np.ndarray,
        baseline_init: float,
        l2_strength: float = 0.0,
        hyperparameters: dict[str, Any] | None = None,
        name: str = "toy_mmm",
    ):
        super().__init__(name=name)
        self.hyperparameters = resolve_hyperparameters(hyperparameters)
        axis_hyperparameters = self.hyperparameters if hyperparameters is not None else None
        enc = metadata["encodings"]
        self.metadata = metadata
        self.n_regions = len(enc["region"])
        self.n_brands = len(enc["brand"])
        self.n_channels = len(enc["channel"])
        self.n_features = len(enc["generic_feature"])
        self.n_vehicles = len(enc["vehicle"])
        self.feature_names = tuple(enc["generic_feature"].keys())
        self.vehicle_names = tuple(enc["vehicle"].keys())
        self.decay_length = int(metadata["config"]["decay_length"])
        self.roi_bounds = roi_bounds_from_hyperparameters(self.hyperparameters)
        self.alpha_bounds = alpha_bounds_from_hyperparameters(self.hyperparameters)
        self.investment_axis_scale = tf.constant(investment_axis_scale.astype(np.float32), dtype=tf.float32)
        norm_factors = metadata["normalizations"]["norm_factors"]
        self.roi_unit_scale = tf.constant(
            float(norm_factors["normalization_factor_spend"]) / float(norm_factors["normalization_factor"]),
            dtype=tf.float32,
        )
        self.baseline_offset = tf.constant(float(baseline_init), dtype=tf.float32)
        norm_factor = float(norm_factors["normalization_factor"])
        config = metadata["config"]
        feature_prior = np.asarray(config["feature_beta_prior"], dtype=np.float32)
        pricing_offset_prior = float(config["pricing_offset_prior"])
        pricing_exponent_prior = float(config["pricing_exponent_prior"])
        pricing_raw_init = np.array(
            [
                np.log(np.expm1(max(pricing_offset_prior - 0.01, 1e-5))),
                raw_from_bounded_np(pricing_exponent_prior, *config["pricing_exponent_bounds"]),
            ],
            dtype=np.float32,
        )
        roi_vehicle_prior = config["roi_vehicle_prior"]
        sigmoid_vehicle_raw = np.stack(
            [
                raw_from_bounded_np(np.asarray(roi_vehicle_prior["asymptote"], dtype=np.float32), *self.roi_bounds["asymptote"]),
                raw_from_bounded_np(np.asarray(roi_vehicle_prior["slope"], dtype=np.float32), *self.roi_bounds["slope"]),
            ],
            axis=-1,
        ).astype(np.float32)
        beta_gamma_vehicle_raw = np.stack(
            [
                raw_from_bounded_np(np.asarray(roi_vehicle_prior["beta"], dtype=np.float32), *self.roi_bounds["beta"]),
                raw_from_bounded_np(np.asarray(roi_vehicle_prior["gamma"], dtype=np.float32), *self.roi_bounds["gamma"]),
            ],
            axis=-1,
        ).astype(np.float32)
        sigmoid_global_init = sigmoid_vehicle_raw.mean(axis=0)
        sigmoid_vehicle_init = sigmoid_vehicle_raw - sigmoid_global_init[None, :]
        beta_gamma_global_init = beta_gamma_vehicle_raw.mean(axis=0)
        beta_gamma_vehicle_init = beta_gamma_vehicle_raw - beta_gamma_global_init[None, :]
        alpha_prior_raw = raw_from_bounded_np(float(config["alpha_prior"]), *self.alpha_bounds)
        self.baseline = SimpleHierEmbedding(
            1,
            self.n_regions,
            self.n_brands,
            self.n_channels,
            l2_strength=l2_strength,
            axis_l2=embedding_axis_l2(axis_hyperparameters, "baseline", fallback_l2_strength=l2_strength),
            global_initializer=np.array([float(config["baseline_prior"]) / norm_factor - float(baseline_init)], dtype=np.float32),
            name="baseline",
        )
        self.feature_beta = FeatureBetaEmbedding(
            self.feature_names,
            self.n_regions,
            self.n_brands,
            self.n_channels,
            l2_strength=l2_strength,
            axis_l2_by_feature={
                feature_name: embedding_axis_l2(
                    axis_hyperparameters,
                    f"feature_beta.{feature_name}",
                    fallback_l2_strength=l2_strength,
                )
                for feature_name in self.feature_names
            },
            global_initializer=feature_prior,
            name="feature_beta",
        )
        self.pricing = SimplePricingLayer(
            self.n_regions,
            self.n_brands,
            self.n_channels,
            l2_strength=l2_strength,
            axis_l2=embedding_axis_l2(axis_hyperparameters, "pricing", fallback_l2_strength=l2_strength),
            global_initializer=pricing_raw_init,
            name="pricing",
        )
        self.sigmoid_curve = SimpleSigmoidCurveLayer(
            self.n_regions,
            self.n_brands,
            self.n_channels,
            n_vehicles=self.n_vehicles,
            bounds=self.roi_bounds,
            l2_strength=l2_strength,
            axis_l2=embedding_axis_l2(axis_hyperparameters, "sigmoid_curve", fallback_l2_strength=l2_strength),
            global_initializer=sigmoid_global_init,
            vehicle_initializer=sigmoid_vehicle_init,
            name="sigmoid_curve",
        )
        self.beta_gamma = SimpleBetaGammaLayer(
            self.n_regions,
            self.n_brands,
            self.n_channels,
            n_vehicles=self.n_vehicles,
            bounds=self.roi_bounds,
            decay_length=self.decay_length,
            l2_strength=l2_strength,
            axis_l2=embedding_axis_l2(axis_hyperparameters, "beta_gamma", fallback_l2_strength=l2_strength),
            global_initializer=beta_gamma_global_init,
            vehicle_initializer=beta_gamma_vehicle_init,
            name="beta_gamma",
        )
        self.alpha_raw = self.add_weight(
            name="alpha_raw", shape=(), initializer=keras.initializers.Constant(alpha_prior_raw), trainable=True
        )

    def alpha(self) -> tf.Tensor:
        lower, upper = self.alpha_bounds
        return bounded_tf(self.alpha_raw, lower, upper)

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False, return_parts: bool = False):
        region_idx = inputs["region_idx"]
        brand_idx = inputs["brand_idx"]
        channel_idx = inputs["channel_idx"]
        features = inputs["features"]
        price = inputs["price"]
        spends = inputs["vehicle_spends"]

        baseline = self.baseline_offset + self.baseline(region_idx, brand_idx, channel_idx)[:, 0]
        feature_betas = self.feature_beta(region_idx, brand_idx, channel_idx)
        feature_log_mult = tf.clip_by_value(features * feature_betas[:, None, :], -1.0, 1.0)
        generic_feature_mult_by_signal = tf.exp(feature_log_mult)
        generic_feature_mult = tf.reduce_prod(generic_feature_mult_by_signal, axis=-1)

        pricing = self.pricing(price, region_idx, brand_idx, channel_idx, training=training)
        multiplicative_impact = generic_feature_mult * pricing.impact
        multiplicative_by_signal = tf.concat([generic_feature_mult_by_signal, pricing.impact_by_signal], axis=-1)

        normalized_spend = spends / self.investment_axis_scale[None, None, :]
        sigmoid = self.sigmoid_curve(
            normalized_spend,
            self.investment_axis_scale,
            self.roi_unit_scale,
            region_idx,
            brand_idx,
            channel_idx,
        )
        beta_gamma = self.beta_gamma(sigmoid.instant_impact, region_idx, brand_idx, channel_idx)
        total_vehicle_impact = sigmoid.instant_impact + beta_gamma.carryover_impact
        roi_impact = tf.reduce_sum(total_vehicle_impact, axis=-1)
        alpha = self.alpha()
        roi_scaled_by_mult = tf.pow(tf.maximum(multiplicative_impact, 1e-6), alpha) * roi_impact
        prediction = baseline[:, None] * multiplicative_impact + roi_scaled_by_mult

        if not return_parts:
            return prediction

        return prediction, {
            "baseline": baseline,
            "feature_betas": feature_betas,
            "generic_feature_mult_by_signal": generic_feature_mult_by_signal,
            "generic_feature_mult": generic_feature_mult,
            "pricing_offset_raw": pricing.offset_raw,
            "pricing_exponent_raw": pricing.exponent_raw,
            "pricing_offset": pricing.offset,
            "pricing_exponent": pricing.exponent,
            "pricing_multiplier": pricing.impact,
            "pricing_impact_by_signal": pricing.impact_by_signal,
            "multiplicative_by_signal": multiplicative_by_signal,
            "multiplicative_impact": multiplicative_impact,
            "sigmoid_curve_raw": sigmoid.raw,
            "asymptote_raw": sigmoid.asymptote_raw,
            "slope_raw": sigmoid.slope_raw,
            "asymptote": sigmoid.asymptote,
            "slope": sigmoid.slope,
            "beta_gamma_raw": beta_gamma.raw,
            "beta_raw": beta_gamma.beta_raw,
            "gamma_raw": beta_gamma.gamma_raw,
            "beta": beta_gamma.beta,
            "gamma": beta_gamma.gamma,
            "instant_impact": sigmoid.instant_impact,
            "carryover_impact": beta_gamma.carryover_impact,
            "total_vehicle_impact": total_vehicle_impact,
            "roi_impact": roi_impact,
            "roi_scaled_by_mult": roi_scaled_by_mult,
            "alpha": alpha,
        }

    def regularization_loss(self) -> tf.Tensor:
        return (
            self.baseline.deviation_l2()
            + self.feature_beta.deviation_l2()
            + self.pricing.deviation_l2()
            + self.sigmoid_curve.deviation_l2()
            + self.beta_gamma.deviation_l2()
        )

    def deviation_norms(self) -> dict[str, float]:
        norms = {}
        for layer_name, layer in [
            ("baseline", self.baseline),
            ("feature_beta", self.feature_beta),
            ("pricing", self.pricing.params),
            ("sigmoid_curve", self.sigmoid_curve.params),
            ("beta_gamma", self.beta_gamma.params),
        ]:
            for axis, values in layer.numpy_components().items():
                if axis == "global":
                    continue
                norms[f"{layer_name}/{axis}"] = float(np.sqrt(np.mean(np.square(values))))
        return norms
