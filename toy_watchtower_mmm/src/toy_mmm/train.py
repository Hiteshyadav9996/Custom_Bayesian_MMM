from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from toy_mmm.hyperparameters import resolve_hyperparameters
from toy_mmm.model import ToyMMMModel
from toy_mmm.normalization import NormalizedToyData


@dataclass
class TrainingResult:
    model: ToyMMMModel
    history: pd.DataFrame
    checkpoints: list[dict[str, Any]]
    tensors: dict[str, tf.Tensor]
    hyperparameters: dict[str, Any]


def make_tensor_dict(data: NormalizedToyData | dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    if isinstance(data, NormalizedToyData):
        arrays = {
            "features": data.features,
            "price": data.price,
            "vehicle_spends": data.vehicle_spends,
            "region_idx": data.region_idx,
            "brand_idx": data.brand_idx,
            "channel_idx": data.channel_idx,
        }
    else:
        arrays = data
    return {
        "features": tf.constant(arrays["features"], dtype=tf.float32),
        "price": tf.constant(arrays["price"], dtype=tf.float32),
        "vehicle_spends": tf.constant(arrays["vehicle_spends"], dtype=tf.float32),
        "region_idx": tf.constant(arrays["region_idx"], dtype=tf.int32),
        "brand_idx": tf.constant(arrays["brand_idx"], dtype=tf.int32),
        "channel_idx": tf.constant(arrays["channel_idx"], dtype=tf.int32),
    }


def _mask_to_indices(mask: np.ndarray) -> tf.Tensor:
    return tf.constant(np.flatnonzero(mask).astype(np.int32), dtype=tf.int32)


def masked_mse(y_true: tf.Tensor, y_pred: tf.Tensor, week_index: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(tf.gather(y_true, week_index, axis=1) - tf.gather(y_pred, week_index, axis=1)))


def snapshot_model(model: ToyMMMModel, tensors: dict[str, tf.Tensor], epoch: int) -> dict[str, Any]:
    pred, parts = model(tensors, training=False, return_parts=True)
    parts_np = {key: value.numpy() if hasattr(value, "numpy") else value for key, value in parts.items()}
    return {
        "epoch": int(epoch),
        "prediction": pred.numpy(),
        "parts": parts_np,
        "weights": {var.name: var.numpy().copy() for var in model.trainable_variables},
        "deviation_norms": model.deviation_norms(),
    }


def train_model(
    data: NormalizedToyData,
    epochs: int | None = None,
    learning_rate: float | None = None,
    l2_strength: float | None = None,
    snapshot_interval: int | None = None,
    hyperparameters: dict[str, Any] | str | Path | None = None,
    verbose: bool = True,
) -> TrainingResult:
    keras.backend.clear_session()
    tf.random.set_seed(int(data.metadata.get("config", {}).get("seed", 20260702)))
    hparams = resolve_hyperparameters(hyperparameters)
    model_hparams = hparams if hyperparameters is not None else None
    training_cfg = hparams.get("training", {})
    tensors = make_tensor_dict(data)
    y = tf.constant(data.target, dtype=tf.float32)
    train_idx = _mask_to_indices(data.train_week_mask)
    val_idx = _mask_to_indices(data.val_week_mask)
    epochs = int(epochs if epochs is not None else training_cfg.get("epochs", data.metadata["config"].get("training_epochs", 1024)))
    learning_rate = float(learning_rate if learning_rate is not None else training_cfg.get("learning_rate", 0.018))
    l2_strength = float(l2_strength if l2_strength is not None else training_cfg.get("fallback_l2_strength", 0.0))
    history_interval = int(training_cfg.get("history_interval", 5))
    baseline_init = float(np.median(data.target[:, data.train_week_mask]))
    model = ToyMMMModel(
        metadata=data.metadata,
        investment_axis_scale=data.investment_axis_scale,
        baseline_init=baseline_init,
        l2_strength=l2_strength,
        hyperparameters=model_hparams,
    )
    _ = model(tensors, training=False)
    snapshot_interval = snapshot_interval or int(training_cfg.get("snapshot_interval", data.metadata["config"]["snapshot_interval"]))

    if verbose:
        print(f"baseline offset initialized from train median: {baseline_init:.3f}")
        print(f"optimizer=adam learning_rate={learning_rate:.5f}")

    optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError(f"Only adam is implemented in the toy training loop, got {optimizer_name!r}")
    optimizer = keras.optimizers.Adam(learning_rate)
    history_rows = []
    checkpoints: list[dict[str, Any]] = []

    if verbose:
        print(f"full training for {epochs} epochs")
    for epoch in range(epochs + 1):
        if epoch > 0:
            with tf.GradientTape() as tape:
                pred = model(tensors, training=True)
                train_loss = masked_mse(y, pred, train_idx)
                reg_loss = model.regularization_loss()
                loss = train_loss + reg_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients((g, v) for g, v in zip(grads, model.trainable_variables) if g is not None)

        if epoch % history_interval == 0 or epoch == epochs:
            pred_eval = model(tensors, training=False)
            train_mse = float(masked_mse(y, pred_eval, train_idx).numpy())
            val_mse = float(masked_mse(y, pred_eval, val_idx).numpy())
            reg_value = float(model.regularization_loss().numpy())
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "regularization": reg_value,
                    "objective": train_mse + reg_value,
                }
            )
            if verbose and (epoch % 50 == 0 or epoch == epochs):
                print(
                    f"  epoch {epoch:03d}: train_mse={train_mse:.4f}, "
                    f"val_mse={val_mse:.4f}, reg={reg_value:.4f}"
                )
        if epoch % snapshot_interval == 0 or epoch == epochs:
            checkpoints.append(snapshot_model(model, tensors, epoch))

    return TrainingResult(
        model=model,
        history=pd.DataFrame(history_rows),
        checkpoints=checkpoints,
        tensors=tensors,
        hyperparameters=hparams,
    )
