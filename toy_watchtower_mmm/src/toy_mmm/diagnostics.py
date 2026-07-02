from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from toy_mmm.artifacts import build_impact_df, series_meta
from toy_mmm.layers import bounded_np
from toy_mmm.model import ToyMMMModel
from toy_mmm.normalization import NormalizedToyData


def _summary(values: np.ndarray) -> dict[str, float | int]:
    clean = np.asarray(values, dtype=np.float64).reshape(-1)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return {"finite_count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "finite_count": int(len(clean)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "p05": float(np.percentile(clean, 5)),
        "p50": float(np.percentile(clean, 50)),
        "p95": float(np.percentile(clean, 95)),
        "max": float(np.max(clean)),
    }


def collect_intermediaries(checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
    parts = checkpoint["parts"]
    return {
        "prediction": checkpoint["prediction"],
        "baseline": parts["baseline"],
        "generic_feature_mult_by_signal": parts["generic_feature_mult_by_signal"],
        "pricing_multiplier": parts["pricing_multiplier"],
        "multiplicative_impact": parts["multiplicative_impact"],
        "roi_instant": parts["instant_impact"],
        "roi_carryover": parts["carryover_impact"],
        "roi_total": parts["total_vehicle_impact"],
        "alpha": np.asarray(parts["alpha"]),
    }


def flatten_intermediaries(intermediaries: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for path, value in intermediaries.items():
        arr = np.asarray(value)
        rows.append({"path": path, "shape": tuple(arr.shape), **_summary(arr)})
    return pd.DataFrame(rows)


def _component_frame(axis: str, labels: list[str], values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    if values.ndim == 1:
        values = values[None, :]
    df = pd.DataFrame(values, columns=columns)
    df.insert(0, axis, labels)
    return df


def _embedding_frames(prefix: str, components: dict[str, np.ndarray], metadata: dict[str, Any], columns: list[str]) -> dict[str, pd.DataFrame]:
    enc = metadata["encodings"]
    frames = {
        f"{prefix}/global": pd.DataFrame([components["global"]], columns=columns),
        f"{prefix}/region": _component_frame("region", list(enc["region"].keys()), components["region"], columns),
        f"{prefix}/brand": _component_frame("brand", list(enc["brand"].keys()), components["brand"], columns),
        f"{prefix}/channel": _component_frame("channel", list(enc["channel"].keys()), components["channel"], columns),
    }
    if "vehicle" in components:
        frames[f"{prefix}/vehicle"] = _component_frame("vehicle", list(enc["vehicle"].keys()), components["vehicle"], columns)
    return frames


def layer_to_df(model: ToyMMMModel, metadata: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    feature_names = list(metadata["encodings"]["generic_feature"].keys())
    frames.update(_embedding_frames("baseline", model.baseline.numpy_components(), metadata, ["baseline_raw"]))
    frames.update(_embedding_frames("feature_beta", model.feature_beta.numpy_components(), metadata, feature_names))
    for feature_name, components in model.feature_beta.numpy_components_by_feature().items():
        frames.update(_embedding_frames(f"feature_beta/{feature_name}", components, metadata, [feature_name]))
    frames.update(_embedding_frames("pricing", model.pricing.params.numpy_components(), metadata, ["offset_raw", "exponent_raw"]))
    frames.update(
        _embedding_frames(
            "sigmoid_curve",
            model.sigmoid_curve.params.numpy_components(),
            metadata,
            ["asymptote_raw", "slope_raw"],
        )
    )
    frames.update(
        _embedding_frames(
            "beta_gamma",
            model.beta_gamma.params.numpy_components(),
            metadata,
            ["beta_raw", "gamma_raw"],
        )
    )

    meta = series_meta(metadata)
    idx_r = meta["region_idx"].to_numpy()
    idx_b = meta["brand_idx"].to_numpy()
    idx_c = meta["channel_idx"].to_numpy()
    sigmoid = model.sigmoid_curve.params.numpy_components()
    beta_gamma = model.beta_gamma.params.numpy_components()
    sigmoid_series = (
        sigmoid["global"][None, :]
        + sigmoid["region"][idx_r]
        + sigmoid["brand"][idx_b]
        + sigmoid["channel"][idx_c]
    )
    beta_gamma_series = (
        beta_gamma["global"][None, :]
        + beta_gamma["region"][idx_r]
        + beta_gamma["brand"][idx_b]
        + beta_gamma["channel"][idx_c]
    )
    sigmoid_series_vehicle = sigmoid_series[:, None, :] + sigmoid["vehicle"][None, :, :]
    beta_gamma_series_vehicle = beta_gamma_series[:, None, :] + beta_gamma["vehicle"][None, :, :]
    rows = []
    for s_idx, row in meta.iterrows():
        for v_idx, vehicle in enumerate(metadata["encodings"]["vehicle"].keys()):
            sigmoid_raw = sigmoid_series_vehicle[s_idx, v_idx]
            beta_gamma_raw = beta_gamma_series_vehicle[s_idx, v_idx]
            rows.append(
                {
                    "series_id": int(row["series_id"]),
                    "region": row["region"],
                    "brand": row["brand"],
                    "channel": row["channel"],
                    "vehicle": vehicle,
                    "asymptote": bounded_np(sigmoid_raw[0], *model.roi_bounds["asymptote"]),
                    "slope": bounded_np(sigmoid_raw[1], *model.roi_bounds["slope"]),
                    "beta": bounded_np(beta_gamma_raw[0], *model.roi_bounds["beta"]),
                    "gamma": bounded_np(beta_gamma_raw[1], *model.roi_bounds["gamma"]),
                }
            )
    frames["roi_bounded/series_vehicle"] = pd.DataFrame(rows)
    frames["alpha/global"] = pd.DataFrame({"alpha_raw": [float(model.alpha_raw.numpy())], "alpha": [float(model.alpha().numpy())]})
    return frames


def build_stats_df(layer_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for layer_key, df in layer_frames.items():
        layer, axis = layer_key.split("/", 1)
        numeric = df.select_dtypes(include=[np.number])
        for parameter in numeric.columns:
            stats = _summary(numeric[parameter].to_numpy())
            count = stats["finite_count"]
            mean_or_std = stats["std"] if np.isfinite(stats["std"]) and stats["std"] > 0 else abs(stats["mean"])
            overfitting = mean_or_std * (count**2)
            rows.append(
                {
                    "layer": layer,
                    "axis": axis,
                    "parameter": parameter,
                    "count": count,
                    "l2_norm": float(np.sqrt(np.nansum(np.square(numeric[parameter].to_numpy())))),
                    "mean_or_std": mean_or_std,
                    "overfitting": overfitting,
                    **{k: v for k, v in stats.items() if k != "finite_count"},
                }
            )
    out = pd.DataFrame(rows)
    out["overfitting_rel_avg"] = out["overfitting"] / out.groupby(["layer", "parameter"])["overfitting"].transform("mean")
    return out


def build_epoch_impact_dfs(checkpoints: list[dict[str, Any]], data: NormalizedToyData) -> dict[int, pd.DataFrame]:
    return {int(ckpt["epoch"]): build_impact_df(data, ckpt) for ckpt in checkpoints}


def build_impact_convergence_df(epoch_impact_dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    final_epoch = max(epoch_impact_dfs)
    final = epoch_impact_dfs[final_epoch]
    keys = ["series_id", "date", "layer", "signal"]
    final_small = final[keys + ["contribution"]].rename(columns={"contribution": "final_contribution"})
    rows = []
    for epoch, df in epoch_impact_dfs.items():
        merged = df[keys + ["region", "brand", "channel", "contribution"]].merge(final_small, on=keys, how="left")
        merged["sq_diff"] = np.square(merged["contribution"] - merged["final_contribution"])
        for level in ["layer", "signal", "region", "brand", "channel"]:
            grouped = merged.groupby(level)["sq_diff"].mean().reset_index()
            grouped["epoch"] = epoch
            grouped["level"] = level
            grouped = grouped.rename(columns={level: "category", "sq_diff": "mse_to_final"})
            rows.append(grouped)
    out = pd.concat(rows, ignore_index=True)
    first = out.groupby(["level", "category"])["mse_to_final"].transform("first")
    out["relative_to_first"] = out["mse_to_final"] / np.maximum(first, 1e-12)
    return out
