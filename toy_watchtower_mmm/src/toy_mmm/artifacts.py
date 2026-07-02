from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from toy_mmm.layers import normalized_sigmoid_curve_np
from toy_mmm.normalization import NormalizedToyData


def series_meta(metadata: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(metadata["series"]).sort_values("series_id").reset_index(drop=True)


def _date_series_frame(metadata: dict[str, Any]) -> pd.DataFrame:
    meta = series_meta(metadata)
    dates = pd.to_datetime(metadata["dates"])
    return pd.MultiIndex.from_product([meta["series_id"], dates], names=["series_id", "date"]).to_frame(index=False).merge(
        meta, on="series_id", how="left"
    )


def linearize_multiplicative_impacts(multipliers: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    prod = np.prod(multipliers, axis=2, keepdims=True)
    log_prod = np.log(np.maximum(prod, 1e-8))
    result = baseline[:, :, None] * np.log(np.maximum(multipliers, 1e-8)) * np.divide(
        prod - 1.0, log_prod, out=np.zeros_like(prod), where=np.abs(log_prod) > 1e-8
    )
    return np.nan_to_num(result).astype(np.float32)


def build_y_df(data: NormalizedToyData, checkpoint: dict[str, Any]) -> pd.DataFrame:
    factor = data.metadata["normalizations"]["norm_factors"]["normalization_factor"]
    base = _date_series_frame(data.metadata)
    base["observed"] = data.raw.target_raw.reshape(-1)
    base["predicted"] = (checkpoint["prediction"] * factor).reshape(-1)
    base["residual"] = base["observed"] - base["predicted"]
    return base


def build_input_signal_df(data: NormalizedToyData) -> pd.DataFrame:
    base = _date_series_frame(data.metadata)
    frames = []
    for f_idx, feature in enumerate(data.metadata["encodings"]["generic_feature"].keys()):
        df = base.copy()
        df["signal"] = feature
        df["layer"] = "generic_feature"
        df["value"] = data.raw.features_raw[:, :, f_idx].reshape(-1)
        frames.append(df)
    df = base.copy()
    df["signal"] = "price"
    df["layer"] = "pricing"
    df["value"] = data.raw.price_raw.reshape(-1)
    frames.append(df)
    for v_idx, vehicle in enumerate(data.metadata["encodings"]["vehicle"].keys()):
        df = base.copy()
        df["signal"] = vehicle
        df["layer"] = "roi"
        df["value"] = data.raw.spends_raw[:, :, v_idx].reshape(-1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_impact_df(data: NormalizedToyData, checkpoint: dict[str, Any]) -> pd.DataFrame:
    factor = data.metadata["normalizations"]["norm_factors"]["normalization_factor"]
    parts = checkpoint["parts"]
    base = _date_series_frame(data.metadata)
    frames = []

    baseline = parts["baseline"]
    baseline_df = base.copy()
    baseline_df["layer"] = "baseline"
    baseline_df["signal"] = "Baseline"
    baseline_df["contribution"] = np.repeat(baseline * factor, len(data.metadata["dates"]))
    frames.append(baseline_df)

    signal_names = list(data.metadata["encodings"]["generic_feature"].keys()) + ["price"]
    mult_contrib = linearize_multiplicative_impacts(parts["multiplicative_by_signal"], baseline[:, None]) * factor
    for s_idx, signal in enumerate(signal_names):
        df = base.copy()
        df["layer"] = "pricing" if signal == "price" else "generic_feature"
        df["signal"] = signal
        df["contribution"] = mult_contrib[:, :, s_idx].reshape(-1)
        frames.append(df)

    alpha = float(np.asarray(parts["alpha"]))
    roi_scale = np.power(np.maximum(parts["multiplicative_impact"], 1e-6), alpha)
    roi_contrib = parts["total_vehicle_impact"] * roi_scale[:, :, None] * factor
    for v_idx, vehicle in enumerate(data.metadata["encodings"]["vehicle"].keys()):
        df = base.copy()
        df["layer"] = "roi"
        df["signal"] = vehicle
        df["contribution"] = roi_contrib[:, :, v_idx].reshape(-1)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_roi_impact_df(data: NormalizedToyData, checkpoint: dict[str, Any]) -> pd.DataFrame:
    factor = data.metadata["normalizations"]["norm_factors"]["normalization_factor"]
    parts = checkpoint["parts"]
    alpha = float(np.asarray(parts["alpha"]))
    roi_scale = np.power(np.maximum(parts["multiplicative_impact"], 1e-6), alpha)
    base = _date_series_frame(data.metadata)
    frames = []
    for v_idx, vehicle in enumerate(data.metadata["encodings"]["vehicle"].keys()):
        df = base.copy()
        df["vehicle"] = vehicle
        df["instant_raw"] = (parts["instant_impact"][:, :, v_idx] * factor).reshape(-1)
        df["carryover_raw"] = (parts["carryover_impact"][:, :, v_idx] * factor).reshape(-1)
        df["total_raw"] = (parts["total_vehicle_impact"][:, :, v_idx] * factor).reshape(-1)
        df["scaled_total_contribution"] = (parts["total_vehicle_impact"][:, :, v_idx] * roi_scale * factor).reshape(-1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_pricing_df(data: NormalizedToyData, checkpoint: dict[str, Any]) -> pd.DataFrame:
    factor = data.metadata["normalizations"]["norm_factors"]["normalization_factor"]
    parts = checkpoint["parts"]
    baseline = parts["baseline"]
    mult_contrib = linearize_multiplicative_impacts(parts["multiplicative_by_signal"], baseline[:, None])
    price_contrib = mult_contrib[:, :, -1] * factor
    base = _date_series_frame(data.metadata)
    base["price_raw"] = data.raw.price_raw.reshape(-1)
    base["price_norm"] = data.price.reshape(-1)
    base["price_multiplier"] = parts["pricing_multiplier"].reshape(-1)
    base["pricing_contribution"] = price_contrib.reshape(-1)
    meta = series_meta(data.metadata)
    base["offset"] = np.repeat(parts["pricing_offset"][meta["series_id"].to_numpy()], len(data.metadata["dates"]))
    base["exponent"] = np.repeat(parts["pricing_exponent"][meta["series_id"].to_numpy()], len(data.metadata["dates"]))
    return base


def build_beta_gamma_df(data: NormalizedToyData, checkpoint: dict[str, Any]) -> pd.DataFrame:
    parts = checkpoint["parts"]
    meta = series_meta(data.metadata)
    rows = []
    for s_idx, row in meta.iterrows():
        for v_idx, vehicle in enumerate(data.metadata["encodings"]["vehicle"].keys()):
            rows.append(
                {
                    "series_id": int(row["series_id"]),
                    "region": row["region"],
                    "brand": row["brand"],
                    "channel": row["channel"],
                    "vehicle": vehicle,
                    "beta": parts["beta"][s_idx, v_idx],
                    "gamma": parts["gamma"][s_idx, v_idx],
                    "asymptote": parts["asymptote"][s_idx, v_idx],
                    "slope": parts["slope"][s_idx, v_idx],
                }
            )
    return pd.DataFrame(rows)


def build_curve_df(data: NormalizedToyData, checkpoint: dict[str, Any], selected_series: int = 0) -> pd.DataFrame:
    parts = checkpoint["parts"]
    meta = series_meta(data.metadata).set_index("series_id")
    rows = []
    spend_factor = data.metadata["normalizations"]["norm_factors"]["normalization_factor_spend"]
    target_factor = data.metadata["normalizations"]["norm_factors"]["normalization_factor"]
    roi_unit_scale = spend_factor / target_factor
    for v_idx, vehicle in enumerate(data.metadata["encodings"]["vehicle"].keys()):
        spend_max = max(data.raw.spends_raw[selected_series, :, v_idx].max() * 1.15, 1.0)
        spend_grid_raw = np.linspace(0.0, spend_max, 120)
        spend_grid_norm = spend_grid_raw / spend_factor
        scale = data.investment_axis_scale[v_idx]
        instant_norm = roi_unit_scale * scale * normalized_sigmoid_curve_np(
            spend_grid_norm / scale,
            parts["asymptote"][selected_series, v_idx],
            parts["slope"][selected_series, v_idx],
        )
        beta = parts["beta"][selected_series, v_idx]
        gamma = parts["gamma"][selected_series, v_idx]
        decay_length = data.metadata["config"]["decay_length"]
        total_norm = instant_norm * (1.0 + beta * (1.0 - gamma**decay_length) / (1.0 - gamma))
        for spend_raw, instant, total in zip(spend_grid_raw, instant_norm, total_norm):
            rows.append(
                {
                    "curve_type": "roi",
                    "series_id": selected_series,
                    "vehicle": vehicle,
                    "x_raw": spend_raw,
                    "instant_impact_raw": instant * target_factor,
                    "total_impact_raw": total * target_factor,
                    "point_type": "curve",
                }
            )
        point_mask = data.raw.spends_raw[selected_series, :, v_idx] > 0
        for spend_raw, instant, total in zip(
            data.raw.spends_raw[selected_series, point_mask, v_idx],
            parts["instant_impact"][selected_series, point_mask, v_idx],
            parts["total_vehicle_impact"][selected_series, point_mask, v_idx],
        ):
            rows.append(
                {
                    "curve_type": "roi",
                    "series_id": selected_series,
                    "vehicle": vehicle,
                    "x_raw": spend_raw,
                    "instant_impact_raw": instant * target_factor,
                    "total_impact_raw": total * target_factor,
                    "point_type": "historical",
                }
            )

    offset = parts["pricing_offset"][selected_series]
    exponent = parts["pricing_exponent"][selected_series]
    price_mean = data.metadata["normalizations"]["price_mean_by_series"][selected_series]
    price_grid_norm = np.linspace(0.75, 1.25, 120)
    price_mult = ((1 + offset) ** exponent) / ((price_grid_norm + offset) ** exponent)
    for price_norm, mult in zip(price_grid_norm, price_mult):
        rows.append(
            {
                "curve_type": "pricing",
                "series_id": selected_series,
                "vehicle": None,
                "x_raw": price_norm * price_mean,
                "instant_impact_raw": None,
                "total_impact_raw": mult,
                "point_type": "curve",
            }
        )
    for price_raw, price_norm, mult in zip(
        data.raw.price_raw[selected_series], data.price[selected_series], parts["pricing_multiplier"][selected_series]
    ):
        rows.append(
            {
                "curve_type": "pricing",
                "series_id": selected_series,
                "vehicle": None,
                "x_raw": price_raw,
                "instant_impact_raw": None,
                "total_impact_raw": mult,
                "point_type": "historical",
            }
        )
    df = pd.DataFrame(rows)
    for col in ["region", "brand", "channel"]:
        df[col] = meta.loc[selected_series, col]
    return df


def build_all_artifacts(data: NormalizedToyData, checkpoint: dict[str, Any], selected_series: int = 0) -> dict[str, pd.DataFrame]:
    return {
        "y_df": build_y_df(data, checkpoint),
        "input_signal_df": build_input_signal_df(data),
        "impact_df": build_impact_df(data, checkpoint),
        "roi_impact_df": build_roi_impact_df(data, checkpoint),
        "pricing_df": build_pricing_df(data, checkpoint),
        "beta_gamma_df": build_beta_gamma_df(data, checkpoint),
        "curve_df": build_curve_df(data, checkpoint, selected_series=selected_series),
    }
