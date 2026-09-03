from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_training_history(history: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, column, title in zip(
        axes,
        ["train_mse", "val_mse", "objective"],
        ["Train MSE", "Validation MSE", "Objective"],
    ):
        ax.plot(history["epoch"], history[column])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
    plt.tight_layout()
    return fig


def plot_prediction(y_df: pd.DataFrame, series_id: int = 0):
    subset = y_df[y_df["series_id"] == series_id]
    label = f"{subset['region'].iloc[0]} / {subset['brand'].iloc[0]} / {subset['channel'].iloc[0]}"
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(subset["date"], subset["observed"], label="observed", color="black", linewidth=1.3)
    ax.plot(subset["date"], subset["predicted"], label="predicted", alpha=0.85)
    ax.set_title(f"Observed vs predicted: {label}")
    ax.set_ylabel("target")
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig


def plot_historical_decomposition(
    impact_df: pd.DataFrame,
    y_df: pd.DataFrame,
    series_id: int = 0,
    start: int = 0,
    end: int = 90,
    stack_by: str = "signal",
):
    subset = impact_df[impact_df["series_id"] == series_id].copy()
    dates = sorted(subset["date"].unique())[start:end]
    subset = subset[subset["date"].isin(dates)]
    y_subset = y_df[(y_df["series_id"] == series_id) & (y_df["date"].isin(dates))]
    pivot = subset.pivot_table(index="date", columns=stack_by, values="contribution", aggfunc="sum").fillna(0)
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    for column in pivot.columns:
        fig.add_trace(
            go.Bar(
                x=pivot.index,
                y=pivot[column],
                name=str(column),
                hovertemplate="%{x|%Y-%m-%d}<br>" + str(column) + ": %{y:,.1f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=y_subset["date"],
            y=y_subset["observed"],
            mode="lines",
            name="observed volume",
            line=dict(color="black", width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>observed: %{y:,.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_subset["date"],
            y=y_subset["predicted"],
            mode="lines",
            name="predicted volume",
            line=dict(color="#1f77b4", width=2.0),
            hovertemplate="%{x|%Y-%m-%d}<br>predicted: %{y:,.1f}<extra></extra>",
        )
    )
    label = f"{y_subset['region'].iloc[0]} / {y_subset['brand'].iloc[0]} / {y_subset['channel'].iloc[0]}"
    fig.update_layout(
        title=f"Historical decomposition: {label}",
        barmode="relative",
        xaxis_title="week",
        yaxis_title="raw volume / contribution",
        legend_title=stack_by,
        height=560,
        template="plotly_white",
    )
    return fig


def plot_roi_curves(curve_df: pd.DataFrame):
    roi = curve_df[curve_df["curve_type"] == "roi"]
    vehicles = list(roi["vehicle"].dropna().unique())
    cols = 2
    rows = int(np.ceil(len(vehicles) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), squeeze=False)
    for ax, vehicle in zip(axes.ravel(), vehicles):
        cur = roi[(roi["vehicle"] == vehicle) & (roi["point_type"] == "curve")]
        pts = roi[(roi["vehicle"] == vehicle) & (roi["point_type"] == "historical")]
        ax.plot(cur["x_raw"], cur["total_impact_raw"], label="learned long-run curve")
        ax.scatter(pts["x_raw"], pts["total_impact_raw"], s=18, alpha=0.45, label="historical points")
        ax.set_title(vehicle)
        ax.set_xlabel("spend")
        ax.set_ylabel("impact")
        ax.legend(frameon=False)
    for ax in axes.ravel()[len(vehicles) :]:
        ax.axis("off")
    fig.suptitle("ROI curves with actual datapoints", y=1.01)
    plt.tight_layout()
    return fig


def plot_pricing_curve(curve_df: pd.DataFrame):
    pricing = curve_df[curve_df["curve_type"] == "pricing"]
    cur = pricing[pricing["point_type"] == "curve"]
    pts = pricing[pricing["point_type"] == "historical"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cur["x_raw"], cur["total_impact_raw"], label="learned pricing curve")
    ax.scatter(pts["x_raw"], pts["total_impact_raw"], s=18, alpha=0.45, label="historical price weeks")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_title("Pricing multiplier curve")
    ax.set_xlabel("raw price")
    ax.set_ylabel("volume multiplier")
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig


def plot_recovery(compare_df: pd.DataFrame, title: str, color_col: str):
    params = list(compare_df["parameter"].unique())
    fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 4.2), squeeze=False)
    categories = list(compare_df[color_col].unique())
    cmap = plt.get_cmap("tab10")
    colors = {cat: cmap(i % 10) for i, cat in enumerate(categories)}
    for ax, param in zip(axes.ravel(), params):
        subset = compare_df[compare_df["parameter"] == param]
        for cat in categories:
            group = subset[subset[color_col] == cat]
            ax.scatter(group["true"], group["learned"], s=28, alpha=0.75, color=colors[cat], label=cat)
        lo = min(subset["true"].min(), subset["learned"].min())
        hi = max(subset["true"].max(), subset["learned"].max())
        pad = max((hi - lo) * 0.08, 0.02)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1)
        ax.set_title(param)
        ax.set_xlabel("true")
        ax.set_ylabel("learned")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=color_col, loc="center right", frameon=False)
    fig.suptitle(title, y=1.04)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    return fig


def plot_impact_convergence(convergence_df: pd.DataFrame, level: str = "layer"):
    subset = convergence_df[convergence_df["level"] == level]
    fig, ax = plt.subplots(figsize=(12, 5))
    for category, group in subset.groupby("category"):
        ax.plot(group["epoch"], group["relative_to_first"], marker="o", label=category)
    ax.set_title(f"Impact convergence by {level}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative MSE to final")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    return fig


def plot_stats_overview(stats_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    top = stats_df.sort_values("overfitting_rel_avg", ascending=False).head(20)
    labels = top["layer"] + "/" + top["axis"] + "/" + top["parameter"]
    ax.barh(labels[::-1], top["overfitting_rel_avg"].iloc[::-1])
    ax.set_title("Top hierarchy stats by relative overfitting score")
    ax.set_xlabel("relative score")
    plt.tight_layout()
    return fig
