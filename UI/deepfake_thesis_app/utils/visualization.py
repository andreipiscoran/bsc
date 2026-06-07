from __future__ import annotations

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.image_processing import normalize_map


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = (255 * normalize_map(heatmap)).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image, 1 - alpha, colored_rgb, alpha, 0)
    return blended


def plot_color_statistics(means: list[float], stds: list[float]) -> go.Figure:
    fig = go.Figure()
    channels = ["R", "G", "B"]
    fig.add_trace(go.Bar(name="Mean", x=channels, y=means))
    fig.add_trace(go.Bar(name="Std", x=channels, y=stds))
    fig.update_layout(barmode="group", title="RGB Channel Statistics", height=360)
    return fig


def plot_correlation_heatmap(correlation: np.ndarray) -> go.Figure:
    fig = px.imshow(
        correlation,
        x=["R", "G", "B"],
        y=["R", "G", "B"],
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Channel Correlation Matrix",
    )
    fig.update_layout(height=360)
    return fig


def plot_intensity_histogram(hist: np.ndarray, bins: np.ndarray) -> go.Figure:
    x = bins[:-1]
    fig = go.Figure(go.Scatter(x=x, y=hist, mode="lines", name="Intensity PDF"))
    fig.update_layout(
        title="Intensity Distribution",
        xaxis_title="Intensity",
        yaxis_title="Density",
        height=360,
    )
    return fig
