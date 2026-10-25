"""Plotly figures and feature-map / embedding helpers."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from .dataset import CLASSES
from .experiment import CONDITIONS, EVAL_LABELS

COLORS = {"A_baseline": "#6b7280", "B_bg_random": "#2563eb", "C_color_aug": "#d97706", "D_hue_aug": "#7c3aed"}
SHORT = {"A_baseline": "A Baseline", "B_bg_random": "B BG random", "C_color_aug": "C Colour aug", "D_hue_aug": "D Hue aug"}
_LAYOUT = dict(template="plotly_white", font=dict(size=13), margin=dict(l=40, r=20, t=50, b=40),
               legend=dict(orientation="h", y=-0.2))


def accuracy_bars(summary: Dict[str, dict], sets=("id", "shifted", "neutral")) -> go.Figure:
    fig = go.Figure()
    for cond, s in summary.items():
        fig.add_bar(
            name=SHORT.get(cond, cond), x=[_set_label(k) for k in sets],
            y=[s[f"{k}_acc_mean"] for k in sets],
            error_y=dict(type="data", array=[s[f"{k}_acc_std"] for k in sets], visible=s["n_seeds"] > 1),
            marker_color=COLORS.get(cond), text=[f"{s[f'{k}_acc_mean']:.2f}" for k in sets], textposition="outside")
    fig.add_hline(y=1 / 3, line_dash="dot", line_color="#9ca3af", annotation_text="chance")
    fig.update_layout(title="Accuracy by test set", yaxis=dict(range=[0, 1.1], title="accuracy"), barmode="group", **_LAYOUT)
    return fig


def _set_label(k: str) -> str:
    return {"id": "In-distribution", "shifted": "Shifted (reversed bg)", "neutral": "Neutral bg"}[k]


def gap_bars(summary: Dict[str, dict]) -> go.Figure:
    conds = list(summary)
    fig = go.Figure(go.Bar(
        x=[SHORT.get(c, c) for c in conds], y=[summary[c]["gap_mean"] for c in conds],
        error_y=dict(type="data", array=[summary[c]["gap_std"] for c in conds],
                     visible=any(summary[c]["n_seeds"] > 1 for c in conds)),
        marker_color=[COLORS.get(c) for c in conds],
        text=[f"{summary[c]['gap_mean']:+.2f}" for c in conds], textposition="outside"))
    fig.update_layout(title="Robustness gap (ID acc − shifted acc; lower = more robust)",
                      yaxis_title="gap", **_LAYOUT)
    return fig


def follow_bars(summary: Dict[str, dict]) -> go.Figure:
    conds = list(summary)
    fig = go.Figure(go.Bar(
        x=[SHORT.get(c, c) for c in conds], y=[summary[c]["bg_follow_mean"] for c in conds],
        marker_color=[COLORS.get(c) for c in conds],
        text=[f"{summary[c]['bg_follow_mean']:.2f}" for c in conds], textposition="outside"))
    fig.update_layout(title="Shortcut-follow rate on shifted set (prediction = background-implied class)",
                      yaxis=dict(range=[0, 1.1], title="rate"), **_LAYOUT)
    return fig


def confusion_figure(cm, title: str) -> go.Figure:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(
        z=norm, x=CLASSES, y=CLASSES, zmin=0, zmax=1, colorscale="Blues",
        text=cm, texttemplate="%{text}", hovertemplate="true %{y} → pred %{x}: %{text}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="predicted", yaxis=dict(title="true", autorange="reversed"),
                      height=330, margin=dict(l=40, r=10, t=50, b=40), template="plotly_white", font=dict(size=12))
    return fig


def learning_curves(runs: List[dict]) -> go.Figure:
    fig = go.Figure()
    for r in runs:
        h = r["history"]
        fig.add_scatter(x=[e["epoch"] for e in h], y=[e["val_acc"] for e in h], mode="lines+markers",
                        name=f"{SHORT.get(r['condition'])} s{r['seed']}", line=dict(color=COLORS.get(r["condition"])))
    fig.update_layout(title="Validation accuracy per epoch (condition's own training distribution)",
                      xaxis_title="epoch", yaxis=dict(range=[0, 1.05], title="val accuracy"), **_LAYOUT)
    return fig


def image_hwc(x: np.ndarray) -> np.ndarray:
    return (np.transpose(x, (1, 2, 0)) * 255).astype(np.uint8)


def image_grid(xs: np.ndarray, captions: List[str], ncols: int = 6) -> go.Figure:
    n = len(xs)
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=captions, vertical_spacing=0.12 if nrows > 1 else 0.1)
    for i, x in enumerate(xs):
        fig.add_trace(go.Image(z=image_hwc(x)), row=i // ncols + 1, col=i % ncols + 1)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_annotations(font_size=11)
    fig.update_layout(height=170 * nrows + 40, margin=dict(l=5, r=5, t=40, b=5))
    return fig


def feature_map_figure(model, x: np.ndarray, layer: str, max_maps: int = 16) -> go.Figure:
    """Show the input and the first ``max_maps`` channels (highest mean activation first)."""
    maps = model.feature_maps(torch.from_numpy(x[None]), layer)[0].numpy()
    order = np.argsort(-maps.mean(axis=(1, 2)))[:max_maps]
    ncols = 6
    n = len(order) + 1
    nrows = int(np.ceil(n / ncols))
    titles = ["input"] + [f"ch {c}" for c in order]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, vertical_spacing=0.1)
    fig.add_trace(go.Image(z=image_hwc(x)), row=1, col=1)
    for k, c in enumerate(order, start=1):
        m = maps[c]
        rng = m.max() - m.min()
        m = (m - m.min()) / rng if rng > 1e-8 else np.zeros_like(m)
        fig.add_trace(go.Heatmap(z=m, colorscale="Viridis", showscale=False),
                      row=k // ncols + 1, col=k % ncols + 1)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    fig.update_annotations(font_size=11)
    fig.update_layout(height=160 * nrows + 40, margin=dict(l=5, r=5, t=40, b=5))
    return fig


def pca_2d(features: np.ndarray) -> np.ndarray:
    f = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(f, full_matrices=False)
    return f @ vt[:2].T


def embedding_figure(model, x: np.ndarray, y: np.ndarray, bg: np.ndarray, color_by: str = "true shape") -> go.Figure:
    with torch.no_grad():
        feats = model.embed(torch.from_numpy(x)).numpy()
    p = pca_2d(feats)
    lab = y if color_by == "true shape" else bg
    fig = px.scatter(x=p[:, 0], y=p[:, 1], color=[CLASSES[i] for i in lab],
                     color_discrete_map={"circle": "#dc2626", "square": "#16a34a", "triangle": "#2563eb"},
                     labels={"x": "PC1", "y": "PC2", "color": color_by}, opacity=0.75)
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(title=f"PCA of penultimate features, coloured by {color_by}", height=420, template="plotly_white")
    return fig
