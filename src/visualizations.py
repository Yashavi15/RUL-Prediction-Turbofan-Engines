"""
Visualization Module for RUL Prediction and Turbofan Sensor Analytics.
Provides interactive Plotly charts for the Streamlit dashboard
and Matplotlib figures for static research reports.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SENSOR_DESCRIPTIONS, OP_SETTINGS_DESCRIPTIONS


# ==============================================================================
# Plotly Interactive Visualizations for Streamlit
# ==============================================================================

def plot_engine_sensor_trends(
    engine_df: pd.DataFrame,
    sensor_cols: List[str],
    unit_id: int
) -> go.Figure:
    """
    Plot selected sensor trends across operating cycles for an individual engine.
    """
    fig = go.Figure()
    for col in sensor_cols:
        desc = SENSOR_DESCRIPTIONS.get(col, col)
        fig.add_trace(go.Scatter(
            x=engine_df["cycle"],
            y=engine_df[col],
            mode="lines",
            name=f"{col.upper()} ({desc.split('(')[0].strip()})"
        ))
        
    fig.update_layout(
        title=f"Sensor Degradation Trajectories — Engine Unit #{unit_id}",
        xaxis_title="Operating Cycle",
        yaxis_title="Sensor Measurement Value",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_predicted_vs_actual_trajectory(
    pred_df: pd.DataFrame,
    unit_id: int,
    model_name: str = "LSTM"
) -> go.Figure:
    """
    Plot predicted RUL curve versus actual ground truth RUL across operating cycles.
    Matches wireframe in Figure 11 of the report.
    """
    fig = go.Figure()
    
    # Predicted RUL
    fig.add_trace(go.Scatter(
        x=pred_df["cycle"],
        y=pred_df["predicted_rul"],
        mode="lines+markers",
        name=f"Predicted RUL ({model_name})",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=4)
    ))
    
    # Actual RUL if available
    if "actual_rul" in pred_df.columns:
        fig.add_trace(go.Scatter(
            x=pred_df["cycle"],
            y=pred_df["actual_rul"],
            mode="lines",
            name="Ground Truth RUL",
            line=dict(color="#2ca02c", width=2.5, dash="dash")
        ))
        
    # Warning threshold lines
    fig.add_hline(y=50, line_dash="dot", line_color="#f4a261", annotation_text="Warning (50 cycles)")
    fig.add_hline(y=20, line_dash="dot", line_color="#e63946", annotation_text="Critical (20 cycles)")
    
    fig.update_layout(
        title=f"Remaining Useful Life (RUL) Trajectory — Engine Unit #{unit_id}",
        xaxis_title="Operating Cycle (Time Index)",
        yaxis_title="Remaining Useful Life (Cycles)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str]) -> go.Figure:
    """
    Interactive correlation matrix for informative sensors and operational settings.
    """
    corr = df[feature_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Sensor & Operating Setting Correlation Matrix"
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_actual_vs_predicted_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> go.Figure:
    """
    Scatter plot of Actual vs. Predicted RUL with ideal 45-degree line.
    """
    max_val = max(np.max(y_true), np.max(y_pred)) + 10
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(size=5, opacity=0.6, color="#457b9d"),
        name="Observations"
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=dict(color="#e63946", dash="dash", width=2),
        name="Ideal Fit (y = x)"
    ))
    
    fig.update_layout(
        title=f"Actual vs. Predicted RUL — {model_name}",
        xaxis_title="Ground Truth RUL (Cycles)",
        yaxis_title="Predicted RUL (Cycles)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_residuals_histogram(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> go.Figure:
    """
    Histogram of prediction residuals (y_pred - y_true) with asymmetric risk zone shading.
    """
    residuals = y_pred - y_true
    fig = px.histogram(
        residuals,
        nbins=50,
        title=f"Prediction Error Distribution (Residuals) — {model_name}",
        labels={"value": "Residual Error [Cycles] (Predicted - Actual)"},
        color_discrete_sequence=["#2a9d8f"]
    )
    fig.add_vline(x=0, line_dash="solid", line_color="#333", annotation_text="Zero Error")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Prediction Error (Predicted - Actual RUL)",
        yaxis_title="Count",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_training_loss_curves(history: Dict[str, List[float]]) -> go.Figure:
    """
    Training and Validation MSE loss progression over epochs.
    """
    fig = go.Figure()
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history.get("loss", []),
        mode="lines+markers",
        name="Training Loss (MSE)",
        line=dict(color="#1f77b4", width=2.5)
    ))
    
    if "val_loss" in history and history["val_loss"]:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history.get("val_loss", []),
            mode="lines+markers",
            name="Validation Loss (MSE)",
            line=dict(color="#ff7f0e", width=2.5)
        ))
        
    fig.update_layout(
        title="LSTM Model Convergence — Loss Progression Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Mean Squared Error (MSE)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
