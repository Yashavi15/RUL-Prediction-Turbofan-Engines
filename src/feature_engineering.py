"""
Feature Engineering and Sensor Selection Module for PHM08 turbofan data.
Computes sensor variability, detects low-variance channels, and constructs features.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SENSOR_COLS, OP_SETTINGS_COLS


def analyze_sensor_variance(df: pd.DataFrame, threshold: float = 1e-4) -> Dict[str, Any]:
    """
    Calculate standard deviations and identify constant or near-constant sensors.
    
    Args:
        df: Training DataFrame
        threshold: Standard deviation cutoff for constant sensors
        
    Returns:
        Dict with standard deviations, constant sensor list, and informative sensor list.
    """
    stds = df[SENSOR_COLS].std()
    constant_sensors = stds[stds < threshold].index.tolist()
    informative_sensors = [s for s in SENSOR_COLS if s not in constant_sensors]
    
    return {
        "sensor_stds": stds.to_dict(),
        "constant_sensors": constant_sensors,
        "informative_sensors": informative_sensors,
        "n_informative": len(informative_sensors)
    }


def select_feature_columns(
    df: pd.DataFrame, 
    include_op_settings: bool = True,
    var_threshold: float = 1e-4
) -> List[str]:
    """
    Return the list of active features (operational settings + informative sensors).
    Operational settings are retained so models can distinguish operating regime from wear.
    """
    analysis = analyze_sensor_variance(df, threshold=var_threshold)
    features = []
    if include_op_settings:
        features.extend(OP_SETTINGS_COLS)
    features.extend(analysis["informative_sensors"])
    return features


def compute_rolling_features(
    df: pd.DataFrame, 
    features: List[str], 
    window: int = 5
) -> pd.DataFrame:
    """
    Extract rolling statistical features (mean, std) per engine unit.
    Useful for tabular baseline regressors (Random Forest).
    """
    df_out = df.copy()
    grouped = df_out.groupby("unit")
    
    for feat in features:
        # Rolling mean and standard deviation within the engine's trajectory
        df_out[f"{feat}_roll_mean"] = grouped[feat].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df_out[f"{feat}_roll_std"] = grouped[feat].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0.0)
        )
        
    return df_out
