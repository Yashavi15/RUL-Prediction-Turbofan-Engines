"""
Data Preprocessing Module for RUL Prediction.
Implements:
1. Supervised RUL label generation (Linear & Piecewise capped)
2. Strict Engine-wise Train/Validation/Test splitting (Zero data leakage)
3. StandardScaler fitted exclusively on training engines
4. Sliding-window sequence construction for LSTM (input shape: [N, window, features])
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_WINDOW_SIZE, DEFAULT_RUL_CAP, DEFAULT_RANDOM_SEED, TRAIN_RATIO, VAL_RATIO


def add_rul_labels(
    df: pd.DataFrame, 
    cap_rul: Optional[int] = DEFAULT_RUL_CAP
) -> pd.DataFrame:
    """
    Generate ground-truth RUL target for run-to-failure engine trajectories.
    RUL(t) = Tf - t  (Equation 1 in report).
    
    Args:
        df: DataFrame containing 'unit' and 'cycle' columns
        cap_rul: If provided, clips RUL at a maximum constant threshold early in life.
                 If None or <= 0, linear unconstrained RUL is used.
    
    Returns:
        DataFrame with added 'RUL' and optionally 'RUL_raw' column.
    """
    df_out = df.copy()
    max_cycles = df_out.groupby("unit")["cycle"].transform("max")
    df_out["RUL_raw"] = max_cycles - df_out["cycle"]
    
    if cap_rul and cap_rul > 0:
        df_out["RUL"] = df_out["RUL_raw"].clip(upper=cap_rul)
    else:
        df_out["RUL"] = df_out["RUL_raw"]
        
    return df_out


def engine_wise_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[int]]]:
    """
    Split dataset strictly at the engine level to prevent temporal data leakage.
    No engine trajectory will ever appear in more than one split (Figure 5).
    
    Returns:
        (train_df, val_df, test_df, engine_dict)
    """
    unique_units = df["unit"].unique()
    rng = np.random.RandomState(random_seed)
    shuffled_units = rng.permutation(unique_units)
    
    n_total = len(shuffled_units)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_units = shuffled_units[:n_train].tolist()
    val_units = shuffled_units[n_train:n_train + n_val].tolist()
    test_units = shuffled_units[n_train + n_val:].tolist()
    
    train_df = df[df["unit"].isin(train_units)].copy().reset_index(drop=True)
    val_df = df[df["unit"].isin(val_units)].copy().reset_index(drop=True)
    test_df = df[df["unit"].isin(test_units)].copy().reset_index(drop=True)
    
    engine_dict = {
        "train": train_units,
        "val": val_units,
        "test": test_units
    }
    
    return train_df, val_df, test_df, engine_dict


class RULPreprocessor:
    """
    Standardization scaler and sequence generator.
    Ensures scaler is fitted exclusively on training engines to prevent data leakage.
    """
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = list(feature_cols)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "RULPreprocessor":
        """Fit scaler on training data only."""
        self.scaler.fit(train_df[self.feature_cols].values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted standardization to a DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        
        df_out = df.copy()
        scaled_vals = self.scaler.transform(df_out[self.feature_cols].values)
        
        # Add scaled feature columns
        for idx, col in enumerate(self.feature_cols):
            df_out[col] = scaled_vals[:, idx]
            
        return df_out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)

    def save(self, filepath: Union[str, Path]):
        payload = {
            "feature_cols": self.feature_cols,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RULPreprocessor":
        payload = joblib.load(filepath)
        obj = cls(payload["feature_cols"])
        obj.scaler = payload["scaler"]
        obj.is_fitted = payload["is_fitted"]
        return obj


def create_sliding_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    window_size: int = DEFAULT_WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct sliding-window sequences for LSTM input (Listing 3 & Figure 6 in report).
    
    Each sample is a 2D matrix of shape (window_size, n_features) representing
    consecutive cycles [t - window_size + 1, ..., t], with target RUL(t).
    
    Returns:
        X: np.ndarray of shape (N, window_size, n_features)
        y: np.ndarray of shape (N,)
        meta: np.ndarray of shape (N, 2) storing [unit, cycle] for tracking
    """
    X_list = []
    y_list = []
    meta_list = []
    
    for unit_id, group in df.groupby("unit"):
        group_sorted = group.sort_values("cycle").reset_index(drop=True)
        feats = group_sorted[feature_cols].values
        targets = group_sorted[target_col].values if target_col in group_sorted else None
        cycles = group_sorted["cycle"].values
        n_rows = len(group_sorted)
        
        if n_rows < window_size:
            # Pad short sequences by repeating the initial state
            pad_len = window_size - n_rows
            padding = np.repeat(feats[:1], pad_len, axis=0)
            padded_feats = np.vstack([padding, feats])
            X_list.append(padded_feats)
            if targets is not None:
                y_list.append(targets[-1])
            meta_list.append([unit_id, cycles[-1]])
            continue
            
        for i in range(window_size, n_rows + 1):
            seq = feats[i - window_size:i]
            X_list.append(seq)
            if targets is not None:
                y_list.append(targets[i - 1])
            meta_list.append([unit_id, cycles[i - 1]])
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if len(y_list) > 0 else np.array([], dtype=np.float32)
    meta = np.array(meta_list, dtype=np.int32)
    
    return X, y, meta


def extract_last_sequence_per_unit(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = DEFAULT_WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the most recent observation window for each engine trajectory.
    Used for trajectory-level inference on test sets (Listing 3 / Section 4.7).
    
    Returns:
        X_last: (N_units, window_size, n_features)
        meta_last: (N_units, 2) [unit, last_cycle]
    """
    X_list = []
    meta_list = []
    
    for unit_id, group in df.groupby("unit"):
        group_sorted = group.sort_values("cycle").reset_index(drop=True)
        feats = group_sorted[feature_cols].values
        last_cycle = group_sorted["cycle"].iloc[-1]
        n_rows = len(group_sorted)
        
        if n_rows < window_size:
            pad_len = window_size - n_rows
            padding = np.repeat(feats[:1], pad_len, axis=0)
            seq = np.vstack([padding, feats])
        else:
            seq = feats[-window_size:]
            
        X_list.append(seq)
        meta_list.append([unit_id, last_cycle])
        
    return np.array(X_list, dtype=np.float32), np.array(meta_list, dtype=np.int32)
