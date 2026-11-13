"""
Inference and Prediction Module for RUL Prediction Platform.
Loads trained artifacts to generate predictions for single engines,
batch trajectories, and the unlabeled challenge test set.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, DEFAULT_WINDOW_SIZE
from src.preprocessing import RULPreprocessor, create_sliding_sequences, extract_last_sequence_per_unit
from src.models import BaselineRegressor, LSTMRegressor


class RULPredictor:
    """
    Unified predictor supporting both Baseline and LSTM models.
    """
    def __init__(self, models_dir: Union[str, Path] = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.metadata: Dict[str, Any] = {}
        self.preprocessor: Optional[RULPreprocessor] = None
        self.baseline_model: Optional[BaselineRegressor] = None
        self.lstm_model: Optional[LSTMRegressor] = None
        self.is_loaded = False
        
        self.load_artifacts()

    def load_artifacts(self):
        meta_file = self.models_dir / "metadata.json"
        scaler_file = self.models_dir / "scaler.joblib"
        baseline_file = self.models_dir / "baseline_rf.joblib"
        lstm_file = self.models_dir / "lstm_model.joblib"
        
        if not meta_file.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in '{self.models_dir}'. Please run 'python -m src.train' first."
            )
            
        with open(meta_file, "r") as f:
            self.metadata = json.load(f)
            
        self.preprocessor = RULPreprocessor.load(scaler_file)
        self.baseline_model = BaselineRegressor.load(baseline_file)
        self.lstm_model = LSTMRegressor.load(lstm_file)
        self.is_loaded = True

    @property
    def window_size(self) -> int:
        return self.metadata.get("window_size", DEFAULT_WINDOW_SIZE)

    @property
    def feature_cols(self) -> List[str]:
        return self.metadata.get("active_features", [])

    def predict_engine_trajectory(
        self,
        engine_df: pd.DataFrame,
        model_type: str = "lstm"
    ) -> pd.DataFrame:
        """
        Generate cycle-by-cycle RUL predictions across an engine's operational history.
        
        Args:
            engine_df: DataFrame containing sensor readings for a single engine.
            model_type: 'lstm' or 'baseline'.
            
        Returns:
            DataFrame with 'cycle', 'predicted_rul', and if present 'actual_rul'.
        """
        if not self.is_loaded:
            raise RuntimeError("Predictor artifacts not loaded.")
            
        # Scale features using training-fitted scaler
        scaled_df = self.preprocessor.transform(engine_df)
        X, y, meta = create_sliding_sequences(
            scaled_df, self.feature_cols, window_size=self.window_size
        )
        
        model = self.lstm_model if model_type.lower() == "lstm" else self.baseline_model
        preds = model.predict(X)
        
        result = pd.DataFrame({
            "cycle": meta[:, 1],
            "predicted_rul": np.round(preds, 1)
        })
        
        if len(y) > 0 and "RUL" in engine_df.columns:
            result["actual_rul"] = y
            result["error"] = result["predicted_rul"] - result["actual_rul"]
            
        return result

    def predict_engine_last_cycle(
        self,
        engine_df: pd.DataFrame,
        model_type: str = "lstm"
    ) -> Dict[str, Any]:
        """
        Estimate current Remaining Useful Life at the engine's latest observed cycle.
        """
        traj = self.predict_engine_trajectory(engine_df, model_type=model_type)
        if traj.empty:
            return {"error": "Insufficient cycle history"}
            
        last_row = traj.iloc[-1]
        pred_rul = float(last_row["predicted_rul"])
        
        # Risk assessment classification
        if pred_rul <= 25:
            risk_level = "CRITICAL (Immediate Maintenance Required)"
            status_color = "#e63946"
        elif pred_rul <= 50:
            risk_level = "WARNING (Schedule Inspection)"
            status_color = "#f4a261"
        else:
            risk_level = "NORMAL (Safe Operational State)"
            status_color = "#2a9d8f"
            
        res = {
            "last_cycle": int(last_row["cycle"]),
            "predicted_rul": round(pred_rul, 1),
            "risk_level": risk_level,
            "status_color": status_color,
            "model_used": "LSTM Neural Network" if model_type.lower() == "lstm" else "Random Forest Baseline"
        }
        if "actual_rul" in last_row:
            res["actual_rul"] = float(last_row["actual_rul"])
            res["error"] = round(float(last_row["error"]), 1)
            
        return res

    def batch_predict_test_set(
        self,
        test_df: pd.DataFrame,
        model_type: str = "lstm"
    ) -> pd.DataFrame:
        """
        Generate end-of-trajectory predictions for all engines in test.txt.
        """
        scaled_df = self.preprocessor.transform(test_df)
        X_last, meta_last = extract_last_sequence_per_unit(
            scaled_df, self.feature_cols, window_size=self.window_size
        )
        model = self.lstm_model if model_type.lower() == "lstm" else self.baseline_model
        preds = model.predict(X_last)
        
        results = []
        for i, row in enumerate(meta_last):
            unit_id, last_cycle = int(row[0]), int(row[1])
            rul_est = float(np.round(preds[i], 1))
            
            if rul_est <= 25:
                status = "CRITICAL"
            elif rul_est <= 50:
                status = "WARNING"
            else:
                status = "NORMAL"
                
            results.append({
                "unit": unit_id,
                "cutoff_cycle": last_cycle,
                "predicted_rul": rul_est,
                "health_status": status
            })
            
        return pd.DataFrame(results)
