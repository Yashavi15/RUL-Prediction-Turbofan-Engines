"""
Evaluation Metrics Module for RUL Prediction.
Implements:
1. Mean Absolute Error (MAE) - Equation 11
2. Root Mean Squared Error (RMSE) - Equation 12
3. Asymmetric PHM08 Challenge Score - Equation 13
4. R-squared (R2) and Error distribution statistics
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the asymmetric PHM08 score defined in Equation (13) of the report:
    
        d_i = y_pred_i - y_true_i
        S = sum_{d_i < 0} (exp(-d_i / 13) - 1) + sum_{d_i >= 0} (exp(d_i / 10) - 1)
        
    Penalty is asymmetric: late predictions (d_i > 0) are penalized more severely
    than early predictions (d_i < 0) because unexpected in-service failures
    carry high operational risk compared to slightly early maintenance.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    d = y_pred - y_true
    
    # Clip large positive d to avoid floating point overflow during evaluation
    # e.g., d = 300 cycles late would yield exp(30) which is 1e13
    d_clipped = np.clip(d, -100.0, 70.0)
    
    early_mask = d < 0
    late_mask = d >= 0
    
    score_early = np.sum(np.exp(-d_clipped[early_mask] / 13.0) - 1.0)
    score_late = np.sum(np.exp(d_clipped[late_mask] / 10.0) - 1.0)
    
    total_score = float(score_early + score_late)
    return total_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """
    Comprehensive regression evaluation against ground truth RUL.
    
    Returns:
        Dict with MAE, RMSE, PHM08 Score, R2, Bias, and error distribution.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if len(y_true) == 0:
        return {"error": "Empty evaluation set"}
        
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    phm_score = calculate_phm08_score(y_true, y_pred)
    
    errors = y_pred - y_true
    early_pct = float(np.mean(errors < 0) * 100.0)
    late_pct = float(np.mean(errors >= 0) * 100.0)
    
    return {
        "model_name": model_name,
        "n_samples": len(y_true),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "phm08_score": round(phm_score, 2),
        "r2_score": round(r2, 4),
        "mean_error_bias": round(float(np.mean(errors)), 3),
        "error_std": round(float(np.std(errors)), 3),
        "early_prediction_pct": round(early_pct, 1),
        "late_prediction_pct": round(late_pct, 1)
    }
