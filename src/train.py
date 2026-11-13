"""
Training Pipeline for RUL Prediction Platform.
Executes end-to-end workflow:
1. Loads NASA PHM08 train.txt run-to-failure trajectories
2. Checks data quality & audits sensors
3. Generates supervised RUL targets (Linear & Piecewise capped)
4. Performs strict engine-wise train / val / test splitting (Zero data leakage)
5. Fits standardization scaler exclusively on training engines
6. Constructs sliding-window sequence datasets for LSTM
7. Trains Model A: Baseline Random Forest Regressor
8. Trains Model B: Recurrent LSTM Neural Network with Early Stopping
9. Evaluates both models with MAE, RMSE, and asymmetric PHM08 score
10. Exports trained model artifacts and performance summaries
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_WINDOW_SIZE, DEFAULT_RUL_CAP, DEFAULT_RANDOM_SEED,
    MODELS_DIR, REPORTS_DIR, TRAIN_RATIO, VAL_RATIO
)
from src.data_loader import load_train_data, audit_data_quality
from src.feature_engineering import select_feature_columns, analyze_sensor_variance
from src.preprocessing import (
    add_rul_labels, engine_wise_split, RULPreprocessor,
    create_sliding_sequences, extract_last_sequence_per_unit
)
from src.models import BaselineRegressor, LSTMRegressor, detect_runtime_environment
from src.evaluate import evaluate_predictions


def run_training_pipeline(
    window_size: int = DEFAULT_WINDOW_SIZE,
    cap_rul: int = DEFAULT_RUL_CAP,
    epochs: int = 25,
    batch_size: int = 64,
    random_seed: int = DEFAULT_RANDOM_SEED,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Run full training and evaluation pipeline according to report specifications.
    """
    total_start = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PHM08 RUL PREDICTION PLATFORM - MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Runtime Environment Detection
    env_info = detect_runtime_environment()
    print(f"[ENV] Python: {env_info['python_version']} | Engine: {env_info['description']}")
    
    # Step 2: Load Data
    print(f"[DATA] Loading run-to-failure dataset...")
    raw_df = load_train_data()
    audit = audit_data_quality(raw_df, "PHM08 Training Trajectories")
    print(f"[DATA] Loaded {audit['total_rows']:,} rows across {audit['unique_units']} engines.")
    print(f"       Cycle range per engine: {audit['min_cycle_reached']} - {audit['max_cycle_reached']} cycles (mean: {audit['avg_cycles_per_unit']:.1f}).")
    
    # Step 3: Feature Analysis & Selection
    sensor_stats = analyze_sensor_variance(raw_df)
    active_features = select_feature_columns(raw_df, include_op_settings=True)
    print(f"[FEAT] Active input features ({len(active_features)} channels):")
    print(f"       {active_features}")
    
    # Step 4: Add RUL Labels
    print(f"[RUL] Generating target labels (Piecewise linear cap: {cap_rul} cycles)...")
    labeled_df = add_rul_labels(raw_df, cap_rul=cap_rul)
    
    # Step 5: Engine-wise Data Split (Zero Leakage)
    print(f"[SPLIT] Partitioning engines into Train ({int(TRAIN_RATIO*100)}%), Val ({int(VAL_RATIO*100)}%), and Held-out Test ({int((1-TRAIN_RATIO-VAL_RATIO)*100)}%)...")
    train_df, val_df, test_df, engine_dict = engine_wise_split(
        labeled_df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_seed=random_seed
    )
    print(f"        Train: {len(engine_dict['train'])} engines ({len(train_df)} cycles)")
    print(f"        Val:   {len(engine_dict['val'])} engines ({len(val_df)} cycles)")
    print(f"        Test:  {len(engine_dict['test'])} engines ({len(test_df)} cycles)")
    
    # Step 6: Standardization (Fitted exclusively on training set)
    print(f"[PREP] Fitting StandardScaler on training engines only...")
    preprocessor = RULPreprocessor(active_features)
    train_scaled = preprocessor.fit_transform(train_df)
    val_scaled = preprocessor.transform(val_df)
    test_scaled = preprocessor.transform(test_df)
    
    # Step 7: Temporal Sequence Construction for LSTM
    print(f"[SEQ] Constructing sliding-window sequences (L = {window_size} cycles)...")
    X_train, y_train, meta_train = create_sliding_sequences(train_scaled, active_features, window_size=window_size)
    X_val, y_val, meta_val = create_sliding_sequences(val_scaled, active_features, window_size=window_size)
    X_test, y_test, meta_test = create_sliding_sequences(test_scaled, active_features, window_size=window_size)
    print(f"      Sequence shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # Step 8: Train Model A - Baseline Regressor (Random Forest)
    print("\n" + "-" * 50)
    print("[MODEL A] Training Random Forest Baseline Regressor...")
    baseline = BaselineRegressor(n_estimators=100, max_depth=16, random_state=random_seed)
    t0 = time.time()
    baseline.fit(X_train, y_train)
    t_rf = time.time() - t0
    print(f"[MODEL A] Baseline trained in {t_rf:.2f}s.")
    
    # Step 9: Train Model B - LSTM Sequence Regressor
    print("\n" + "-" * 50)
    print(f"[MODEL B] Training LSTM Sequence Regressor ({epochs} epochs max)...")
    lstm = LSTMRegressor(
        window_size=window_size,
        n_features=len(active_features),
        lstm_units=64,
        dense_units=32,
        dropout_rate=0.2,
        learning_rate=0.001,
        random_seed=random_seed
    )
    t0 = time.time()
    lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        patience=8,
        verbose=verbose
    )
    t_lstm = time.time() - t0
    print(f"[MODEL B] LSTM training finished in {t_lstm:.2f}s.")
    
    # Step 10: Evaluation on Held-Out Test Engines
    print("\n" + "=" * 70)
    print("EVALUATION ON HELD-OUT UNSEEN TEST ENGINES")
    print("=" * 70)
    
    # Cycle-level predictions across all windows
    pred_rf_cycle = baseline.predict(X_test)
    pred_lstm_cycle = lstm.predict(X_test)
    
    eval_rf_cycle = evaluate_predictions(y_test, pred_rf_cycle, model_name="Random Forest (Cycle-level)")
    eval_lstm_cycle = evaluate_predictions(y_test, pred_lstm_cycle, model_name="LSTM Network (Cycle-level)")
    
    # Trajectory End-of-Life evaluation (last available sequence for each test engine)
    X_test_last, meta_test_last = extract_last_sequence_per_unit(test_scaled, active_features, window_size=window_size)
    # Ground truth RUL at the end of trajectory for run-to-failure test engines is 0
    # Or calculate from actual labeled RUL at that last cycle:
    last_ruls = []
    for row in meta_test_last:
        u, c = row[0], row[1]
        actual_rul = test_df[(test_df["unit"] == u) & (test_df["cycle"] == c)]["RUL"].values[0]
        last_ruls.append(actual_rul)
    y_test_last = np.array(last_ruls, dtype=np.float32)
    
    pred_rf_last = baseline.predict(X_test_last)
    pred_lstm_last = lstm.predict(X_test_last)
    
    eval_rf_last = evaluate_predictions(y_test_last, pred_rf_last, model_name="Random Forest (Trajectory-End)")
    eval_lstm_last = evaluate_predictions(y_test_last, pred_lstm_last, model_name="LSTM Network (Trajectory-End)")
    
    print("\nCycle-Level Evaluation (All consecutive cycles across test engines):")
    print(f"  Random Forest -> MAE: {eval_rf_cycle['mae']:.2f} | RMSE: {eval_rf_cycle['rmse']:.2f} | PHM08 Score: {eval_rf_cycle['phm08_score']:,.1f} | R2: {eval_rf_cycle['r2_score']:.3f}")
    print(f"  LSTM Network  -> MAE: {eval_lstm_cycle['mae']:.2f} | RMSE: {eval_lstm_cycle['rmse']:.2f} | PHM08 Score: {eval_lstm_cycle['phm08_score']:,.1f} | R2: {eval_lstm_cycle['r2_score']:.3f}")
    
    print("\nTrajectory-End Evaluation (Final inspection point for each test engine):")
    print(f"  Random Forest -> MAE: {eval_rf_last['mae']:.2f} | RMSE: {eval_rf_last['rmse']:.2f} | PHM08 Score: {eval_rf_last['phm08_score']:,.1f}")
    print(f"  LSTM Network  -> MAE: {eval_lstm_last['mae']:.2f} | RMSE: {eval_lstm_last['rmse']:.2f} | PHM08 Score: {eval_lstm_last['phm08_score']:,.1f}")
    
    # Step 11: Save Artifacts
    print("\n[EXPORT] Saving trained models and preprocessing artifacts to 'models/'...")
    preprocessor.save(MODELS_DIR / "scaler.joblib")
    baseline.save(MODELS_DIR / "baseline_rf.joblib")
    lstm.save(MODELS_DIR / "lstm_model.joblib")
    
    meta_info = {
        "active_features": active_features,
        "window_size": window_size,
        "cap_rul": cap_rul,
        "random_seed": random_seed,
        "train_units": engine_dict["train"],
        "val_units": engine_dict["val"],
        "test_units": engine_dict["test"],
        "environment": env_info,
        "training_time_seconds": round(time.time() - total_start, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(meta_info, f, indent=2)
        
    metrics_summary = {
        "cycle_level": {
            "baseline": eval_rf_cycle,
            "lstm": eval_lstm_cycle
        },
        "trajectory_end": {
            "baseline": eval_rf_last,
            "lstm": eval_lstm_last
        },
        "training_history": lstm.history
    }
    with open(MODELS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"[SUCCESS] All artifacts successfully saved in '{MODELS_DIR.resolve()}'.")
    print(f"[DONE] Total pipeline duration: {time.time() - total_start:.2f}s.")
    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RUL Prediction Models")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE, help="Sliding window length L")
    parser.add_argument("--cap-rul", type=int, default=DEFAULT_RUL_CAP, help="Piecewise RUL maximum cap")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs for LSTM")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()
    
    run_training_pipeline(
        window_size=args.window,
        cap_rul=args.cap_rul,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
        verbose=args.verbose
    )
