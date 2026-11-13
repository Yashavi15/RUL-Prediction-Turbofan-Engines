"""
Model Architecture Module for RUL Prediction.
Implements:
1. Baseline Regression: Scikit-Learn Random Forest Regressor
2. Sequence Deep Learning: LSTM Neural Network (Listing 4 & Equations 3-8 in report)
   - Automatically utilizes TensorFlow/Keras if installed (e.g. Google Colab / Py3.11)
   - Automatically falls back to high-performance vectorized NumPy LSTM engine
     on environments without external neural network dependencies
   - Environment detection: reports runtime configuration and execution readiness
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# Check for TensorFlow availability
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def detect_runtime_environment() -> Dict[str, Any]:
    """
    Detect available runtime environment and execution engine.
    """
    info = {
        "python_version": sys.version.split()[0],
        "has_tensorflow": HAS_TENSORFLOW,
        "device": "Standard",
        "description": "Optimized Sequence Execution Engine"
    }
    
    if HAS_TENSORFLOW:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["device"] = "Accelerated Engine"
            info["description"] = f"Detected {len(gpus)} Accelerator Device(s)"
        else:
            info["device"] = "TensorFlow Engine"
            info["description"] = "TensorFlow Keras Execution Engine"
    return info




# ==============================================================================
# Model A: Baseline Regression Model
# ==============================================================================

class BaselineRegressor:
    """
    Conventional Regression Baseline using Random Forest (Section 4 of report).
    Provides a solid, interpretable benchmark to evaluate LSTM sequence benefits.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 18,
        min_samples_split: int = 5,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Flatten 3D sequence array (N, window, F) or return 2D array."""
        if X.ndim == 3:
            N, W, F = X.shape
            return X.reshape(N, W * F)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "BaselineRegressor":
        X_flat = self._prepare_features(X)
        if feature_names:
            self.feature_names = feature_names
        self.model.fit(X_flat, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Baseline model has not been trained.")
        X_flat = self._prepare_features(X)
        preds = self.model.predict(X_flat)
        return np.clip(preds, 0, None)  # RUL is strictly non-negative

    def get_feature_importances(self) -> np.ndarray:
        if not self.is_fitted:
            return np.array([])
        return self.model.feature_importances_

    def save(self, filepath: Union[str, Path]):
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state
            }
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaselineRegressor":
        payload = joblib.load(filepath)
        obj = cls(**payload.get("params", {}))
        obj.model = payload["model"]
        obj.feature_names = payload.get("feature_names", [])
        obj.is_fitted = True
        return obj


# ==============================================================================
# Model B: LSTM Sequence Regressor (Keras & Pure Vectorized NumPy Fallback)
# ==============================================================================

class NumpyLSTM:
    """
    High-performance, lightweight vectorized LSTM sequence regression engine.
    Strictly follows Equations (3)-(8) from Section 2.3.6 & Listing 4 of the report:
      f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)
      i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)
      c~_t = tanh(W_c [h_{t-1}, x_t] + b_c)
      c_t = f_t * c_{t-1} + i_t * c~_t
      o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)
      h_t = o_t * tanh(c_t)
    Followed by Dropout, Dense(32, ReLU), and Dense(1, Linear).
    Runs seamlessly with zero framework lock-in.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dense_dim: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        random_seed: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(random_seed)
        
        # Xavier/Glorot weight initialization
        concat_dim = hidden_dim + input_dim
        scale_lstm = np.sqrt(2.0 / (concat_dim + hidden_dim))
        
        # 4 gates concatenated: [forget, input, candidate, output] (4 * hidden_dim, concat_dim)
        self.W = self.rng.randn(4 * hidden_dim, concat_dim).astype(np.float32) * scale_lstm
        self.b = np.zeros((4 * hidden_dim, 1), dtype=np.float32)
        # Forget gate bias initialized to 1.0 for better gradient flow
        self.b[:hidden_dim] = 1.0
        
        # Dense layer 1: (dense_dim, hidden_dim)
        scale_d1 = np.sqrt(2.0 / (hidden_dim + dense_dim))
        self.W_d1 = self.rng.randn(dense_dim, hidden_dim).astype(np.float32) * scale_d1
        self.b_d1 = np.zeros((dense_dim, 1), dtype=np.float32)
        
        # Output dense: (1, dense_dim)
        scale_d2 = np.sqrt(2.0 / (dense_dim + 1))
        self.W_out = self.rng.randn(1, dense_dim).astype(np.float32) * scale_d2
        self.b_out = np.zeros((1, 1), dtype=np.float32)
        
        # Adam optimizer state variables
        self._init_adam()

    def _init_adam(self):
        self.m = {
            "W": np.zeros_like(self.W), "b": np.zeros_like(self.b),
            "W_d1": np.zeros_like(self.W_d1), "b_d1": np.zeros_like(self.b_d1),
            "W_out": np.zeros_like(self.W_out), "b_out": np.zeros_like(self.b_out)
        }
        self.v = {
            "W": np.zeros_like(self.W), "b": np.zeros_like(self.b),
            "W_d1": np.zeros_like(self.W_d1), "b_d1": np.zeros_like(self.b_d1),
            "W_out": np.zeros_like(self.W_out), "b_out": np.zeros_like(self.b_out)
        }
        self.t = 0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def forward_batch(self, X: np.ndarray, training: bool = False):
        """
        Forward pass for a batch of sequences X of shape (B, T, F).
        Returns y_pred of shape (B, 1) and cache for backprop.
        """
        B, T, F = X.shape
        H = self.hidden_dim
        
        h = np.zeros((B, H), dtype=np.float32)
        c = np.zeros((B, H), dtype=np.float32)
        
        cache_steps = []
        
        for t in range(T):
            xt = X[:, t, :]  # (B, F)
            # Concatenate [h_{t-1}, x_t] -> (B, H + F)
            concat = np.hstack([h, xt])  # (B, H + F)
            
            # Affine transformation: (4H, H+F) @ (H+F, B) + (4H, 1) -> (4H, B)
            gates_raw = (self.W @ concat.T) + self.b  # (4H, B)
            gates_raw = gates_raw.T  # (B, 4H)
            
            # Split into 4 gates
            f_raw = gates_raw[:, 0:H]
            i_raw = gates_raw[:, H:2*H]
            c_bar_raw = gates_raw[:, 2*H:3*H]
            o_raw = gates_raw[:, 3*H:4*H]
            
            f_t = self._sigmoid(f_raw)
            i_t = self._sigmoid(i_raw)
            c_bar_t = np.tanh(c_bar_raw)
            o_t = self._sigmoid(o_raw)
            
            c_prev = c
            h_prev = h
            
            # Cell state update
            c = f_t * c_prev + i_t * c_bar_t
            # Hidden state update
            tanh_c = np.tanh(c)
            h = o_t * tanh_c
            
            if training:
                cache_steps.append((concat, h_prev, c_prev, f_t, i_t, c_bar_t, o_t, tanh_c))
                
        # Top Dense Layers
        h_last = h  # (B, H)
        
        # Dropout during training
        drop_mask = np.ones_like(h_last)
        if training and self.dropout_rate > 0.0:
            drop_mask = (self.rng.rand(*h_last.shape) >= self.dropout_rate).astype(np.float32) / (1.0 - self.dropout_rate)
            h_last = h_last * drop_mask
            
        # Dense 1 (ReLU)
        d1_raw = (h_last @ self.W_d1.T) + self.b_d1.T  # (B, dense_dim)
        d1_act = np.maximum(0.0, d1_raw)
        
        # Dense Output (Linear)
        out = (d1_act @ self.W_out.T) + self.b_out.T  # (B, 1)
        
        cache = {
            "cache_steps": cache_steps,
            "h_last_raw": h,
            "drop_mask": drop_mask,
            "d1_raw": d1_raw,
            "d1_act": d1_act,
            "X": X
        }
        return out, cache

    def backward_batch(self, y_true: np.ndarray, y_pred: np.ndarray, cache: dict):
        """Vectorized gradient backpropagation and parameter update."""
        B = len(y_true)
        H = self.hidden_dim
        
        # Loss: MSE = (1/B) * sum (y_pred - y_true)^2
        # dL/dy_pred = (2/B) * (y_pred - y_true)
        grad_out = (2.0 / B) * (y_pred - y_true.reshape(-1, 1))  # (B, 1)
        
        d1_act = cache["d1_act"]
        dW_out = grad_out.T @ d1_act  # (1, dense_dim)
        db_out = np.sum(grad_out, axis=0, keepdims=True).T  # (1, 1)
        
        # Backprop through Dense 1
        grad_d1 = grad_out @ self.W_out  # (B, dense_dim)
        grad_d1_raw = grad_d1 * (cache["d1_raw"] > 0)  # ReLU grad
        
        dW_d1 = grad_d1_raw.T @ (cache["h_last_raw"] * cache["drop_mask"])  # (dense_dim, H)
        db_d1 = np.sum(grad_d1_raw, axis=0, keepdims=True).T  # (dense_dim, 1)
        
        grad_h = (grad_d1_raw @ self.W_d1) * cache["drop_mask"]  # (B, H)
        grad_c = np.zeros((B, H), dtype=np.float32)
        
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        
        # BPTT through time steps
        cache_steps = cache["cache_steps"]
        T = len(cache_steps)
        
        for t in reversed(range(T)):
            concat, h_prev, c_prev, f_t, i_t, c_bar_t, o_t, tanh_c = cache_steps[t]
            
            # Hidden state grads
            do = grad_h * tanh_c  # (B, H)
            grad_c += grad_h * o_t * (1.0 - tanh_c ** 2)
            
            df = grad_c * c_prev
            di = grad_c * c_bar_t
            dc_bar = grad_c * i_t
            
            # Gate pre-activation grads
            df_raw = df * f_t * (1.0 - f_t)
            di_raw = di * i_t * (1.0 - i_t)
            dc_bar_raw = dc_bar * (1.0 - c_bar_t ** 2)
            do_raw = do * o_t * (1.0 - o_t)
            
            dgates_raw = np.hstack([df_raw, di_raw, dc_bar_raw, do_raw])  # (B, 4H)
            
            dW += dgates_raw.T @ concat  # (4H, H + F)
            db += np.sum(dgates_raw, axis=0, keepdims=True).T
            
            # Gradient flowing to previous time step
            W_h = self.W[:, :H]  # (4H, H)
            grad_h = dgates_raw @ W_h  # (B, H)
            grad_c = grad_c * f_t      # (B, H)
            
        # Gradient clipping to prevent exploding gradients
        for grad in [dW, db, dW_d1, db_d1, dW_out, db_out]:
            np.clip(grad, -5.0, 5.0, out=grad)
            
        # Adam parameter update
        self.t += 1
        lr = self.learning_rate
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        grads = {
            "W": dW, "b": db,
            "W_d1": dW_d1, "b_d1": db_d1,
            "W_out": dW_out, "b_out": db_out
        }
        params = {
            "W": self.W, "b": self.b,
            "W_d1": self.W_d1, "b_d1": self.b_d1,
            "W_out": self.W_out, "b_out": self.b_out
        }
        
        for k in grads:
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * grads[k]
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1.0 - beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - beta2 ** self.t)
            params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class LSTMRegressor:
    """
    Unified LSTM Regressor matching Listing 4 & Table 10 of the report.
    Automatically leverages TensorFlow/Keras when available, or our optimized
    vectorized NumPy sequence engine.
    """
    def __init__(
        self,
        window_size: int = 30,
        n_features: int = 24,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        random_seed: int = 42
    ):
        self.window_size = window_size
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        self.use_tf = HAS_TENSORFLOW
        self.tf_model = None
        self.numpy_model = None
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
        self.is_fitted: bool = False
        
        self._build_model()

    def _build_model(self):
        if self.use_tf:
            tf.random.set_seed(self.random_seed)
            self.tf_model = Sequential([
                LSTM(self.lstm_units, input_shape=(self.window_size, self.n_features)),
                Dropout(self.dropout_rate),
                Dense(self.dense_units, activation="relu"),
                Dense(1)
            ])
            self.tf_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss="mse"
            )
        else:
            self.numpy_model = NumpyLSTM(
                input_dim=self.n_features,
                hidden_dim=self.lstm_units,
                dense_dim=self.dense_units,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
                random_seed=self.random_seed
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 25,
        batch_size: int = 64,
        patience: int = 8,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model with early stopping on validation loss.
        """
        if self.use_tf and self.tf_model is not None:
            callbacks = []
            if validation_data is not None:
                callbacks.append(EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=verbose
                ))
            hist = self.tf_model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            self.history["loss"] = [float(x) for x in hist.history.get("loss", [])]
            self.history["val_loss"] = [float(x) for x in hist.history.get("val_loss", [])]
            self.is_fitted = True
            return self.history

        # Pure NumPy Training Loop (Vectorized Mini-batch Adam)
        assert self.numpy_model is not None
        N = len(X_train)
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0
        
        self.history = {"loss": [], "val_loss": []}
        
        if verbose:
            print(f"[LSTM] Training sequence model ({N} sequences, {epochs} max epochs)...")
            
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            # Shuffle batch indices
            indices = np.random.permutation(N)
            epoch_losses = []
            
            for start_idx in range(0, N, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                X_b = X_train[batch_idx]
                y_b = y_train[batch_idx]
                
                # Forward pass
                y_pred_b, cache = self.numpy_model.forward_batch(X_b, training=True)
                # MSE loss
                loss_b = float(np.mean((y_pred_b.flatten() - y_b) ** 2))
                epoch_losses.append(loss_b)
                
                # Backward pass & Adam update
                self.numpy_model.backward_batch(y_b, y_pred_b, cache)
                
            train_loss = float(np.mean(epoch_losses))
            self.history["loss"].append(train_loss)
            
            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_preds = self.predict(X_val)
                val_loss = float(np.mean((val_preds - y_val) ** 2))
                self.history["val_loss"].append(val_loss)
                
                # Check Early Stopping
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Deep copy best weights
                    best_weights = {
                        "W": self.numpy_model.W.copy(),
                        "b": self.numpy_model.b.copy(),
                        "W_d1": self.numpy_model.W_d1.copy(),
                        "b_d1": self.numpy_model.b_d1.copy(),
                        "W_out": self.numpy_model.W_out.copy(),
                        "b_out": self.numpy_model.b_out.copy()
                    }
                else:
                    patience_counter += 1
                    
            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs or patience_counter >= patience):
                val_str = f" - val_loss: {val_loss:.2f}" if val_loss is not None else ""
                print(f"Epoch {epoch:2d}/{epochs} - loss: {train_loss:.2f}{val_str}")
                
            if validation_data is not None and patience_counter >= patience:
                if verbose:
                    print(f"[LSTM] Early stopping triggered at epoch {epoch}. Restoring best weights.")
                if best_weights:
                    self.numpy_model.W = best_weights["W"]
                    self.numpy_model.b = best_weights["b"]
                    self.numpy_model.W_d1 = best_weights["W_d1"]
                    self.numpy_model.b_d1 = best_weights["b_d1"]
                    self.numpy_model.W_out = best_weights["W_out"]
                    self.numpy_model.b_out = best_weights["b_out"]
                break
                
        elapsed = time.time() - start_time
        if verbose:
            print(f"[LSTM] Training completed in {elapsed:.1f}s.")
            
        self.is_fitted = True
        return self.history

    def predict(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        if not self.is_fitted and self.numpy_model is None and self.tf_model is None:
            raise RuntimeError("LSTM model has not been trained or loaded.")
            
        if self.use_tf and self.tf_model is not None:
            preds = self.tf_model.predict(X, batch_size=batch_size, verbose=0)
            return np.clip(preds.flatten(), 0.0, None)
            
        # NumPy forward pass in chunks to preserve memory
        assert self.numpy_model is not None
        N = len(X)
        all_preds = []
        for i in range(0, N, batch_size):
            X_chunk = X[i:i + batch_size]
            y_chunk, _ = self.numpy_model.forward_batch(X_chunk, training=False)
            all_preds.append(y_chunk.flatten())
            
        preds = np.concatenate(all_preds)
        return np.clip(preds, 0.0, None)

    def save(self, filepath: Union[str, Path]):
        meta = {
            "window_size": self.window_size,
            "n_features": self.n_features,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "random_seed": self.random_seed,
            "use_tf": self.use_tf,
            "history": self.history
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.use_tf and self.tf_model is not None:
            # Save Keras weights
            meta_path = path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            self.tf_model.save(str(path))
        else:
            # Save NumPy weights via joblib
            payload = {
                "meta": meta,
                "weights": {
                    "W": self.numpy_model.W,
                    "b": self.numpy_model.b,
                    "W_d1": self.numpy_model.W_d1,
                    "b_d1": self.numpy_model.b_d1,
                    "W_out": self.numpy_model.W_out,
                    "b_out": self.numpy_model.b_out
                }
            }
            joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LSTMRegressor":
        path = Path(filepath)
        if path.is_dir() or str(path).endswith(".keras") or str(path).endswith(".h5"):
            # Load Keras model
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            else:
                meta = {}
            obj = cls(
                window_size=meta.get("window_size", 30),
                n_features=meta.get("n_features", 24),
                lstm_units=meta.get("lstm_units", 64),
                dense_units=meta.get("dense_units", 32)
            )
            obj.tf_model = tf.keras.models.load_model(str(path))
            obj.use_tf = True
            obj.is_fitted = True
            obj.history = meta.get("history", {})
            return obj
            
        # Load NumPy model
        payload = joblib.load(filepath)
        meta = payload["meta"]
        obj = cls(
            window_size=meta["window_size"],
            n_features=meta["n_features"],
            lstm_units=meta["lstm_units"],
            dense_units=meta["dense_units"],
            dropout_rate=meta["dropout_rate"],
            learning_rate=meta["learning_rate"],
            random_seed=meta["random_seed"]
        )
        obj.use_tf = False
        obj.numpy_model = NumpyLSTM(
            input_dim=meta["n_features"],
            hidden_dim=meta["lstm_units"],
            dense_dim=meta["dense_units"],
            dropout_rate=meta["dropout_rate"],
            learning_rate=meta["learning_rate"],
            random_seed=meta["random_seed"]
        )
        w = payload["weights"]
        obj.numpy_model.W = w["W"]
        obj.numpy_model.b = w["b"]
        obj.numpy_model.W_d1 = w["W_d1"]
        obj.numpy_model.b_d1 = w["b_d1"]
        obj.numpy_model.W_out = w["W_out"]
        obj.numpy_model.b_out = w["b_out"]
        obj.history = meta.get("history", {})
        obj.is_fitted = True
        return obj
