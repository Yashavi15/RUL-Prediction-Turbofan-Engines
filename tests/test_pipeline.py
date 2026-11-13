"""
Automated Test Suite for RUL Prediction Platform.
Tests data loading, preprocessing, zero-leakage splits,
model training/inference, and asymmetric PHM08 score evaluation.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, SENSOR_COLS, OP_SETTINGS_COLS
from src.data_loader import load_train_data, load_test_data, audit_data_quality
from src.preprocessing import (
    add_rul_labels, engine_wise_split, RULPreprocessor,
    create_sliding_sequences
)
from src.feature_engineering import analyze_sensor_variance, select_feature_columns
from src.evaluate import calculate_phm08_score, evaluate_predictions
from src.models import BaselineRegressor, LSTMRegressor


class TestRULPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load sample from train dataset for tests
        cls.train_df = load_train_data()

    def test_01_data_loader_integrity(self):
        """Verify PHM08 dataset dimensions, columns, and data quality."""
        self.assertEqual(len(self.train_df.columns), 26)
        self.assertEqual(self.train_df["unit"].nunique(), 218)
        self.assertEqual(self.train_df.isnull().sum().sum(), 0)
        self.assertEqual(self.train_df.duplicated().sum(), 0)
        
        audit = audit_data_quality(self.train_df)
        self.assertTrue(audit["passed_integrity"])
        self.assertGreater(audit["total_rows"], 40000)

    def test_02_rul_label_generation(self):
        """Verify RUL calculation and piecewise capping."""
        df_labeled = add_rul_labels(self.train_df, cap_rul=125)
        self.assertIn("RUL", df_labeled.columns)
        self.assertIn("RUL_raw", df_labeled.columns)
        
        # At failure cycle (last cycle of unit 1), RUL must be exactly 0
        u1_last = df_labeled[df_labeled["unit"] == 1].iloc[-1]
        self.assertEqual(u1_last["RUL"], 0)
        self.assertEqual(u1_last["RUL_raw"], 0)
        
        # Max capped value cannot exceed 125
        self.assertLessEqual(df_labeled["RUL"].max(), 125)

    def test_03_zero_data_leakage_split(self):
        """Verify engine-wise partitioning strictly isolates engine trajectories."""
        train_df, val_df, test_df, engine_dict = engine_wise_split(
            self.train_df, train_ratio=0.7, val_ratio=0.15, random_seed=42
        )
        
        train_set = set(engine_dict["train"])
        val_set = set(engine_dict["val"])
        test_set = set(engine_dict["test"])
        
        # Assert completely mutually exclusive engine sets
        self.assertTrue(train_set.isdisjoint(val_set))
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertTrue(val_set.isdisjoint(test_set))
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), 218)

    def test_04_preprocessor_scaling_and_sequences(self):
        """Verify standardization and sliding-window dimensions."""
        features = select_feature_columns(self.train_df)
        preprocessor = RULPreprocessor(features)
        
        # Take a subset of 3 engines for fast test
        subset = self.train_df[self.train_df["unit"].isin([1, 2, 3])].copy()
        subset_labeled = add_rul_labels(subset, cap_rul=125)
        
        scaled = preprocessor.fit_transform(subset_labeled)
        self.assertTrue(preprocessor.is_fitted)
        
        window = 20
        X, y, meta = create_sliding_sequences(scaled, features, window_size=window)
        
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[1], window)
        self.assertEqual(X.shape[2], len(features))
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(meta))

    def test_05_asymmetric_phm08_score(self):
        """Verify PHM08 score correctly penalizes late predictions more than early ones."""
        y_true = np.array([100.0, 100.0])
        # Late prediction: pred=110, d=+10 -> penalty = exp(10/10) - 1 = e - 1 ~= 1.718
        y_late = np.array([110.0, 100.0])
        # Early prediction: pred=90, d=-10 -> penalty = exp(10/13) - 1 = exp(0.769) - 1 ~= 1.158
        y_early = np.array([90.0, 100.0])
        
        score_late = calculate_phm08_score(y_true, y_late)
        score_early = calculate_phm08_score(y_true, y_early)
        
        self.assertGreater(score_late, score_early, "Late predictions must be penalized more heavily than early ones.")
        
        # Perfect prediction score must be 0.0
        self.assertAlmostEqual(calculate_phm08_score(y_true, y_true), 0.0, places=4)

    def test_06_model_inference(self):
        """Test Baseline Random Forest and LSTM Regressor execution."""
        features = select_feature_columns(self.train_df)
        subset = self.train_df[self.train_df["unit"].isin([1, 2])].copy()
        subset_labeled = add_rul_labels(subset, cap_rul=125)
        
        preprocessor = RULPreprocessor(features)
        scaled = preprocessor.fit_transform(subset_labeled)
        X, y, _ = create_sliding_sequences(scaled, features, window_size=15)
        
        # Baseline model
        rf = BaselineRegressor(n_estimators=5, max_depth=5, random_state=42)
        rf.fit(X[:30], y[:30])
        preds_rf = rf.predict(X[:10])
        self.assertEqual(len(preds_rf), 10)
        self.assertTrue((preds_rf >= 0).all())
        
        # LSTM model
        lstm = LSTMRegressor(window_size=15, n_features=len(features), lstm_units=16, dense_units=8)
        lstm.fit(X[:30], y[:30], epochs=2, batch_size=16, verbose=0)
        preds_lstm = lstm.predict(X[:10])
        self.assertEqual(len(preds_lstm), 10)
        self.assertTrue((preds_lstm >= 0).all())


if __name__ == "__main__":
    unittest.main()
