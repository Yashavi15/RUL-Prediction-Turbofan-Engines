"""
Data Loader Module for NASA PHM08 turbofan engine dataset.
Loads, validates, and audits raw space-separated text files.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALL_RAW_COLS, TRAIN_DATA_PATH, TEST_DATA_PATH, FINAL_TEST_DATA_PATH


def load_dataset(file_path: Path, dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Load a PHM08 whitespace-delimited data file with standard column names.
    
    Args:
        file_path: Path to .txt data file
        dataset_name: Label for logging/errors
        
    Returns:
        pd.DataFrame containing formatted engine time-series records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: '{path.resolve()}'. "
            "Please run 'python scripts/download_data.py' to acquire the NASA PHM08 dataset."
        )

    # Space-separated file without header
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_RAW_COLS)
    
    # Cast identifiers to int
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    
    return df


def load_train_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load PHM08 train.txt run-to-failure dataset (218 engines)."""
    return load_dataset(path or TRAIN_DATA_PATH, "PHM08 Training Set")


def load_test_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load PHM08 test.txt dataset (218 engines ending before failure)."""
    return load_dataset(path or TEST_DATA_PATH, "PHM08 Test Set")


def load_final_test_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load PHM08 final_test.txt dataset (435 engines)."""
    return load_dataset(path or FINAL_TEST_DATA_PATH, "PHM08 Final Test Set")


def audit_data_quality(df: pd.DataFrame, name: str = "Data Quality Audit") -> Dict[str, Any]:
    """
    Perform a comprehensive data quality check on a loaded PHM08 DataFrame.
    
    Returns:
        Dict with total_rows, total_columns, unique_units, null_count, duplicate_count,
        min_cycles, max_cycles, avg_cycles_per_unit.
    """
    null_count = int(df.isnull().sum().sum())
    dup_count = int(df.duplicated().sum())
    unique_units = int(df["unit"].nunique())
    cycles_per_unit = df.groupby("unit")["cycle"].max()

    return {
        "dataset_name": name,
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "unique_units": unique_units,
        "null_count": null_count,
        "duplicate_count": dup_count,
        "min_cycle_reached": int(cycles_per_unit.min()),
        "max_cycle_reached": int(cycles_per_unit.max()),
        "avg_cycles_per_unit": float(cycles_per_unit.mean()),
        "passed_integrity": (null_count == 0 and dup_count == 0 and unique_units > 0)
    }


if __name__ == "__main__":
    train_df = load_train_data()
    audit = audit_data_quality(train_df, "Train Set")
    print("Audit results:", audit)
