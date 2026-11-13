"""
Global configuration and constants for the RUL Prediction Platform.
Matches NASA PHM08 Challenge specification.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TRAIN_DATA_PATH = DATA_DIR / "train.txt"
TEST_DATA_PATH = DATA_DIR / "test.txt"
FINAL_TEST_DATA_PATH = DATA_DIR / "final_test.txt"

# Dataset Column Definitions
ID_COLS = ["unit", "cycle"]
OP_SETTINGS_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_RAW_COLS = ID_COLS + OP_SETTINGS_COLS + SENSOR_COLS

# Meaningful sensor descriptions for turbofan C-MAPSS / PHM08 architecture
SENSOR_DESCRIPTIONS = {
    "s1": "Fan inlet temperature (T2) [°R]",
    "s2": "LPC outlet temperature (T24) [°R]",
    "s3": "HPC outlet temperature (T30) [°R]",
    "s4": "LPT outlet temperature (T50) [°R]",
    "s5": "Fan inlet pressure (P2) [psia]",
    "s6": "Bypass duct pressure (P15) [psia]",
    "s7": "HPC outlet pressure (P30) [psia]",
    "s8": "Physical fan speed (Nf) [rpm]",
    "s9": "Physical core speed (Nc) [rpm]",
    "s10": "Engine pressure ratio (epr)",
    "s11": "HPC outlet static pressure (Ps30) [psia]",
    "s12": "Ratio of fuel flow to Ps30 (phi) [pps/psia]",
    "s13": "Corrected fan speed (NRf) [rpm]",
    "s14": "Corrected core speed (NRc) [rpm]",
    "s15": "Bypass ratio (BPR)",
    "s16": "Burner fuel-air ratio (farB)",
    "s17": "Bleed enthalpy (htBleed)",
    "s18": "Demanded fan speed (Nf_dmd) [rpm]",
    "s19": "Demanded corrected fan speed (PCNfR_dmd) [rpm]",
    "s20": "HPT coolant bleed (W31) [lbm/s]",
    "s21": "LPT coolant bleed (W32) [lbm/s]"
}

# Operational Settings descriptions
OP_SETTINGS_DESCRIPTIONS = {
    "op1": "Altitude [0 - 42,000 ft]",
    "op2": "Mach Number [0.0 - 0.84]",
    "op3": "Throttle Resolver Angle (TRA) [0 - 100]"
}

# Default Model Parameters
DEFAULT_WINDOW_SIZE = 30      # L consecutive cycles
DEFAULT_RUL_CAP = 125          # Piecewise linear RUL cap (common in PHM/C-MAPSS literature)
DEFAULT_RANDOM_SEED = 42

# Train / Validation / Test Engine split ratio
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
