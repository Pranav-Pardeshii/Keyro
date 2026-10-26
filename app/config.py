"""
Application Configuration and Constants.
"""

import os
import json
from typing import List, Dict, Any

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_FILE = os.path.join(DATA_DIR, "DSL-StrongPasswordData.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BEST_MODEL_CONFIG_FILE = os.path.join(RESULTS_DIR, "best_model_config.json")
BENCHMARK_SUMMARY_FILE = os.path.join(RESULTS_DIR, "benchmark_summary.json")

STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
USER_MODELS_DIR = os.path.join(STORAGE_DIR, "models")
USER_DATA_FILE = os.path.join(STORAGE_DIR, "users.json")
VERIFICATION_LOGS_FILE = os.path.join(STORAGE_DIR, "verification_logs.json")

# Ensure required directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(USER_MODELS_DIR, exist_ok=True)

# Fixed Passphrase definition
TARGET_PASSPHRASE = ".tie5Roanl"
EXPECTED_KEY_SEQUENCE = [".", "t", "i", "e", "5", "R", "o", "a", "n", "l", "Enter"]
REQUIRED_ENROLLMENT_SAMPLES = 20

# 31 standard CMU keystroke features
FEATURE_NAMES: List[str] = [
    "H.period", "DD.period.t", "UD.period.t",
    "H.t", "DD.t.i", "UD.t.i",
    "H.i", "DD.i.e", "UD.i.e",
    "H.e", "DD.e.five", "UD.e.five",
    "H.five", "DD.five.Shift.r", "UD.five.Shift.r",
    "H.Shift.r", "DD.Shift.r.o", "UD.Shift.r.o",
    "H.o", "DD.o.a", "UD.o.a",
    "H.a", "DD.a.n", "UD.a.n",
    "H.n", "DD.n.l", "UD.n.l",
    "H.l", "DD.l.Return", "UD.l.Return",
    "H.Return"
]

def load_best_model_config() -> Dict[str, Any]:
    """Loads the best model config discovered in Part 1 offline ML."""
    if os.path.exists(BEST_MODEL_CONFIG_FILE):
        try:
            with open(BEST_MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Default fallback if benchmark has not run
    return {
        "best_model": "One-Class SVM",
        "mean_eer_percent": 14.12,
        "mean_auc": 0.9113,
        "parameters": {"nu": 0.05, "kernel": "rbf", "gamma": "scale"}
    }
