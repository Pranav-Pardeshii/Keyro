"""
ML package for Keystroke Dynamics Authentication.
"""
from ml.dataset import load_cmu_dataset, get_user_splits, FEATURE_COLUMNS
from ml.models import ScaledManhattanDetector, OneClassSVMDetector, IsolationForestDetector
from ml.evaluate import evaluate_model, compute_eer, calculate_far_frr

__all__ = [
    "load_cmu_dataset",
    "get_user_splits",
    "FEATURE_COLUMNS",
    "ScaledManhattanDetector",
    "OneClassSVMDetector",
    "IsolationForestDetector",
    "evaluate_model",
    "compute_eer",
    "calculate_far_frr",
]
