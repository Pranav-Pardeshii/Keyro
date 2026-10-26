"""
Anomaly Detection Models for Keystroke Dynamics Authentication.
Implements:
1. Scaled Manhattan Distance Baseline (Killourhy & Maxion, 2009)
2. One-Class Support Vector Machine (OC-SVM)
3. Isolation Forest
"""

import os
from typing import Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import joblib


class BaseKeystrokeDetector:
    """Base class for keystroke anomaly detectors."""

    def __init__(self, name: str):
        self.name = name
        self.threshold: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "BaseKeystrokeDetector":
        raise NotImplementedError

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Computes anomaly scores.
        Convention: HIGHER score = MORE ANOMALOUS (likely impostor).
        LOWER score = MORE NORMAL (likely genuine).
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predicts binary classification:
        True (1) = Genuine (ACCEPTED)
        False (0) = Impostor (REJECTED)
        """
        thresh = threshold if threshold is not None else self.threshold
        scores = self.score_samples(X)
        return scores < thresh

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BaseKeystrokeDetector":
        return joblib.load(filepath)


class ScaledManhattanDetector(BaseKeystrokeDetector):
    """
    Scaled Manhattan Distance anomaly detector.
    Computes absolute difference from the genuine user's feature mean,
    scaled by standard deviation (or MAD):
    Score(x) = (1 / D) * sum(|x_i - mean_i| / std_i)
    """

    def __init__(self, epsilon: float = 1e-6, use_mad: bool = False):
        super().__init__(name="ScaledManhattan")
        self.epsilon = epsilon
        self.use_mad = use_mad
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ScaledManhattanDetector":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)

        if self.use_mad:
            mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
            self.scale_ = mad + self.epsilon
        else:
            std = np.std(X, axis=0, ddof=1) if len(X) > 1 else np.ones(X.shape[1])
            self.scale_ = np.where(std < self.epsilon, self.epsilon, std)

        self.is_fitted = True
        train_scores = self.score_samples(X)
        # Default operational threshold: mean + 2.5 * std of genuine scores
        self.threshold = float(np.mean(train_scores) + 2.5 * (np.std(train_scores) or 0.1))
        return self

    def get_threshold_bounds(self):
        """Returns (min_thresh, max_thresh, step) for slider calibration."""
        t = self.threshold
        min_t = max(0.2, round(t - 0.8, 3))
        max_t = round(t + 1.2, 3)
        return (min_t, max_t, 0.01)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        diff = np.abs(X - self.mean_) / self.scale_
        return np.mean(diff, axis=1)


class OneClassSVMDetector(BaseKeystrokeDetector):
    """
    One-Class Support Vector Machine anomaly detector.
    Features are standardized via StandardScaler before kernel fitting.
    Score = -decision_function(x), so higher score = outlier / anomaly.
    """

    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale"):
        super().__init__(name="OneClassSVM")
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        X = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        self.is_fitted = True
        train_scores = self.score_samples(X)
        self.threshold = float(np.percentile(train_scores, 95))
        return self

    def get_threshold_bounds(self):
        """Returns (min_thresh, max_thresh, step) for slider calibration."""
        t = round(self.threshold, 4)
        min_t = round(min(-0.20, t - 0.20), 3)
        max_t = round(max(0.20, t + 0.20), 3)
        return (min_t, max_t, 0.005)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        # Sklearn decision_function is positive for inliers, negative for outliers
        return -self.model.decision_function(X_scaled)


class IsolationForestDetector(BaseKeystrokeDetector):
    """
    Isolation Forest anomaly detector.
    Score = -score_samples(x), so higher score = anomalous (faster tree isolation).
    """

    def __init__(self, n_estimators: int = 100, contamination: float = 0.05, random_state: int = 42):
        super().__init__(name="IsolationForest")
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        X = np.asarray(X, dtype=np.float64)
        self.model.fit(X)

        self.is_fitted = True
        train_scores = self.score_samples(X)
        self.threshold = float(np.percentile(train_scores, 95))
        return self

    def get_threshold_bounds(self):
        """Returns (min_thresh, max_thresh, step) for slider calibration."""
        t = self.threshold
        min_t = round(max(0.20, t - 0.25), 3)
        max_t = round(min(0.90, t + 0.25), 3)
        return (min_t, max_t, 0.005)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # score_samples returns opposite of anomaly score (lower is more abnormal)
        return -self.model.score_samples(X)


def create_model(model_name: str, **kwargs) -> BaseKeystrokeDetector:
    """Factory function for creating anomaly detectors."""
    name_lower = model_name.lower().replace("-", "").replace("_", "")
    if "manhattan" in name_lower:
        return ScaledManhattanDetector(**kwargs)
    elif "svm" in name_lower:
        return OneClassSVMDetector(**kwargs)
    elif "forest" in name_lower or "isolation" in name_lower:
        return IsolationForestDetector(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
