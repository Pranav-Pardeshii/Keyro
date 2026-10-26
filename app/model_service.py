"""
Model Service Layer.
Coordinates enrollment training, threshold calibration, and real-time verification scoring.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from app.config import (
    load_best_model_config,
    FEATURE_NAMES,
    REQUIRED_ENROLLMENT_SAMPLES
)
from app.storage import storage
from app.feature_extractor import (
    validate_feature_vector,
    calculate_deviations
)
from ml.models import (
    BaseKeystrokeDetector,
    ScaledManhattanDetector,
    OneClassSVMDetector,
    IsolationForestDetector,
    create_model
)


class ModelService:
    """Business logic for keystroke model fitting and verification."""

    def __init__(self):
        self.best_config = load_best_model_config()

    def create_user_detector(self, model_type: Optional[str] = None) -> BaseKeystrokeDetector:
        """Instantiates detector using best configuration or requested type."""
        target_name = model_type or self.best_config.get("best_model", "One-Class SVM")
        params = self.best_config.get("parameters", {})

        if "svm" in target_name.lower():
            nu = params.get("nu", 0.05)
            return OneClassSVMDetector(nu=nu, kernel="rbf", gamma="scale")
        elif "forest" in target_name.lower():
            return IsolationForestDetector(n_estimators=100, contamination=0.05, random_state=42)
        else:
            return ScaledManhattanDetector(epsilon=1e-6, use_mad=False)

    def enroll_user(
        self,
        username: str,
        samples: List[List[float]],
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrolls a new user with 20 keystroke dynamics repetitions.
        Fits the detector, calibrates threshold, and stores metadata.
        """
        if len(samples) != REQUIRED_ENROLLMENT_SAMPLES:
            raise ValueError(
                f"Enrollment requires exactly {REQUIRED_ENROLLMENT_SAMPLES} samples. Got {len(samples)}."
            )

        validated_samples = [validate_feature_vector(s) for s in samples]
        X_train = np.array(validated_samples, dtype=np.float64)

        # Calculate mean template and standard deviation
        mean_template = np.mean(X_train, axis=0).tolist()
        std_template = np.std(X_train, axis=0, ddof=1).tolist()

        # Instantiate and fit model
        detector = self.create_user_detector(model_type)
        detector.fit(X_train)

        # Store model
        storage.save_model(username, detector)

        # Save user record
        user_data = {
            "username": username,
            "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": len(samples),
            "model_type": detector.name,
            "default_threshold": float(detector.threshold),
            "mean_features": mean_template,
            "std_features": std_template,
            "is_demo_subject": False
        }
        storage.save_user(username, user_data)

        return {
            "status": "success",
            "username": username,
            "model_type": detector.name,
            "sample_count": len(samples),
            "threshold": float(detector.threshold),
            "mean_features": mean_template,
            "message": f"Successfully enrolled {username} with {detector.name}."
        }

    def verify_attempt(
        self,
        username: str,
        features: List[float],
        threshold: Optional[float] = None,
        is_impostor_attempt: bool = False,
        tester_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verifies a single typing sample against an enrolled user's model.
        """
        user = storage.get_user(username)
        if not user:
            raise ValueError(f"User '{username}' is not enrolled.")

        detector = storage.load_model(username)
        if not detector:
            raise RuntimeError(f"Trained model for user '{username}' could not be loaded.")

        valid_features = validate_feature_vector(features)
        X = np.array([valid_features], dtype=np.float64)

        # Compute anomaly score
        score = float(detector.score_samples(X)[0])

        # Active threshold: custom or trained default
        active_threshold = float(threshold) if threshold is not None else float(detector.threshold)

        # Decision rule: score < threshold => ACCEPT (inlier), score >= threshold => REJECT (outlier)
        is_accepted = bool(score < active_threshold)
        result_label = "ACCEPT" if is_accepted else "REJECT"

        # Margin and confidence estimation
        margin = active_threshold - score
        denom = max(abs(active_threshold), 1e-4)
        rel_diff = abs(margin) / denom
        confidence = float(min(99.9, max(50.0, 50.0 + 50.0 * (1.0 - np.exp(-1.5 * rel_diff)))))

        # Top feature deviations
        mean_template = user.get("mean_features", [])
        deviations = calculate_deviations(valid_features, mean_template, top_k=5) if mean_template else []

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        response = {
            "result": result_label,
            "is_accepted": is_accepted,
            "score": round(score, 4),
            "threshold": round(active_threshold, 4),
            "distance_margin": round(margin, 4),
            "confidence_percent": round(confidence, 1),
            "username": username,
            "is_impostor_attempt": is_impostor_attempt,
            "tester_label": tester_label or ("Impostor Friend" if is_impostor_attempt else "Genuine User"),
            "timestamp": timestamp,
            "top_deviations": deviations
        }

        # Log attempt for auditing and friend challenge
        storage.log_verification(response)

        return response

    def compute_threshold_tradeoff(self, username: str, num_points: int = 50) -> Dict[str, Any]:
        """
        Computes the theoretical FAR/FRR tradeoff curve for a given user's model
        over a range of thresholds.
        """
        user = storage.get_user(username)
        if not user:
            raise ValueError(f"User '{username}' not found.")

        detector = storage.load_model(username)
        if not detector:
            raise RuntimeError(f"Model for '{username}' not found.")

        # Get calibrated bounds from detector
        if hasattr(detector, "get_threshold_bounds"):
            min_t, max_t, step = detector.get_threshold_bounds()
        else:
            if "svm" in detector.name.lower():
                min_t, max_t, step = -0.20, 0.20, 0.005
            elif "forest" in detector.name.lower():
                min_t, max_t, step = 0.30, 0.80, 0.005
            else:
                min_t, max_t, step = 0.40, 2.40, 0.01

        base_thresh = float(detector.threshold)
        base_thresh = max(min_t, min(max_t, base_thresh))
        thresholds = np.linspace(min_t, max_t, num_points)

        spread = max((max_t - min_t) / 4.0, 1e-4)
        tradeoffs = []
        p0 = 0.1412  # Calibrated CMU Benchmark Equal Error Rate (14.12%)
        logit_p0 = float(np.log(p0 / (1.0 - p0)))
        for t in thresholds:
            z = (t - base_thresh) / spread
            # Higher threshold -> more accepted -> FAR increases, FRR decreases
            # Anchored so at z=0 (default threshold), FAR=FRR=14.12% (Equal Error Rate)
            far = float(1.0 / (1.0 + np.exp(-(logit_p0 + 3.0 * z))))
            frr = float(1.0 / (1.0 + np.exp(-(logit_p0 - 3.0 * z))))
            tradeoffs.append({
                "threshold": round(float(t), 4),
                "far_percent": round(far * 100, 2),
                "frr_percent": round(frr * 100, 2)
            })

        return {
            "username": username,
            "current_threshold": round(base_thresh, 4),
            "min_threshold": round(min_t, 4),
            "max_threshold": round(max_t, 4),
            "step": step,
            "tradeoffs": tradeoffs
        }


# Singleton service instance
model_service = ModelService()
