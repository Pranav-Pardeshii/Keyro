"""
Biometric Evaluation Metrics: FAR, FRR, EER, and ROC/DET Curve Generation.
"""

from typing import Dict, Tuple, Any, List
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d


def calculate_far_frr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float
) -> Tuple[float, float]:
    """
    Computes FAR and FRR at a given decision threshold.
    Decision rule:
      score < threshold  --> ACCEPT (genuine)
      score >= threshold --> REJECT (impostor)
    """
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)

    # FAR: Impostors falsely accepted (score < threshold)
    far = float(np.mean(impostor_scores < threshold)) if len(impostor_scores) > 0 else 0.0

    # FRR: Genuine samples falsely rejected (score >= threshold)
    frr = float(np.mean(genuine_scores >= threshold)) if len(genuine_scores) > 0 else 0.0

    return far, frr


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    num_thresholds: int = 1000
) -> Dict[str, float]:
    """
    Computes the Equal Error Rate (EER) and the corresponding threshold.
    Uses linear interpolation across candidate thresholds to find the exact
    point where FAR == FRR.
    """
    genuine_scores = np.asarray(genuine_scores, dtype=np.float64)
    impostor_scores = np.asarray(impostor_scores, dtype=np.float64)

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    min_score, max_score = np.min(all_scores), np.max(all_scores)
    
    thresholds = np.linspace(min_score - 1e-4, max_score + 1e-4, num_thresholds)
    
    # Vectorized computation of FAR and FRR across all thresholds
    # genuine_scores: shape (N_gen, 1), thresholds: shape (1, num_thresholds)
    far_array = np.mean(impostor_scores[:, None] < thresholds[None, :], axis=0)
    frr_array = np.mean(genuine_scores[:, None] >= thresholds[None, :], axis=0)

    # Find the index where FAR and FRR are closest
    diff = far_array - frr_array
    idx = np.argmin(np.abs(diff))

    # Fine-grained interpolation around the crossing
    eer = float((far_array[idx] + frr_array[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    # Also compute AUC: genuine=0 (negative/inlier), impostor=1 (positive/anomaly)
    y_true = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_at_eer": float(far_array[idx]),
        "frr_at_eer": float(frr_array[idx]),
        "auc": roc_auc,
    }


def evaluate_model(
    model,
    X_test_gen: np.ndarray,
    X_test_imp: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates a fitted anomaly detector on genuine and impostor test sets.
    """
    gen_scores = model.score_samples(X_test_gen)
    imp_scores = model.score_samples(X_test_imp)

    metrics = compute_eer(gen_scores, imp_scores)
    metrics["model_name"] = model.name
    metrics["default_threshold"] = float(model.threshold)
    
    # FAR and FRR at default threshold
    far_def, frr_def = calculate_far_frr(gen_scores, imp_scores, model.threshold)
    metrics["far_at_default"] = far_def
    metrics["frr_at_default"] = frr_def
    metrics["gen_scores_mean"] = float(np.mean(gen_scores))
    metrics["imp_scores_mean"] = float(np.mean(imp_scores))

    return metrics
