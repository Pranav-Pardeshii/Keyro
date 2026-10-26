"""
Feature Extractor for Keystroke Dynamics.
Converts raw key events for the passphrase '.tie5Roanl' + Enter into the
standard 31 CMU timing features (Hold, Down-Down, Up-Down).
"""

from typing import List, Dict, Any, Optional
import numpy as np
from app.config import FEATURE_NAMES, EXPECTED_KEY_SEQUENCE
from app.schemas import KeyEvent


def normalize_key_name(key: str) -> str:
    """Normalizes key strings for sequence validation."""
    key = str(key).strip()
    if key in [".", "period", "Decimal"]:
        return "."
    if key in ["Enter", "Return"]:
        return "Enter"
    return key


def extract_features_from_events(events: List[KeyEvent]) -> List[float]:
    """
    Extracts the 31 CMU features from a sequence of 11 key events.
    Key sequence: ['.', 't', 'i', 'e', '5', 'R', 'o', 'a', 'n', 'l', 'Enter']
    Features (31 total):
      For m in 0..9:
        H.km           (up_m - down_m)
        DD.km.k_{m+1}  (down_{m+1} - down_m)
        UD.km.k_{m+1}  (down_{m+1} - up_m)
      H.Return         (up_10 - down_10)
    """
    if len(events) != 11:
        raise ValueError(f"Expected exactly 11 key events (10 chars + Enter), received {len(events)} events.")

    # Validate key sequence
    for idx, (expected, ev) in enumerate(zip(EXPECTED_KEY_SEQUENCE, events)):
        norm_key = normalize_key_name(ev.key)
        # Handle 'R' / 'r' case-insensitively or as capital
        if expected.lower() != norm_key.lower():
            raise ValueError(
                f"Key mismatch at position {idx}: expected '{expected}', found '{ev.key}'."
            )

    down_times = [ev.down_time for ev in events]
    up_times = [ev.up_time for ev in events]

    # Validate timestamps
    for i in range(11):
        if up_times[i] < down_times[i]:
            raise ValueError(f"Key {events[i].key} has invalid duration (keyup before keydown).")

    features = []
    for m in range(10):
        # Hold time of current key
        h = up_times[m] - down_times[m]
        # Down-Down time to next key
        dd = down_times[m + 1] - down_times[m]
        # Up-Down time to next key
        ud = down_times[m + 1] - up_times[m]

        features.append(float(h))
        features.append(float(dd))
        features.append(float(ud))

    # Hold time of last key (Return)
    h_return = up_times[10] - down_times[10]
    features.append(float(h_return))

    if len(features) != 31:
        raise RuntimeError(f"Feature computation error: generated {len(features)} features instead of 31.")

    return features


def validate_feature_vector(features: List[float]) -> List[float]:
    """Validates that a feature vector has 31 valid, finite timing numbers."""
    if len(features) != 31:
        raise ValueError(f"Feature vector must have exactly 31 elements, got {len(features)}.")

    features = [float(f) for f in features]
    for idx, val in enumerate(features):
        if not np.isfinite(val):
            raise ValueError(f"Feature at index {idx} ({FEATURE_NAMES[idx]}) is not a finite number: {val}")
        if val < -5.0 or val > 60.0:
            raise ValueError(f"Feature at index {idx} ({FEATURE_NAMES[idx]}) has out-of-range value: {val:.4f}s")

    return features


def calculate_deviations(
    attempt_features: List[float],
    mean_template: List[float],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Identifies the top keystroke timing deviations compared to the user's template."""
    deviations = []
    for i, name in enumerate(FEATURE_NAMES):
        user_val = mean_template[i]
        curr_val = attempt_features[i]
        diff_ms = (curr_val - user_val) * 1000.0  # in milliseconds
        ratio = (curr_val / (user_val + 1e-4)) if user_val > 0 else 1.0
        deviations.append({
            "feature": name,
            "feature_index": i,
            "user_mean_ms": round(user_val * 1000.0, 1),
            "attempt_ms": round(curr_val * 1000.0, 1),
            "diff_ms": round(diff_ms, 1),
            "abs_diff_ms": round(abs(diff_ms), 1),
            "direction": "slower" if diff_ms > 0 else "faster"
        })

    deviations.sort(key=lambda d: d["abs_diff_ms"], reverse=True)
    return deviations[:top_k]
