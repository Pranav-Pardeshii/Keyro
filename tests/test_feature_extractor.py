"""
Unit tests for keystroke dynamics feature extractor.
"""

import pytest
from app.feature_extractor import (
    extract_features_from_events,
    validate_feature_vector,
    calculate_deviations
)
from app.schemas import KeyEvent
from app.config import EXPECTED_KEY_SEQUENCE, FEATURE_NAMES


def create_mock_events(base_time=100.0, hold_time=0.1, flight_time=0.15):
    """Generates 11 valid mock key events for .tie5Roanl + Enter."""
    events = []
    curr = base_time
    for k in EXPECTED_KEY_SEQUENCE:
        down = curr
        up = curr + hold_time
        events.append(KeyEvent(key=k, down_time=down, up_time=up))
        curr = down + flight_time
    return events


def test_extract_features_shape_and_order():
    events = create_mock_events(hold_time=0.12, flight_time=0.25)
    features = extract_features_from_events(events)

    assert len(features) == 31
    # Check H.period
    assert pytest.approx(features[0], 1e-4) == 0.12
    # Check DD.period.t
    assert pytest.approx(features[1], 1e-4) == 0.25
    # Check UD.period.t (flight_time - hold_time = 0.25 - 0.12 = 0.13)
    assert pytest.approx(features[2], 1e-4) == 0.13
    # Check H.Return (final feature)
    assert pytest.approx(features[30], 1e-4) == 0.12


def test_extract_features_invalid_key_raises():
    events = create_mock_events()
    events[1] = KeyEvent(key="x", down_time=events[1].down_time, up_time=events[1].up_time)
    with pytest.raises(ValueError, match="Key mismatch"):
        extract_features_from_events(events)


def test_extract_features_invalid_count_raises():
    events = create_mock_events()[:9]
    with pytest.raises(ValueError, match="Expected exactly 11 key events"):
        extract_features_from_events(events)


def test_validate_feature_vector_valid():
    vec = [0.1] * 31
    res = validate_feature_vector(vec)
    assert len(res) == 31


def test_validate_feature_vector_invalid_length():
    with pytest.raises(ValueError, match="must have exactly 31 elements"):
        validate_feature_vector([0.1] * 30)


def test_validate_feature_vector_nan():
    vec = [0.1] * 31
    vec[5] = float("nan")
    with pytest.raises(ValueError, match="not a finite number"):
        validate_feature_vector(vec)


def test_calculate_deviations():
    attempt = [0.2] * 31
    template = [0.1] * 31
    devs = calculate_deviations(attempt, template, top_k=3)
    assert len(devs) == 3
    assert devs[0]["diff_ms"] == 100.0
    assert devs[0]["direction"] == "slower"
