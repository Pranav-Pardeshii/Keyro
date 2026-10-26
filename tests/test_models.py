"""
Unit tests for Anomaly Detection models and Biometric evaluation metrics.
"""

import os
import tempfile
import pytest
import numpy as np

from ml.models import (
    ScaledManhattanDetector,
    OneClassSVMDetector,
    IsolationForestDetector,
    create_model
)
from ml.evaluate import compute_eer, calculate_far_frr


@pytest.fixture
def dummy_train_and_test():
    np.random.seed(42)
    # 20 samples with 31 features, mean around 0.15, std around 0.03
    X_train = np.random.normal(loc=0.15, scale=0.03, size=(20, 31))
    # Genuine test: same distribution
    X_test_gen = np.random.normal(loc=0.15, scale=0.03, size=(10, 31))
    # Impostor test: different distribution (mean around 0.40)
    X_test_imp = np.random.normal(loc=0.40, scale=0.08, size=(10, 31))
    return X_train, X_test_gen, X_test_imp


def test_scaled_manhattan_detector(dummy_train_and_test):
    X_train, X_test_gen, X_test_imp = dummy_train_and_test
    model = ScaledManhattanDetector()
    model.fit(X_train)

    assert model.is_fitted
    assert model.threshold > 0

    scores_gen = model.score_samples(X_test_gen)
    scores_imp = model.score_samples(X_test_imp)

    assert len(scores_gen) == 10
    assert len(scores_imp) == 10
    # Genuine anomaly scores should be significantly lower than impostor scores
    assert np.mean(scores_gen) < np.mean(scores_imp)

    # Test serialization
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        loaded = ScaledManhattanDetector.load(tmp_path)
        assert loaded.is_fitted
        np.testing.assert_array_almost_equal(loaded.score_samples(X_test_gen), scores_gen)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_one_class_svm_detector(dummy_train_and_test):
    X_train, X_test_gen, X_test_imp = dummy_train_and_test
    model = OneClassSVMDetector(nu=0.1)
    model.fit(X_train)

    assert model.is_fitted
    scores_gen = model.score_samples(X_test_gen)
    scores_imp = model.score_samples(X_test_imp)

    assert np.mean(scores_gen) < np.mean(scores_imp)


def test_isolation_forest_detector(dummy_train_and_test):
    X_train, X_test_gen, X_test_imp = dummy_train_and_test
    model = IsolationForestDetector(n_estimators=50)
    model.fit(X_train)

    assert model.is_fitted
    scores_gen = model.score_samples(X_test_gen)
    scores_imp = model.score_samples(X_test_imp)

    assert np.mean(scores_gen) < np.mean(scores_imp)


def test_compute_eer():
    # Perfectly separable scores
    gen_scores = np.array([0.1, 0.2, 0.3, 0.4])
    imp_scores = np.array([0.7, 0.8, 0.9, 1.0])

    res = compute_eer(gen_scores, imp_scores)
    assert res["eer"] <= 0.05
    assert res["auc"] == 1.0


def test_calculate_far_frr():
    gen_scores = np.array([1.0, 1.2, 1.4])
    imp_scores = np.array([2.0, 2.5, 3.0])

    # With threshold 1.5:
    # All gen < 1.5 -> FRR = 0
    # All imp >= 1.5 -> FAR = 0
    far, frr = calculate_far_frr(gen_scores, imp_scores, threshold=1.5)
    assert far == 0.0
    assert frr == 0.0
