"""
Integration tests for FastAPI endpoints.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.config import REQUIRED_ENROLLMENT_SAMPLES

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "keyro"


def test_serve_index():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "Keyro" in res.text


def test_enroll_and_verify_flow():
    test_user = "test_alice"
    np.random.seed(101)
    # Generate 20 samples
    samples = np.random.normal(loc=0.18, scale=0.02, size=(REQUIRED_ENROLLMENT_SAMPLES, 31)).tolist()

    # 1. Enroll User
    enroll_payload = {
        "username": test_user,
        "samples": samples,
        "model_type": "One-Class SVM"
    }
    enroll_res = client.post("/enroll", json=enroll_payload)
    assert enroll_res.status_code == 201, enroll_res.text
    data = enroll_res.json()
    assert data["username"] == test_user
    assert data["sample_count"] == 20
    assert data["threshold"] is not None

    # 2. Verify Genuine Attempt (similar distribution)
    genuine_sample = np.random.normal(loc=0.18, scale=0.02, size=31).tolist()
    verify_payload = {
        "username": test_user,
        "features": genuine_sample
    }
    verify_res = client.post("/verify", json=verify_payload)
    assert verify_res.status_code == 200, verify_res.text
    vdata = verify_res.json()
    assert "result" in vdata
    assert "score" in vdata
    assert "threshold" in vdata
    assert vdata["username"] == test_user

    # 3. Verify Impostor Attempt (wildly different distribution)
    impostor_sample = np.random.normal(loc=0.85, scale=0.15, size=31).tolist()
    imp_payload = {
        "username": test_user,
        "features": impostor_sample,
        "is_impostor_attempt": True,
        "tester_label": "Hacker Bob"
    }
    imp_res = client.post("/verify", json=imp_payload)
    assert imp_res.status_code == 200
    idata = imp_res.json()
    assert idata["result"] == "REJECT"
    assert idata["is_accepted"] is False
    assert idata["is_impostor_attempt"] is True

    # 4. List Users
    users_res = client.get("/users")
    assert users_res.status_code == 200
    user_list = users_res.json()
    assert any(u["username"] == test_user for u in user_list)

    # 5. Get User Stats
    stats_res = client.get(f"/stats/{test_user}")
    assert stats_res.status_code == 200
    sdata = stats_res.json()
    assert sdata["total_attempts"] >= 2
    assert sdata["impostor_attempts"] >= 1

    # 6. Get Tradeoff
    tradeoff_res = client.get(f"/tradeoff/{test_user}")
    assert tradeoff_res.status_code == 200
    tdata = tradeoff_res.json()
    assert "tradeoffs" in tdata
    assert len(tdata["tradeoffs"]) > 0


def test_enroll_invalid_sample_count():
    res = client.post("/enroll", json={
        "username": "bad_user",
        "samples": [[0.1] * 31] * 5  # only 5 samples instead of 20
    })
    assert res.status_code == 400
    assert "Enrollment requires exactly 20 samples" in res.json()["detail"]


def test_verify_non_existent_user():
    res = client.post("/verify", json={
        "username": "ghost_user",
        "features": [0.1] * 31
    })
    assert res.status_code == 404
    assert "not enrolled" in res.json()["detail"]


def test_benchmark_summary_endpoint():
    res = client.get("/benchmark/summary")
    assert res.status_code == 200
    bdata = res.json()
    assert bdata["passphrase"] == ".tie5Roanl"
    assert bdata["feature_count"] == 31
    assert "best_config" in bdata
