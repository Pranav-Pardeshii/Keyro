"""
Keyro Keystroke Dynamics Authentication Server.
FastAPI Application serving REST API endpoints and static SPA frontend.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    APP_DIR,
    RESULTS_DIR,
    BENCHMARK_SUMMARY_FILE,
    BEST_MODEL_CONFIG_FILE,
    TARGET_PASSPHRASE,
    FEATURE_NAMES
)
from app.schemas import (
    EnrollRequest,
    EnrollResponse,
    VerifyRequest,
    VerifyResponse,
    UserSummary
)
from app.storage import storage
from app.model_service import model_service
from app.feature_extractor import extract_features_from_events

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to preload demo users on startup."""
    try:
        preloaded = storage.preload_cmu_demo_users(count=3)
        if preloaded:
            print(f"[Keyro] Initialized demo subjects: {', '.join(preloaded)}")
    except Exception as e:
        print(f"[Keyro] Note: Demo subjects preload skipped: {e}")
    yield

app = FastAPI(
    title="Keyro — Keystroke Dynamics Biometric Authentication",
    description="End-to-End behavioral biometric authentication using typing rhythm anomalies.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local testing flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file directories
STATIC_DIR = os.path.join(APP_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)

# Mount static files and results
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(RESULTS_DIR):
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the single-page application frontend."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "keyro", "passphrase": TARGET_PASSPHRASE}


@app.post("/enroll", response_model=EnrollResponse, status_code=status.HTTP_201_CREATED)
async def enroll_user(request: EnrollRequest):
    """
    Enrolls a user by storing 20 typing samples and training their anomaly detector.
    """
    username = request.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty.")

    try:
        result = model_service.enroll_user(
            username=username,
            samples=request.samples,
            model_type=request.model_type
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@app.post("/verify", response_model=VerifyResponse)
async def verify_user(request: VerifyRequest):
    """
    Verifies a keystroke dynamics attempt against an enrolled user's model.
    Accepts either pre-extracted 31 features or raw key event timestamps.
    """
    username = request.username.strip()
    if not storage.user_exists(username):
        raise HTTPException(status_code=404, detail=f"User '{username}' is not enrolled.")

    features = request.features
    if not features and request.raw_events:
        try:
            features = extract_features_from_events(request.raw_events)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Feature extraction failed: {str(e)}")

    if not features:
        raise HTTPException(status_code=400, detail="Must provide either 'features' (31 floats) or 'raw_events'.")

    try:
        response = model_service.verify_attempt(
            username=username,
            features=features,
            threshold=request.threshold,
            is_impostor_attempt=request.is_impostor_attempt,
            tester_label=request.tester_label
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.get("/users", response_model=List[UserSummary])
async def list_users():
    """Returns a list of all enrolled users and their verification stats."""
    users = storage.list_users()
    summaries = []
    for u in users:
        uname = u["username"]
        stats = storage.get_user_stats(uname)
        summaries.append(
            UserSummary(
                username=uname,
                enrolled_at=u.get("enrolled_at", "N/A"),
                sample_count=u.get("sample_count", 0),
                model_type=u.get("model_type", "Unknown"),
                default_threshold=round(u.get("default_threshold", 0.0), 4),
                total_verifications=stats.get("total_attempts", 0),
                genuine_attempts=stats.get("genuine_attempts", 0),
                impostor_attempts=stats.get("impostor_attempts", 0)
            )
        )
    return summaries


@app.get("/stats/{username}")
async def get_user_stats(username: str):
    """Returns detailed statistics and recent verification logs for a user."""
    if not storage.user_exists(username):
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")
    return storage.get_user_stats(username)


@app.get("/tradeoff/{username}")
async def get_threshold_tradeoff(username: str):
    """Returns simulated FAR/FRR tradeoffs across threshold variations."""
    if not storage.user_exists(username):
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")
    try:
        return model_service.compute_threshold_tradeoff(username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmark/summary")
async def get_benchmark_summary():
    """Returns the offline ML benchmark evaluation results and best model config."""
    summary_data = {}
    config_data = {}
    if os.path.exists(BENCHMARK_SUMMARY_FILE):
        import json
        with open(BENCHMARK_SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
    if os.path.exists(BEST_MODEL_CONFIG_FILE):
        import json
        with open(BEST_MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)

    return {
        "passphrase": TARGET_PASSPHRASE,
        "feature_count": len(FEATURE_NAMES),
        "best_config": config_data,
        "summary": summary_data
    }


@app.post("/demo/preload")
async def preload_demo():
    """Preloads CMU benchmark demo users (s002, s004, s005)."""
    preloaded = storage.preload_cmu_demo_users(count=3)
    return {"status": "success", "preloaded_subjects": preloaded}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
