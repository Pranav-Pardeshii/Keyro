"""
Storage and Persistence Layer for Keyro.
Manages user profiles, trained models, verification logs, and demo preloading.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import numpy as np

from app.config import (
    STORAGE_DIR,
    USER_MODELS_DIR,
    USER_DATA_FILE,
    VERIFICATION_LOGS_FILE,
    DATA_FILE,
    FEATURE_NAMES
)
from ml.models import BaseKeystrokeDetector, create_model


class StorageManager:
    """Handles JSON file storage for user metadata and joblib files for models."""

    def __init__(self):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        os.makedirs(USER_MODELS_DIR, exist_ok=True)
        if not os.path.exists(USER_DATA_FILE):
            self._write_json(USER_DATA_FILE, {})
        if not os.path.exists(VERIFICATION_LOGS_FILE):
            self._write_json(VERIFICATION_LOGS_FILE, [])

    def _read_json(self, path: str) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {} if path == USER_DATA_FILE else []

    def _write_json(self, path: str, data: Any) -> None:
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, path)

    def save_user(self, username: str, user_data: Dict[str, Any]) -> None:
        users = self._read_json(USER_DATA_FILE)
        users[username] = user_data
        self._write_json(USER_DATA_FILE, users)

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        users = self._read_json(USER_DATA_FILE)
        return users.get(username)

    def list_users(self) -> List[Dict[str, Any]]:
        users = self._read_json(USER_DATA_FILE)
        return list(users.values())

    def user_exists(self, username: str) -> bool:
        users = self._read_json(USER_DATA_FILE)
        return username in users

    def get_model_path(self, username: str) -> str:
        safe_name = "".join(c for c in username if c.isalnum() or c in ("-", "_"))
        return os.path.join(USER_MODELS_DIR, f"{safe_name}.joblib")

    def save_model(self, username: str, model: BaseKeystrokeDetector) -> str:
        path = self.get_model_path(username)
        model.save(path)
        return path

    def load_model(self, username: str) -> Optional[BaseKeystrokeDetector]:
        path = self.get_model_path(username)
        if os.path.exists(path):
            return BaseKeystrokeDetector.load(path)
        return None

    def log_verification(self, log_entry: Dict[str, Any]) -> None:
        logs = self._read_json(VERIFICATION_LOGS_FILE)
        logs.append(log_entry)
        self._write_json(VERIFICATION_LOGS_FILE, logs)

    def get_user_verifications(self, username: str) -> List[Dict[str, Any]]:
        logs = self._read_json(VERIFICATION_LOGS_FILE)
        return [l for l in logs if l.get("username") == username]

    def get_user_stats(self, username: str) -> Dict[str, Any]:
        user = self.get_user(username)
        if not user:
            return {}
        attempts = self.get_user_verifications(username)
        genuine_attempts = [a for a in attempts if not a.get("is_impostor_attempt", False)]
        impostor_attempts = [a for a in attempts if a.get("is_impostor_attempt", False)]

        gen_accepted = sum(1 for a in genuine_attempts if a.get("is_accepted", False))
        gen_rejected = len(genuine_attempts) - gen_accepted
        imp_accepted = sum(1 for a in impostor_attempts if a.get("is_accepted", False))
        imp_rejected = len(impostor_attempts) - imp_accepted

        empirical_frr = (gen_rejected / len(genuine_attempts) * 100) if genuine_attempts else 0.0
        empirical_far = (imp_accepted / len(impostor_attempts) * 100) if impostor_attempts else 0.0
        defense_rate = (imp_rejected / len(impostor_attempts) * 100) if impostor_attempts else 100.0

        return {
            "username": username,
            "total_attempts": len(attempts),
            "genuine_attempts": len(genuine_attempts),
            "genuine_accepted": gen_accepted,
            "genuine_rejected": gen_rejected,
            "empirical_frr_percent": round(empirical_frr, 2),
            "impostor_attempts": len(impostor_attempts),
            "impostor_accepted": imp_accepted,
            "impostor_blocked": imp_rejected,
            "empirical_far_percent": round(empirical_far, 2),
            "defense_rate_percent": round(defense_rate, 2),
            "recent_attempts": attempts[-10:]
        }

    def preload_cmu_demo_users(self, count: int = 3) -> List[str]:
        """Preloads a few CMU benchmark subjects for immediate testing."""
        if not os.path.exists(DATA_FILE):
            return []

        import pandas as pd
        df = pd.read_csv(DATA_FILE)
        selected_subjects = ["s002", "s004", "s005"][:count]
        preloaded = []

        for subj in selected_subjects:
            if self.user_exists(subj):
                preloaded.append(subj)
                continue

            sub_df = df[df["subject"] == subj].iloc[:20]  # First 20 reps
            samples = sub_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
            mean_vec = np.mean(samples, axis=0).tolist()
            std_vec = np.std(samples, axis=0, ddof=1).tolist()

            # Train One-Class SVM as best model
            from ml.models import OneClassSVMDetector
            model = OneClassSVMDetector(nu=0.05, kernel="rbf", gamma="scale")
            model.fit(samples)
            self.save_model(subj, model)

            user_entry = {
                "username": subj,
                "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_count": len(samples),
                "model_type": model.name,
                "default_threshold": float(model.threshold),
                "mean_features": mean_vec,
                "std_features": std_vec,
                "is_demo_subject": True
            }
            self.save_user(subj, user_entry)
            preloaded.append(subj)

        return preloaded


# Singleton instance
storage = StorageManager()
