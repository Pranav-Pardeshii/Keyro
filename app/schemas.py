"""
Pydantic schemas for the Keyro REST API.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class KeyEvent(BaseModel):
    key: str
    down_time: float = Field(..., description="Timestamp of keydown in seconds")
    up_time: float = Field(..., description="Timestamp of keyup in seconds")


class EnrollRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    samples: List[List[float]] = Field(..., description="List of 20 31-dimensional feature vectors")
    model_type: Optional[str] = Field(default=None, description="Optional model override")


class EnrollResponse(BaseModel):
    status: str
    username: str
    model_type: str
    sample_count: int
    threshold: float
    mean_features: List[float]
    message: str


class VerifyRequest(BaseModel):
    username: str
    features: Optional[List[float]] = Field(default=None, description="31 timing features")
    raw_events: Optional[List[KeyEvent]] = Field(default=None, description="Raw key event timestamps")
    threshold: Optional[float] = Field(default=None, description="Custom threshold from slider")
    is_impostor_attempt: bool = Field(default=False, description="Flag for friend impersonation mode")
    tester_label: Optional[str] = Field(default=None, description="Identifier of the tester / friend")


class VerifyResponse(BaseModel):
    result: str  # "ACCEPT" or "REJECT"
    is_accepted: bool
    score: float
    threshold: float
    distance_margin: float
    confidence_percent: float
    username: str
    is_impostor_attempt: bool
    tester_label: Optional[str]
    timestamp: str
    top_deviations: Optional[List[Dict[str, Any]]] = None


class UserSummary(BaseModel):
    username: str
    enrolled_at: str
    sample_count: int
    model_type: str
    default_threshold: float
    total_verifications: int
    genuine_attempts: int
    impostor_attempts: int


class ThresholdTradeoffResponse(BaseModel):
    username: str
    current_threshold: float
    tradeoffs: List[Dict[str, float]]
