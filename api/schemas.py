"""Pydantic schemas for the platform API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, Field, field_serializer

from .models import SubmissionPool


class MinerSubmitRequest(BaseModel):
    hotkey: str
    model_code_url: Optional[str] = None
    model_id: Optional[str] = Field(default=None, description="Stable identifier for the submission.")
    meta: Dict[str, Any] = Field(default_factory=dict)


class MinerSubmitResponse(BaseModel):
    submission_id: int
    model_id: str
    current_pool: SubmissionPool


class ValidatorRequestJobRequest(BaseModel):
    validator_hotkey: str
    capacity_hint_sec: Optional[int] = None


class TrainingJobResponse(BaseModel):
    submission_id: int
    model_id: str
    pool: SubmissionPool
    checkpoint_url: Optional[str] = None
    model_code_url: Optional[str] = None
    training_time_limit_sec: int
    lease_expires_at: Optional[datetime] = None

    @field_serializer("lease_expires_at")
    def _serialize_dt(self, value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        return value.isoformat()


class ValidatorSubmitResultsRequest(BaseModel):
    submission_id: int
    model_id: str
    new_score: float
    new_checkpoint_url: Optional[str] = None
    logs_url: Optional[str] = None
    validator_hotkey: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class HeartbeatRequest(BaseModel):
    validator_hotkey: str


class WeightsResponse(BaseModel):
    weights: Mapping[str, float]
