"""Validator-facing API routes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import or_, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..database import get_session
from ..models import ModelSubmission, SubmissionPool, ValidatorHeartbeat
from ..schemas import (
    HeartbeatRequest,
    TrainingJobResponse,
    ValidatorRequestJobRequest,
    ValidatorSubmitResultsRequest,
)
from ..security import enforce_validator_hotkey, verify_bearer_token

router = APIRouter(prefix="/validator", tags=["validator"], dependencies=[Depends(verify_bearer_token)])


def _lease_duration_for_pool(settings: Settings, pool: SubmissionPool) -> int:
    if pool == SubmissionPool.FINAL:
        return settings.final_pool_lease_seconds
    if pool == SubmissionPool.MEDIUM:
        return settings.medium_pool_lease_seconds
    return settings.shallow_pool_lease_seconds


def _lease_candidate(
    session: Session,
    pool: SubmissionPool,
    validator_hotkey: str,
    lease_seconds: int,
) -> Optional[ModelSubmission]:
    now = datetime.now(timezone.utc)
    stmt_base = (
        select(ModelSubmission)
        .where(
            ModelSubmission.current_pool == pool,
            ModelSubmission.current_pool.notin_([SubmissionPool.WINNER, SubmissionPool.REJECTED]),
            or_(
                ModelSubmission.assigned_validator.is_(None),
                ModelSubmission.lease_expires_at <= now,
                ModelSubmission.assigned_validator == validator_hotkey,
            ),
        )
        .order_by(
            ModelSubmission.current_score.is_(None),
            ModelSubmission.current_score,
            ModelSubmission.submission_time,
        )
        .limit(1)
    )

    stmt = stmt_base
    try:
        stmt = stmt.with_for_update(skip_locked=True)
    except AttributeError:  # SQLAlchemy <2.0 compatibility
        pass

    try:
        candidate = session.execute(stmt).scalars().first()
    except DBAPIError:
        # Some SQLite builds do not support FOR UPDATE; retry without the hint.
        candidate = session.execute(stmt_base).scalars().first()
    if candidate is None:
        return None
    candidate.assigned_validator = validator_hotkey
    candidate.lease_expires_at = now + timedelta(seconds=lease_seconds)
    candidate.last_updated = now
    session.flush()
    return candidate


@router.post("/request-training-job", response_model=TrainingJobResponse, status_code=status.HTTP_200_OK)
def request_training_job(
    payload: ValidatorRequestJobRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> TrainingJobResponse | Response:
    """Lease the highest-priority submission available."""

    enforce_validator_hotkey(payload.validator_hotkey, settings)
    for pool in (SubmissionPool.FINAL, SubmissionPool.MEDIUM, SubmissionPool.SHALLOW):
        lease_seconds = _lease_duration_for_pool(settings, pool)
        candidate = _lease_candidate(session, pool, payload.validator_hotkey, lease_seconds)
        if candidate is not None:
            if payload.capacity_hint_sec is not None:
                limit = min(payload.capacity_hint_sec, lease_seconds)
            else:
                limit = lease_seconds
            return TrainingJobResponse(
                submission_id=candidate.id,
                model_id=candidate.model_id,
                pool=candidate.current_pool,
                checkpoint_url=candidate.current_checkpoint_url,
                model_code_url=candidate.model_code_url,
                training_time_limit_sec=limit,
                lease_expires_at=candidate.lease_expires_at,
            )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/submit-results", status_code=status.HTTP_200_OK)
def submit_results(
    payload: ValidatorSubmitResultsRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Persist validator results and release the lease."""

    enforce_validator_hotkey(payload.validator_hotkey, settings)
    submission = session.get(ModelSubmission, payload.submission_id)
    if submission is None or submission.model_id != payload.model_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")
    if submission.assigned_validator != payload.validator_hotkey:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Validator does not hold the lease")
    now = datetime.now(timezone.utc)
    if submission.lease_expires_at and submission.lease_expires_at < now:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Lease expired")

    submission.current_score = payload.new_score
    if payload.new_checkpoint_url:
        submission.current_checkpoint_url = payload.new_checkpoint_url
    meta = dict(submission.meta or {})
    meta.update(payload.meta)
    if payload.logs_url:
        meta["logs_url"] = payload.logs_url
    submission.meta = meta
    submission.last_updated = now
    submission.release_lease()
    session.flush()
    return {"status": "ok"}


@router.post("/heartbeat", status_code=status.HTTP_200_OK)
def heartbeat(
    payload: HeartbeatRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Record the validator heartbeat."""

    enforce_validator_hotkey(payload.validator_hotkey, settings)
    now = datetime.now(timezone.utc)
    record = session.get(ValidatorHeartbeat, payload.validator_hotkey)
    if record is None:
        record = ValidatorHeartbeat(hotkey=payload.validator_hotkey, last_heartbeat=now)
        session.add(record)
    else:
        record.last_heartbeat = now
    session.flush()
    return {"status": "ok"}


__all__ = ["router"]
