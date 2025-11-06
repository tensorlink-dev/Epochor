"""Miner-facing API routes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..database import get_session
from ..models import ModelSubmission, SubmissionPool
from ..schemas import MinerSubmitRequest, MinerSubmitResponse
from ..security import enforce_miner_hotkey, verify_bearer_token

router = APIRouter(prefix="/miner", tags=["miner"], dependencies=[Depends(verify_bearer_token)])


def _resolve_model_id(requested: Optional[str]) -> str:
    if requested:
        return requested
    return f"submission-{uuid4().hex[:10]}"


@router.post("/submit", response_model=MinerSubmitResponse)
def submit_model(
    payload: MinerSubmitRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> MinerSubmitResponse:
    """Register or refresh a miner submission."""

    enforce_miner_hotkey(payload.hotkey, settings)
    model_id = _resolve_model_id(payload.model_id)

    stmt = select(ModelSubmission).where(ModelSubmission.model_id == model_id)
    submission = session.execute(stmt).scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if submission:
        submission.hotkey = payload.hotkey
        submission.model_code_url = payload.model_code_url
        submission.current_pool = SubmissionPool.SHALLOW
        submission.current_score = None
        submission.current_checkpoint_url = payload.meta.get("checkpoint_url")
        submission.meta = {**(submission.meta or {}), **payload.meta}
        submission.release_lease()
        submission.submission_time = now
    else:
        submission = ModelSubmission(
            hotkey=payload.hotkey,
            model_id=model_id,
            model_code_url=payload.model_code_url,
            current_pool=SubmissionPool.SHALLOW,
            current_checkpoint_url=payload.meta.get("checkpoint_url"),
            meta=payload.meta,
        )
        submission.submission_time = now
        session.add(submission)
        session.flush()

    session.flush()
    return MinerSubmitResponse(
        submission_id=submission.id,
        model_id=submission.model_id,
        current_pool=submission.current_pool,
    )


__all__ = ["router"]
