"""Scoring-related API routes."""
from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import ModelSubmission, SubmissionPool
from ..schemas import WeightsResponse
from ..security import verify_bearer_token

router = APIRouter(prefix="/scoring", tags=["scoring"], dependencies=[Depends(verify_bearer_token)])


def _inverse_score(score: float) -> float:
    if score <= 0:
        return 1.0
    return 1.0 / score


def _proportional_weights(submissions: list[ModelSubmission]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for submission in submissions:
        if submission.current_score is None:
            continue
        values[submission.hotkey] = _inverse_score(submission.current_score)
    total = sum(values.values())
    if total <= 0:
        return {}
    return {hotkey: value / total for hotkey, value in values.items()}


@router.get("/weights", response_model=WeightsResponse)
def get_weights(session: Session = Depends(get_session)) -> WeightsResponse:
    """Compute subnet weights across pools."""

    winner_stmt = select(ModelSubmission).where(ModelSubmission.current_pool == SubmissionPool.WINNER).order_by(ModelSubmission.current_score)
    winner = session.execute(winner_stmt).scalars().first()

    final_stmt = select(ModelSubmission).where(ModelSubmission.current_pool == SubmissionPool.FINAL)
    final_submissions = session.execute(final_stmt).scalars().all()
    medium_stmt = select(ModelSubmission).where(ModelSubmission.current_pool == SubmissionPool.MEDIUM)
    medium_submissions = session.execute(medium_stmt).scalars().all()

    weights: Dict[str, float] = {}
    share_winner = 0.7 if winner else 0.0
    final_weights = _proportional_weights([s for s in final_submissions if not winner or s.id != winner.id])
    share_final = 0.2 if final_weights else 0.0
    medium_weights = _proportional_weights(medium_submissions)
    share_medium = 0.1 if medium_weights else 0.0

    assigned = share_winner + share_final + share_medium
    leftover = max(0.0, 1.0 - assigned)

    if winner:
        weights[winner.hotkey] = weights.get(winner.hotkey, 0.0) + share_winner
    for hotkey, fraction in final_weights.items():
        weights[hotkey] = weights.get(hotkey, 0.0) + share_final * fraction
    for hotkey, fraction in medium_weights.items():
        weights[hotkey] = weights.get(hotkey, 0.0) + share_medium * fraction

    if leftover > 0:
        if winner:
            weights[winner.hotkey] = weights.get(winner.hotkey, 0.0) + leftover
        elif final_weights:
            for hotkey, fraction in final_weights.items():
                weights[hotkey] = weights.get(hotkey, 0.0) + leftover * fraction
        elif medium_weights:
            for hotkey, fraction in medium_weights.items():
                weights[hotkey] = weights.get(hotkey, 0.0) + leftover * fraction

    return WeightsResponse(weights=weights)


__all__ = ["router"]
