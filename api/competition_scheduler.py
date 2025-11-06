"""APScheduler coordinator for pool promotions and cleanup."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .database import session_scope
from .models import ModelSubmission, SubmissionPool, ValidatorHeartbeat, enforce_single_winner


def _sorted_by_score(submissions: Iterable[ModelSubmission]) -> list[ModelSubmission]:
    return sorted(
        submissions,
        key=lambda item: (item.current_score is None, item.current_score, item.submission_time),
    )


def promote_to_pool(
    session: Session,
    submissions: Sequence[ModelSubmission],
    target_pool: SubmissionPool,
) -> None:
    for submission in submissions:
        submission.current_pool = target_pool
        submission.release_lease()


def promote_to_medium(session: Session, settings: Settings) -> None:
    threshold = settings.shallow_pool_promotion_threshold
    stmt = select(ModelSubmission).where(ModelSubmission.current_pool == SubmissionPool.SHALLOW)
    candidates = _sorted_by_score(session.execute(stmt).scalars())
    eligible = [s for s in candidates if s.current_score is not None and s.current_score <= threshold] if threshold > 0 else [s for s in candidates if s.current_score is not None]
    promotions = eligible[: settings.max_medium_promotions]
    promote_to_pool(session, promotions, SubmissionPool.MEDIUM)


def promote_to_final(session: Session, settings: Settings) -> None:
    threshold = settings.medium_pool_promotion_threshold
    stmt = select(ModelSubmission).where(ModelSubmission.current_pool == SubmissionPool.MEDIUM)
    candidates = _sorted_by_score(session.execute(stmt).scalars())
    eligible = [s for s in candidates if s.current_score is not None and s.current_score <= threshold] if threshold > 0 else [s for s in candidates if s.current_score is not None]
    promotions = eligible[: settings.max_final_promotions]
    promote_to_pool(session, promotions, SubmissionPool.FINAL)


def select_winner(session: Session) -> None:
    stmt = select(ModelSubmission).where(ModelSubmission.current_pool == SubmissionPool.FINAL)
    finalists = _sorted_by_score(session.execute(stmt).scalars())
    if not finalists:
        return
    winner = finalists[0]
    winner.current_pool = SubmissionPool.WINNER
    enforce_single_winner(session, winner)


def cleanup_rejected(session: Session) -> None:
    stmt = select(ModelSubmission).where(
        ModelSubmission.current_pool.in_([SubmissionPool.SHALLOW, SubmissionPool.MEDIUM, SubmissionPool.FINAL]),
        ModelSubmission.current_score.is_(None),
    )
    for submission in session.execute(stmt).scalars():
        if submission.current_score is None:
            submission.current_pool = SubmissionPool.REJECTED
            submission.release_lease()


def release_stale_leases(session: Session, settings: Settings) -> None:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=settings.heartbeat_timeout_seconds)
    stale_validators = set(
        session.execute(
            select(ValidatorHeartbeat.hotkey).where(ValidatorHeartbeat.last_heartbeat < cutoff)
        ).scalars()
    )
    stmt = select(ModelSubmission).where(ModelSubmission.assigned_validator.is_not(None))
    for submission in session.execute(stmt).scalars():
        if submission.lease_expires_at and submission.lease_expires_at < now:
            submission.release_lease()
            continue
        if submission.assigned_validator and submission.assigned_validator in stale_validators:
            submission.release_lease()


class CompetitionScheduler:
    """Coordinates daily promotions across pools."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.scheduler = AsyncIOScheduler()
        self._configure_jobs()

    def _configure_jobs(self) -> None:
        self.scheduler.add_job(self.run_daily_cycle, CronTrigger(hour=0, minute=0))
        self.scheduler.add_job(self.reap_stale_leases, IntervalTrigger(minutes=5))

    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def run_daily_cycle(self) -> None:
        with session_scope() as session:
            promote_to_medium(session, self.settings)
            promote_to_final(session, self.settings)
            select_winner(session)
            cleanup_rejected(session)

    def reap_stale_leases(self) -> None:
        with session_scope() as session:
            release_stale_leases(session, self.settings)


__all__ = [
    "CompetitionScheduler",
    "promote_to_medium",
    "promote_to_final",
    "select_winner",
    "cleanup_rejected",
    "release_stale_leases",
]
