"""SQLAlchemy ORM models for the platform API."""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import JSON, DateTime, Enum, Float, Integer, String, UniqueConstraint, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from .database import Base


class SubmissionPool(str, enum.Enum):
    SHALLOW = "shallow"
    MEDIUM = "medium"
    FINAL = "final"
    WINNER = "winner"
    REJECTED = "rejected"


class ModelSubmission(Base):
    """Miner model submission tracked across the tournament pools."""

    __tablename__ = "model_submissions"
    __table_args__ = (
        UniqueConstraint("model_id", name="uq_model_submissions_model_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hotkey: Mapped[str] = mapped_column(String(255), index=True)
    model_id: Mapped[str] = mapped_column(String(255), unique=True)
    model_code_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    submission_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    current_pool: Mapped[SubmissionPool] = mapped_column(
        Enum(SubmissionPool), index=True, default=SubmissionPool.SHALLOW
    )
    current_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_checkpoint_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    assigned_validator: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    meta: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    def release_lease(self) -> None:
        """Clear any active lease for the submission."""

        self.assigned_validator = None
        self.lease_expires_at = None

    @property
    def lease_active(self) -> bool:
        if not (self.assigned_validator and self.lease_expires_at):
            return False
        now = datetime.now(timezone.utc)
        return self.lease_expires_at > now


class ValidatorHeartbeat(Base):
    """Tracks validator heartbeat timestamps."""

    __tablename__ = "validator_heartbeats"

    hotkey: Mapped[str] = mapped_column(String(255), primary_key=True)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


def enforce_single_winner(session: Session, winner: ModelSubmission) -> None:
    """Ensure only one submission is marked as winner."""

    stmt = select(ModelSubmission).where(
        ModelSubmission.current_pool == SubmissionPool.WINNER,
        ModelSubmission.id != winner.id,
    )
    for row in session.execute(stmt).scalars().all():
        row.current_pool = SubmissionPool.REJECTED
        row.release_lease()


__all__ = [
    "ModelSubmission",
    "ValidatorHeartbeat",
    "SubmissionPool",
    "enforce_single_winner",
]
