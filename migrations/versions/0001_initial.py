"""initial platform tables"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    submission_pool = sa.Enum("shallow", "medium", "final", "winner", "rejected", name="submissionpool")
    submission_pool.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "model_submissions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("hotkey", sa.String(length=255), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False, unique=True),
        sa.Column("model_code_url", sa.String(length=1024), nullable=True),
        sa.Column("submission_time", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("current_pool", submission_pool, nullable=False),
        sa.Column("current_score", sa.Float(), nullable=True),
        sa.Column("current_checkpoint_url", sa.String(length=1024), nullable=True),
        sa.Column("assigned_validator", sa.String(length=255), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=True),
    )
    op.create_index("ix_model_submissions_hotkey", "model_submissions", ["hotkey"])
    op.create_index("ix_model_submissions_current_pool", "model_submissions", ["current_pool"])
    op.create_index("ix_model_submissions_assigned_validator", "model_submissions", ["assigned_validator"])
    op.create_index("ix_model_submissions_lease_expires_at", "model_submissions", ["lease_expires_at"])

    op.create_table(
        "validator_heartbeats",
        sa.Column("hotkey", sa.String(length=255), primary_key=True),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("validator_heartbeats")
    op.drop_index("ix_model_submissions_current_pool", table_name="model_submissions")
    op.drop_index("ix_model_submissions_hotkey", table_name="model_submissions")
    op.drop_index("ix_model_submissions_lease_expires_at", table_name="model_submissions")
    op.drop_index("ix_model_submissions_assigned_validator", table_name="model_submissions")
    op.drop_table("model_submissions")
    sa.Enum(name="submissionpool").drop(op.get_bind(), checkfirst=True)
