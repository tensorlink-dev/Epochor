"""Seed the platform database with development data."""
from __future__ import annotations

from datetime import datetime, timezone

from api.config import get_settings
from api.database import init_db, session_scope
from api.models import ModelSubmission, SubmissionPool


def main() -> None:
    settings = get_settings()
    init_db()
    with session_scope() as session:
        now = datetime.now(timezone.utc)
        examples = [
            ModelSubmission(
                hotkey="miner-alpha",
                model_id="alpha",
                model_code_url="https://huggingface.co/tensor-link/alpha",
                current_pool=SubmissionPool.SHALLOW,
                submission_time=now,
                meta={"notes": "seed"},
            ),
            ModelSubmission(
                hotkey="miner-beta",
                model_id="beta",
                model_code_url="https://huggingface.co/tensor-link/beta",
                current_pool=SubmissionPool.MEDIUM,
                current_score=0.3,
                submission_time=now,
            ),
        ]
        for item in examples:
            session.merge(item)
        session.flush()
        print(f"Seeded {len(examples)} submissions into {settings.database_url}")


if __name__ == "__main__":
    main()
