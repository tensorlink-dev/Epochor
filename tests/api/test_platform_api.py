from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api import config
from api.competition_scheduler import (
    cleanup_rejected,
    promote_to_final,
    promote_to_medium,
    release_stale_leases,
    select_winner,
)
from api.database import configure_engine, init_db, session_scope
from api.main import create_app
from api.models import ModelSubmission, SubmissionPool


@pytest.fixture
def settings(tmp_path, monkeypatch):
    config.get_settings.cache_clear()
    database_url = f"sqlite:///{tmp_path/'platform.db'}"
    settings = config.Settings(
        database_url=database_url,
        api_token=None,
        allowed_miner_hotkeys=["miner-a", "miner-b"],
        allowed_validator_hotkeys=["validator-a", "validator-b"],
        shallow_pool_promotion_threshold=1.0,
        medium_pool_promotion_threshold=0.5,
        max_medium_promotions=10,
        max_final_promotions=10,
        heartbeat_timeout_seconds=60,
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    configure_engine(database_url)
    init_db()
    return settings


@pytest.fixture
def client(settings):
    app = create_app()
    return TestClient(app)


def _count_assignments() -> int:
    with session_scope() as session:
        return (
            session.query(ModelSubmission)
            .filter(ModelSubmission.assigned_validator.isnot(None))
            .count()
        )


def test_miner_submit_creates_submission(client):
    response = client.post(
        "/miner/submit",
        json={"hotkey": "miner-a", "model_code_url": "https://hf/alpha", "meta": {"notes": "demo"}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["current_pool"] == "shallow"

    with session_scope() as session:
        submission = session.get(ModelSubmission, payload["submission_id"])
        assert submission is not None
        assert submission.hotkey == "miner-a"
        assert submission.model_code_url == "https://hf/alpha"
        assert submission.meta["notes"] == "demo"


def _seed_submission(**kwargs) -> ModelSubmission:
    submission = ModelSubmission(**kwargs)
    with session_scope() as session:
        session.add(submission)
        session.flush()
        session.refresh(submission)
    return submission


def test_validator_request_prioritizes_pools(client):
    final_sub = _seed_submission(
        hotkey="miner-b",
        model_id="final-1",
        model_code_url="https://hf/final",
        current_pool=SubmissionPool.FINAL,
        current_score=0.2,
    )
    shallow_sub = _seed_submission(
        hotkey="miner-a",
        model_id="shallow-1",
        model_code_url="https://hf/shallow",
        current_pool=SubmissionPool.SHALLOW,
    )

    response = client.post(
        "/validator/request-training-job",
        json={"validator_hotkey": "validator-a"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["submission_id"] == final_sub.id
    assert payload["pool"] == "final"

    response = client.post(
        "/validator/request-training-job",
        json={"validator_hotkey": "validator-a"},
    )
    assert response.status_code == 200
    assert response.json()["submission_id"] == shallow_sub.id


def test_submit_results_releases_lease(client):
    submission = _seed_submission(
        hotkey="miner-a",
        model_id="mid-1",
        current_pool=SubmissionPool.MEDIUM,
    )
    job = client.post(
        "/validator/request-training-job",
        json={"validator_hotkey": "validator-a"},
    ).json()
    assert job["submission_id"] == submission.id

    response = client.post(
        "/validator/submit-results",
        json={
            "submission_id": submission.id,
            "model_id": submission.model_id,
            "new_score": 0.25,
            "new_checkpoint_url": "https://hf/checkpoint",
            "validator_hotkey": "validator-a",
        },
    )
    assert response.status_code == 200
    with session_scope() as session:
        refreshed = session.get(ModelSubmission, submission.id)
        assert refreshed.current_score == pytest.approx(0.25)
        assert refreshed.assigned_validator is None
        assert refreshed.current_checkpoint_url == "https://hf/checkpoint"


def test_release_stale_leases(client, settings):
    submission = _seed_submission(
        hotkey="miner-b",
        model_id="lease-1",
        current_pool=SubmissionPool.MEDIUM,
        assigned_validator="validator-a",
        lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    with session_scope() as session:
        release_stale_leases(session, settings)
        refreshed = session.get(ModelSubmission, submission.id)
        assert refreshed.assigned_validator is None


def test_weights_sum_to_one(client):
    winner = _seed_submission(
        hotkey="miner-a",
        model_id="winner",
        current_pool=SubmissionPool.WINNER,
        current_score=0.1,
    )
    final_other = _seed_submission(
        hotkey="miner-b",
        model_id="final-other",
        current_pool=SubmissionPool.FINAL,
        current_score=0.2,
    )
    medium = _seed_submission(
        hotkey="miner-c",
        model_id="medium",
        current_pool=SubmissionPool.MEDIUM,
        current_score=0.5,
    )
    response = client.get("/scoring/weights")
    assert response.status_code == 200
    weights = response.json()["weights"]
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights[winner.hotkey] > weights[final_other.hotkey]


def test_promotion_flow(settings):
    shallow = _seed_submission(
        hotkey="miner-a",
        model_id="shallow",
        current_pool=SubmissionPool.SHALLOW,
        current_score=0.8,
    )
    medium = _seed_submission(
        hotkey="miner-b",
        model_id="medium",
        current_pool=SubmissionPool.MEDIUM,
        current_score=0.4,
    )
    with session_scope() as session:
        promote_to_medium(session, settings)
        promote_to_final(session, settings)
        select_winner(session)
        cleanup_rejected(session)
        session.flush()
        refreshed_shallow = session.get(ModelSubmission, shallow.id)
        refreshed_medium = session.get(ModelSubmission, medium.id)
    assert refreshed_shallow.current_pool in {SubmissionPool.MEDIUM, SubmissionPool.REJECTED}
    assert refreshed_medium.current_pool in {SubmissionPool.FINAL, SubmissionPool.WINNER}


def test_concurrent_request_prevents_double_assignment(client):
    submission = _seed_submission(
        hotkey="miner-a",
        model_id="concurrent",
        current_pool=SubmissionPool.SHALLOW,
    )
    first = client.post(
        "/validator/request-training-job",
        json={"validator_hotkey": "validator-a"},
    )
    second = client.post(
        "/validator/request-training-job",
        json={"validator_hotkey": "validator-b"},
    )
    assigned_count = _count_assignments()
    assert assigned_count == 1
    assert first.status_code == 200
    assert second.status_code in {200, 204}


def test_end_to_end_cycle(client, settings):
    submit_resp = client.post(
        "/miner/submit",
        json={"hotkey": "miner-a", "model_code_url": "https://hf/new"},
    ).json()
    job = client.post(
        "/validator/request-training-job",
        json={"validator_hotkey": "validator-a", "capacity_hint_sec": 1800},
    ).json()
    assert job["submission_id"] == submit_resp["submission_id"]

    client.post(
        "/validator/submit-results",
        json={
            "submission_id": job["submission_id"],
            "model_id": job["model_id"],
            "new_score": 0.4,
            "new_checkpoint_url": "https://hf/out",
            "validator_hotkey": "validator-a",
        },
    )

    with session_scope() as session:
        promote_to_medium(session, settings)
        promote_to_final(session, settings)
        select_winner(session)

    weights = client.get("/scoring/weights").json()["weights"]
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
