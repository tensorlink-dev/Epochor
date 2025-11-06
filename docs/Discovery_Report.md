# Discovery Report

## Repository Topology (Top 3 Levels)
- `/api`
  - FastAPI control plane: app factory (`main.py`), configuration (`config.py`), database helpers (`database.py`), ORM models (`models.py`), background scheduler (`competition_scheduler.py`), security utilities, and request routers under `routes/`.
- `/templates/validator_training`
  - Chutes-compatible validator training template (entry script, miner protocol shim, Hugging Face IO helpers, template builder).
- `/neurons`
  - Legacy miner/validator entrypoints plus refactored validator submodules (state, sandbox, evaluation, scoring, weight setter shim that now consumes API weights).
- `/migrations`
  - Alembic environment and versioned migrations for the platform database.
- `/docs`
  - Reference documentation, including this discovery report and the compatibility matrix.
- `/tests`
  - Pytest suite covering API flows (`tests/api`), training template behaviour (`tests/templates`), and historical unit tests for core libraries.
- `/epochor`, `/competitions`, `/constants`, `/scripts`
  - Pre-existing library packages, schedules, constants, and helper scripts retained from the legacy repo.

## Key Modules & Responsibilities
- `api/main.py`
  - FastAPI application factory that initialises the database and wires miner, validator, and scoring routers.
- `api/routes/miner.py`
  - `/miner/submit` endpoint enforcing hotkey allowlists, registering submissions, and resetting tournament state.
- `api/routes/validator.py`
  - `/validator/request-training-job`, `/validator/submit-results`, `/validator/heartbeat` endpoints handling lease orchestration, results ingestion, and liveness tracking.
- `api/routes/scoring.py`
  - `/scoring/weights` endpoint exposing deterministic subnet weight calculations.
- `api/competition_scheduler.py`
  - APScheduler-powered promotion cycle (shallow→medium→final→winner), stale lease reaper, and cleanup routines.
- `api/models.py`
  - SQLAlchemy models for submissions and heartbeats plus winner invariants.
- `api/config.py` & `api/security.py`
  - Centralised settings (Pydantic) and bearer token / hotkey allowlist enforcement.
- `templates/validator_training/trainer_entry.py`
  - Validator-owned training loop enforcing determinism, 1-hour cap, Hugging Face uploads, and Chutes environment contracts.
- `templates/validator_training/template_builder.py`
  - Factory for building pinned Chutes images with proper mounts and environment variables.
- `neurons/validator/weight_setter.py`
  - Background loop that now pulls weights from the API before setting them on-chain.
- `neurons/miner.py`
  - Legacy miner CLI augmented to auto-submit to the platform API when configured.

## Entrypoints, Configs, Environment
- CLI scripts
  - `python api/main.py` served via `uvicorn api.main:app` (FastAPI control plane).
  - `python neurons/validator.py` legacy validator orchestrator (still initialises Bittensor stack but can consume API URLs/tokens for weights).
  - `python neurons/miner.py` miner heartbeat loop with optional API auto-submit flags (`--platform_api_url`, `--platform_api_token`, `--model_id`).
- Configuration
  - API settings via environment variables prefixed `EPOCHOR_` (`database_url`, `api_token`, hotkey allowlists, lease durations, promotion thresholds, scheduler cadences).
  - Validator/miner CLI flags remain under `neurons/config.py`; new optional fields surfaced in validator state (`platform_api_url`, `platform_api_token`, sandbox overrides).
- Environment variables
  - `EPOCHOR_API_URL`, `EPOCHOR_API_TOKEN`, `EPOCHOR_MODEL_ID`, `EPOCHOR_MODEL_CODE_URL` for miners.
  - `HF_WRITE_TOKEN_ENV` consumed by template/HF uploads; defaults respected.
  - Scheduler/lease timings configurable via `EPOCHOR_*_POOL_LEASE_SECONDS`, `EPOCHOR_HEARTBEAT_TIMEOUT_SECONDS`.

## Training / Evaluation Boundaries
- Training harness now lives inside `templates/validator_training/trainer_entry.py` (Chutes template) and still invokes miner submissions via `MinerSubmissionProtocol`.
- Evaluation and scoring remain in `neurons/validator/evaluation_service.py` & `scoring_service.py`, but API now owns tournament lifecycle and weight computation.
- Hugging Face uploads delegated through `epochor.utils.hf_io` which the template re-exports; validator weight setter consumes API `/scoring/weights`.
- Sandbox/network restrictions enforced inside the template (network disabled by default, deterministic seeds, 1-hour cap) and by validator sandbox runtime config.

## Existing APIs / Servers / DB Layers
- FastAPI server exposing miner, validator, and scoring routes (see Key Modules).
- SQLAlchemy ORM with Alembic migrations stored under `/migrations`; `migrations/env.py` binds to `EPOCHOR_DATABASE_URL` for offline/online runs.
- APScheduler-based competition scheduler integrated in API module; reaper job handles stale leases.
- Weight setter fetches weights via REST before calling Bittensor `set_weights`.

## Validator State Management
- Legacy state persists via `neurons/validator/state.py`; still handles EMA tracker, UID queues, and disk cache.
- Validator weight setter now synchronises with API to align on winners/weights.
- Validator heartbeat endpoint persists `ValidatorHeartbeat` rows for lease reaping.

## Known Forks / Duplication Signals
- `templates/validator_training/hf_io.py` simply re-exports canonical `epochor.utils.hf_io` helpers to maintain backwards-compatible imports.
- `templates/validator_training/miner_protocol.py` subclasses the canonical protocol for template-local imports.
- Legacy validator orchestrator still contains older flow (metagraph sync, competition manager); API introduces new control plane but no duplicate scheduler inside validator code yet.

## Tests & Coverage Overview
- `tests/api/test_platform_api.py` covers miner submit, job leasing priority, results submission, lease reaping, promotion chain, concurrency guards, and weight aggregation.
- `tests/templates/test_trainer_entry.py` validates determinism, time cap, protocol enforcement, resume, checkpoint hashing, and HF upload scaffolding.
- Historical tests under `tests/` remain untouched (datasets, disk stores, validator utilities) but rely on torch/hf dependencies.
- No coverage reports committed; pytest remains the primary runner.

## "Do Not Break" Surfaces
- Public import paths (`templates.validator_training.miner_protocol.MinerSubmissionProtocol`, `neurons.validator.weight_setter.WeightSetter`, `api.schemas.*`).
- REST endpoints contracts (payload schemas and status codes) for miners/validators/scoring.
- Alembic migration history (`0001_initial`) and schema invariants (unique `model_id`, single winner enforcement).
- Hugging Face IO utilities (`epochor.utils.hf_io`) and template environment variables (`SUBMISSION_DIR`, `ARTIFACTS_DIR`).
- Validator/miner CLI flags and configuration surfaces inherited from legacy code.
