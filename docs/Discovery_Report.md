# Discovery Report

## Repository Topology (Top 3 Levels)
- `/competitions`
  - Schedules and competition metadata (`competitions.py`, `epsilon.py`).
- `/constants`
  - Global defaults, runtime constants, environment toggles.
- `/docs`
  - Existing run-books for staging/testnet/mainnet and stream tutorial assets.
- `/epochor`
  - Core library package: configs, datasets, generators, training/evaluation utilities, model stores, helpers.
- `/neurons`
  - CLI entrypoints (`miner.py`, `validator.py`) and validator submodules (`validator/`).
- `/scripts`
  - Shell helpers for dependency compatibility/install.
- `/tests`
  - Pytest suite for validators, datasets, disk/HF stores, model utilities.

## Key Modules & Responsibilities
- `epochor/training/validator_runner.py`
  - Validator-owned training loop (`run_training`, `load_miner_module`).
- `epochor/evaluation/evaluation.py`
  - Score calculation helpers (CRPS, EMA smoothing integration).
- `epochor/model/*`
  - `storage/` backends (disk, HF, metadata via chain), model tracker/updater.
- `neurons/validator/*`
  - `state.py`: persistent validator state, EMA tracker, UID queues.
  - `model_manager.py`: fetch/update miner submissions, Hugging Face sync.
  - `evaluation_service.py`: sandbox execution, artifact uploads, scoring payloads.
  - `scoring_service.py`: transforms scores into weight updates.
  - `competition_manager.py`: rotates competitions, prepares data batches.
  - `sandbox.py`: wraps submission execution with resource limits.
  - `weight_setter.py`: pushes weights on-chain via subtensor.
- `neurons/config.py`
  - Shared argparse-based CLI configuration for miner/validator entrypoints.

## Entrypoints, Configs, Environment
- CLI scripts
  - `python neurons/validator.py` – main validator loop; accepts Bittensor wallet/subtensor flags plus validator-specific args (`--model_dir`, `--sandbox_*`, etc.).
  - `python neurons/miner.py` – lightweight miner heartbeat; relies on environment `TOKENIZERS_PARALLELISM=true`.
- Configuration
  - CLI flags via `neurons/config.py` (base + validator-specific arguments).
  - Constants in `constants/__init__.py` (WANDB project, cadences, EMA alpha, etc.).
  - `.env.example` for WANDB/HF tokens (loaded indirectly in model stores via `dotenv`).
- Environment variables in active use
  - `IS_LOCAL_DEVELOPMENT_MODE`, `LOCAL_MODE_NEURONS_COUNT` for development behavior.
  - `TOKENIZERS_PARALLELISM` toggled in miner.
  - Hugging Face token expected via `HF_WRITE_TOKEN_ENV` (in `epochor/utils/hf_io.py`).

## Training / Evaluation Boundaries
- Training harness: `epochor/training/validator_runner.py` (interfaces with miner submissions through `MinerSubmissionProtocol`).
- Sandbox invocation: `neurons/validator/sandbox.py` (wraps training runner inside containerized execution, enforces limits).
- Evaluation scoring: `epochor/evaluation/evaluation.py` and `neurons/validator/scoring_service.py` (CRPS, EMA accumulation, weight computation).
- Storage boundaries: `epochor/model/storage/hf_model_store.py`, `disk_model_store.py`, `metadata_model_store.py` (HF uploads, disk caching, chain metadata).

## Existing APIs / Servers / DB Layers
- No FastAPI/HTTP servers present.
- No ORM or persistent DB models; storage relies on disk artifacts plus chain metadata.
- Scheduling handled in-process via `CompetitionManager` and timer loops; no external job queues.
- Weight setting handled directly through `WeightSetter` interacting with Bittensor subtensor.

## Validator State Management
- `neurons/validator/state.py`
  - Pickle/JSON-backed persistence of model tracker, competition EMA tracker, UID queues.
  - State directory computed from `config.model_dir / "vali-state"`.
  - `ModelTracker` (in `epochor/model/model_tracker.py`) stores miner submissions & checkpoints.

## Known Forks / Duplication Signals
- Storage implementations under `epochor/model/storage/` include disk + Hugging Face variants; no apparent duplicates.
- No parallel validator implementations detected; single canonical orchestrator in `neurons/validator.py` with modular subcomponents.
- No existing API server stubs or alternative scheduler pipelines.

## Tests & Coverage Overview
- Pytest suite under `/tests` covers:
  - Dataset generation utilities (`test_data_generator.py`).
  - Disk/HF model store behavior (`test_disk_model_store.py`).
  - Model tracker utilities (`test_model_tracker.py`).
  - Validator orchestration pieces (`test_validator.py`, `test_validator_training.py`).
  - Misc helpers (`test_disk_utils.py`, `test_model_utils.py`).
- No coverage reports committed; default `pytest` runner.

## "Do Not Break" Surfaces
- Public imports from `neurons` package (validator/miner entrypoints) and `epochor.training.validator_runner`.
- CLI flags defined in `neurons/config.py` and inherited Bittensor CLI arguments.
- Competition schedule data contract (`competitions/competitions.py`).
- Miner submission protocol defined in `epochor/training/validator_contract.py`.
- Model tracker storage schema (`epochor/model/model_tracker.py`).
- EMA tracker & weighting pipeline (`epochor/validation/ema_tracker.py`, `neurons/validator/weight_setter.py`).
- Subtensor interaction contract for weights (configurable cadences, wallet requirements).
