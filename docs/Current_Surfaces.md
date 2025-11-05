# Current Public Surfaces

## Python Imports
- `neurons.validator.Validator`
  - Entry orchestrator used by validators; imports submodules `ValidatorState`, `ModelManager`, etc.
- `neurons.miner`
  - Heartbeat script; exposes `main()` executed under `if __name__ == "__main__"`.
- `neurons.config.validator_config()` / `miner_config()`
  - Shared argparse-based configuration factories. External callers expect identical flag names.
- `epochor.training.validator_runner`
  - `run_training`, `load_miner_module`, and `TrainingSummary` imported by sandbox and tests.
- `epochor.training.validator_contract.MinerSubmissionProtocol`
  - Structural contract miners implement; used throughout validator pipeline.
- `epochor.validation.validation.ValidationService`
  - Consumed by validator scoring; external code may reference CRPS helpers.
- `epochor.model.storage.*`
  - `DiskModelStore`, `HuggingFaceModelStore`, `ChainModelMetadataStore` used by validator services/tests.
- `epochor.model.model_tracker.ModelTracker`
  - Persisted state for miner submissions.

## CLI & Configuration
- Validator CLI flags (via `neurons/config.py`):
  - `--device`, `--wandb.off`, `--wandb_project`, `--offline`, `--netuid`.
  - Validator-only: `--blocks_per_epoch`, `--sample_min`, `--updated_models_limit`, `--dont_set_weights`, `--model_dir`, `--sandbox_image`, `--sandbox_timeout`, `--sandbox_memory`, `--sandbox_cpus`, `--sandbox_gpus`.
  - Plus all Bittensor-provided flags (wallet, subtensor, logging, axon).
- Miner CLI uses base flags (device, wandb toggles, offline, netuid) + Bittensor standard args.
- Environment expectations:
  - `HF_WRITE_TOKEN_ENV` (from `epochor/utils/hf_io.py`), `.env` support via `dotenv`.
  - `IS_LOCAL_DEVELOPMENT_MODE`, `LOCAL_MODE_NEURONS_COUNT` recognized in `constants`.
  - `TOKENIZERS_PARALLELISM` set within miner startup.

## Data & Protocol Schemas
- Competition schedule contract (`competitions/competitions.py`): `Competition` objects with `eval_tasks`, dataset IDs, weighting metadata.
- Validator state persistence (`neurons/validator/state.py`): file layout (`model_tracker.pickle`, `competition_tracker.json`, `uids.pickle`, `version.txt`).
- Model tracker entries (`epochor/model/model_tracker.py`): `MinerSubmissionSnapshot` objects keyed by hotkey.
- Sandbox inputs/outputs (`neurons/validator/sandbox.py`): expects `run_submission_in_sandbox` to return summary dict with `train_metrics`, `val_metrics`, `artifacts_dir`, `stdout`, `stderr`.
- Weight setting payloads computed via `neurons/validator/scoring_service.py` and emitted to `WeightSetter`.

## Tests Referencing Surfaces
- `tests/test_validator.py` imports validator orchestrator and state helpers.
- `tests/test_validator_training.py` covers `run_training` contract and sandbox invocation.
- `tests/test_model_tracker.py` expects persistent schema invariants.
- `tests/test_disk_model_store.py` relies on disk storage layout.
- `tests/test_model_utils.py` references helper APIs under `epochor/utils`.

## External Integrations
- Bittensor client (`bt.*`) usage in validator/miner configuration and weight setting.
- Hugging Face Hub uploads via `epochor/utils/hf_io.py` and `epochor/model/storage/hf_model_store.py`.
- Weights & Biases initialization triggered in `neurons/validator.Validator` when `wandb.on` true.
