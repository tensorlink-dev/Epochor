# Compatibility Matrix

| Existing Surface | Planned Successor / Wrapper | Strategy | Notes |
|------------------|-----------------------------|----------|-------|
| `neurons.validator.Validator` main loop | `neurons/validator.py` (refactored to call API client) | Maintain file & class; inject API adapter that preserves CLI flags while delegating job leasing to new HTTP client. | Add deprecation warnings for any new flags; keep existing configuration contract. |
| `neurons/validator/competition_manager.py` | `api/competition_scheduler.py` + thin shim in validator module | Extract scheduling logic into centralized API scheduler; keep current module exporting wrappers that call API or reuse legacy logic when API disabled. | Preserve current functions for backward compatibility; mark internal methods as `@deprecated` once API endpoints stable. |
| `neurons/validator/weight_setter.py` | Same module using `api/routes/scoring.py` | Update weight setter to fetch weights from API; keep existing `WeightSetter` class but delegate to HTTP client when remote mode enabled. | Provide fallback to legacy scoring for offline mode. |
| `neurons/miner.py` heartbeat | New `api/routes/miner.py` | Continue exposing same CLI; miner submits via REST client instead of direct disk/HF operations. | Ensure legacy submission script path still honored; add shim for API submission payloads. |
| `epochor/model/model_tracker.py` & state pickles | Database-backed `api/models.py` (`ModelSubmission`) | Keep tracker for local cache; add migration layer syncing DB state. | Provide import shim exporting `ModelSubmissionSnapshot` for backward compatibility. |
| `epochor/training/validator_runner.run_training` | Unchanged | Reuse existing training runner invoked by validators after leasing jobs. | No interface changes anticipated. |
| `competitions/competitions.py` schedule | API scheduler tables | Mirror schedule data into database with daily promotion jobs; keep module as canonical source exported to scheduler for now. | Add adaptor translating schedule definitions into DB seeds. |
| `epochor/utils/hf_io.py` uploads | API-managed checkpoint URLs | Keep functions for uploading artifacts; API will store returned URLs in DB. | Ensure new endpoints accept existing HF URLs. |
| CLI config flags (`neurons/config.py`) | `api/config.py` with env defaults | Keep CLI structure; new config module will read same env vars for API service. | Document additional env vars for API while preserving existing ones. |
| Weight computation (`neurons/validator/scoring_service.py`) | `api/routes/scoring.py` | Reuse scoring functions within API to ensure identical math; expose as HTTP response. | Possibly refactor scoring logic into shared module reused by API and validator for tests. |

## Deprecations & Shims
- `CompetitionManager.get_next_competition` slated for deprecation once scheduler provides leasing; maintain for offline/test mode with `@deprecated` warning.
- `ValidatorState.pending_uids_to_eval` will remain but used only for offline path; API path will store state in DB.
- Import re-exports: keep `neurons.validator` namespace exporting `CompetitionManager`, `EvaluationService`, etc., even if implementations move under `api` or shared modules.

## CLI / Config Compatibility
- Existing validator flags retained; new API-related flags (e.g., `--api-url`, `--api-token`) will default to values compatible with legacy mode. Add warnings when deprecated flags are used.
- Miner CLI remains unchanged; new env var `EPOCHOR_API_URL` will be optional with default pointing to local API.
- Config keys in `constants` remain stable; new config module for API will read same constants where applicable.
