# Compatibility Matrix

| Existing Surface | New Surface / Adapter | Strategy | Notes |
|------------------|-----------------------|----------|-------|
| `neurons.validator.Validator` main loop | `neurons/validator.py` (unchanged entrypoint) + optional API integrations | Maintained file/class; additional config attributes allow validators to consume API weights and sandbox overrides without breaking CLI contracts. | Legacy flow still supported for offline mode; API usage is opt-in via new flags. |
| `neurons.validator.weight_setter.WeightSetter` | Same module now calling `/scoring/weights` before `set_weights` | Preserve class interface; inject API URL/token through existing constructor args so external callers stay compatible. | Falls back to legacy behaviour when no API URL configured. |
| `neurons.validator.model_manager` leasing logic | Superseded by `api/routes/validator.py` leasing endpoints | Validator continues exporting module but delegates leasing to API by consuming remote jobs; local manager retains legacy cache utilities. | No direct shim required; documented expectation to use API for coordinated tournaments. |
| Miner CLI auto submission scripts | `/miner/submit` endpoint (FastAPI) | Miner CLI adds optional `--platform_api_url/token` flags to register submissions remotely while keeping legacy HF-only mode. | When URL absent the miner behaves exactly as before. |
| Tournament scheduler (`CompetitionManager`) | `api/competition_scheduler.py` | Scheduler logic centralised in API while `CompetitionManager` remains for historical flows/tests. | API promotions follow same thresholds/ordering; compatibility maintained via shared constants. |
| Hugging Face upload helpers (`epochor.utils.hf_io`) | `templates/validator_training/hf_io.py` re-export | Template exposes identical import path used by historical submissions. | No behaviour change; ensures miners referencing old path continue to function. |
| Miner submission protocol (`epochor.training.validator_contract.MinerSubmissionProtocol`) | `templates/validator_training/miner_protocol.MinerSubmissionProtocol` subclass | Provides stable path for template consumers while inheriting canonical protocol definition. | Keeps validation logic in one place. |
| Training entrypoint (`epochor/training/validator_runner.run_training`) | `templates/validator_training/trainer_entry.run` | Chutes template executes miner submissions inside deterministic loop while respecting legacy protocol semantics. | Resume checkpoints, metrics, and artifact metadata remain schema-compatible. |
| Weight computation (`neurons/validator/scoring_service.py`) | `api/routes/scoring.py` | API reuses proportional weighting logic (inverse-loss) and ensures deterministic ordering. | Weight setter now fetches REST weights; scoring service still available for validator offline mode/tests. |

## Deprecations & Shims
- Competition promotion now canonically handled via API scheduler; `neurons/validator/competition_manager` retained for backwards compatibility but slated for deprecation once all validators migrate.
- Legacy on-disk state/lease tracking remains for offline usage; API path stores authoritative lease assignments in the database.
- No CLI flags removed; new flags for API URL/token default to empty strings and are safe for older scripts.

## CLI / Config Compatibility
- API configuration introduced via `EPOCHOR_*` environment variables; legacy validator/miner config continues to work without them.
- Miners can provide `EPOCHOR_MODEL_ID`/`EPOCHOR_MODEL_CODE_URL` to auto-register; absence preserves old HF-only workflow.
- Validator sandbox parameters (`--sandbox_*`) preserved; template enforces complementary restrictions (network disable, deterministic seeds).
