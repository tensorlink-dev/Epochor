# Simple Validator Training Chute Demo

This walkthrough shows how to do a semi-live end-to-end demo: build the validator-training Chute image, run the sandbox trainer against a toy miner submission, and (optionally) publish the resulting checkpoint to Hugging Face. The flow mirrors what validators execute in production and is safe to run from Google Colab or a local workstation.

## 1. Environment bootstrap
1. Clone the repository and install the dependencies that power the validator sandbox and the Chute builder.
   ```bash
   !git clone https://github.com/tensorlink-dev/epochor.git
   %cd epochor
   !pip install -r requirements.txt
   !pip install chutes bittensor fastapi uvicorn apscheduler
   ```
   The validator template relies on the `chutes` builder helper plus PyTorch, Safetensors, and NumPy pinned inside `DEFAULT_PIP_PACKAGES`.【F:templates/validator_training/template_builder.py†L12-L51】

2. (Optional) set any secrets up front. To successfully push artifacts to Hugging Face you must export a token under the environment variable referenced by `HF_WRITE_TOKEN_ENV` (defaults to `HF_TOKEN`).【F:templates/validator_training/trainer_entry.py†L16-L22】【F:templates/validator_training/trainer_entry.py†L248-L332】

   ```python
   import os
   os.environ["HF_TOKEN"] = "hf_..."  # personal access token with write scope
   ```

## 2. Build and inspect the validator-training Chute image
```python
from templates.validator_training.template_builder import build_validator_training_template

chute = build_validator_training_template(
    username="demo-handle",
    gpu_count=1,
    min_vram_gb_per_gpu=12,
    concurrency=1,
    timeout_seconds=1800,
    extra_pip=["wandb==0.17.0"],
)

print(chute.image.python_version)      # -> "3.11"
print(chute.image.pip_packages)        # includes torch/safetensors/numpy + wandb
print(chute.entry_file, chute.entry_point)
print(chute.environment)
```
This call verifies that the template copies `templates/validator_training` into `/app`, wires the `trainer_entry.run` entrypoint, and injects the `SUBMISSION_DIR`/`ARTIFACTS_DIR` environment variables expected by the sandbox.【F:templates/validator_training/template_builder.py†L30-L50】

## 3. Author a minimal miner submission
Create a `demo_submission/miner.py` that satisfies `MinerSubmissionProtocol`. The toy dataset exposed by the sandbox uses dense regression features (`x`) and targets (`y`), so a single linear layer is sufficient.

```python
%%bash
mkdir -p demo_submission
cat <<'PY' > demo_submission/miner.py
from templates.validator_training import MinerSubmissionProtocol
from torch import nn
import torch.optim as optim

class Submission(MinerSubmissionProtocol):
    def build_model(self, cfg):
        d_in = int(cfg.get("input_dim", 16))
        return nn.Linear(d_in, 1)

    def build_optimizer(self, model, cfg):
        lr = float(cfg.get("lr", 1e-3))
        return optim.Adam(model.parameters(), lr=lr)

    def train_step(self, model, batch, optimizer, step_idx, cfg):
        model.train()
        optimizer.zero_grad()
        preds = model(batch["x"]).squeeze(-1)
        loss = (preds - batch["y"].squeeze(-1)).pow(2).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "step": step_idx}
PY
```
The loader in `trainer_entry` checks for either `Submission` or `get_submission()` and enforces the `MinerSubmissionProtocol` contract before training begins.【F:templates/validator_training/trainer_entry.py†L40-L74】

## 4. Run the sandbox trainer locally
The trainer expects asynchronous execution, the submission directory mounted at `SUBMISSION_DIR`, and a writable `ARTIFACTS_DIR` for checkpoints and metadata.

```python
import asyncio
import os
from pathlib import Path

from templates.validator_training import trainer_entry

submission_dir = Path("demo_submission").resolve()
artifacts_dir = Path("demo_artifacts").resolve()
artifacts_dir.mkdir(exist_ok=True)

os.environ["SUBMISSION_DIR"] = str(submission_dir)
os.environ["ARTIFACTS_DIR"] = str(artifacts_dir)

cfg = {
    "seed": 1234,
    "train_batch_size": 128,
    "input_dim": 16,
    "max_steps": 200,
    "max_seconds": 120,
    "lr": 5e-3,
    "miner_hotkey": "demo-hotkey",
    "hf_repo_namespace": "demo-hotkey",
    "hf_repo_name": "validator-demo",
    "output_tag": "demo",
}

lease = {
    "submission_id": "demo-submission",
    "model_id": "demo-model",
    "round": 1,
}

result = asyncio.run(trainer_entry.run({"cfg": cfg, "lease": lease}))
print(result)
```
`trainer_entry.run` performs the following during the demo:
- Seeds all RNGs for deterministic behaviour.【F:templates/validator_training/trainer_entry.py†L314-L322】
- Loads the submission, constructs the model/optimizer, and iterates over the toy dataloader, enforcing finite loss values.【F:templates/validator_training/trainer_entry.py†L108-L197】【F:templates/validator_training/trainer_entry.py†L226-L286】
- Saves the best checkpoint as safetensors, computes a SHA-256 artifact ID, and writes JSON metadata alongside the training metrics.【F:templates/validator_training/trainer_entry.py†L198-L244】
- Stages files for upload and attempts to push them to Hugging Face using the configured repo ID; failures are reported in the returned payload so the rest of the flow can continue.【F:templates/validator_training/trainer_entry.py†L248-L332】

If the upload succeeds (`ok: True`), Hugging Face will contain the checkpoint and metadata. Without a token, the function returns `ok: False` but still leaves artifacts and metadata locally for inspection.

## 5. Inspect artifacts
```python
list(artifacts_dir.iterdir())
```
You should see the safetensors checkpoint, JSON metadata, and a prepared `upload_*/` folder. Open the metadata file to confirm that the recorded lease, config, loss metrics, and elapsed time look reasonable.【F:templates/validator_training/trainer_entry.py†L210-L244】

## 6. (Optional) Register the submission with the platform API
To complete the semi-live demo, stand up the FastAPI app in-memory and register the freshly trained model. This exercises the same endpoints miners and validators use during real competitions.

```python
from fastapi.testclient import TestClient
from api.config import Settings
from api import database
from api.main import create_app

# in-memory SQLite backing store
os.environ["EPOCHOR_DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
database.configure_engine(os.environ["EPOCHOR_DATABASE_URL"])
database.init_db()
app = create_app()
client = TestClient(app)

settings = Settings()
settings.allowed_miner_hotkeys.append("demo-hotkey")
settings.allowed_validator_hotkeys.append("demo-validator")

submission = client.post("/miner/submit", json={
    "hotkey": "demo-hotkey",
    "model_code_url": "https://huggingface.co/demo-hotkey/validator-demo",
}).json()

lease = client.post("/validator/request-training-job", json={
    "validator_hotkey": "demo-validator",
}).json()

print(submission)
print(lease)
```
The lease payload mirrors what the sandbox consumed earlier, and you can optionally feed the recorded `artifact_id` and `model_id` back into `/validator/submit-results` to close the loop.【F:api/main.py†L1-L120】【F:api/database.py†L1-L120】

---
Following these steps gives you a repeatable “semi-live” smoke test that touches the Chute build path, the sandbox trainer, artifact generation, and the platform API without needing full subnet infrastructure.
