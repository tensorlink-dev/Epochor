# Local CPU Sandbox Demo

This walkthrough shows how to spin up a self-contained sandbox exercise on a CPU-only
machine. It installs the validator dependencies, synthesises two toy miner submissions
(`moving_average` and `trend`), and then runs each submission through the same
validator-owned training loop that powers production by invoking the
`epochor.training.sandbox_entry` module directly. The run produces summaries and
artifacts you can inspect to verify end-to-end behaviour without needing a live
subnet or GPUs.

## 1. Environment setup

1. **Create and activate a virtual environment** (Python 3.10+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**. The demo relies on the regular validator requirements
   plus the repository in editable mode so the generated submissions can import
   `epochor` modules:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

3. *(Optional)* **Verify the test harness** if you want an additional sanity check on
   the validator training loop:

   ```bash
   pytest tests/test_validator_training.py::test_run_training_respects_step_cap
   ```

## 2. Run the local sandbox demo

The helper script `scripts/local_cpu_demo.py` orchestrates the demo. It writes two
reference miner submissions, stages synthetic training/validation/evaluation payloads,
and then executes the sandbox entry point once per submission. You can override the
output directory, context window, forecast horizon, or random seed with CLI flags if
desired.【F:scripts/local_cpu_demo.py†L3-L205】

```bash
python scripts/local_cpu_demo.py --output-dir local_demo
```

The script prints a short summary once both runs finish, and the destination directory
is populated with:

```
local_demo/
  submissions/
    moving_average/submission.py
    trend/submission.py
  staging/
    cfg.json
    train_batches.pt
    val_batches.pt
    eval_samples.pt
    eval_tasks.pt
  runs/
    moving_average/
      summary.json
      artifacts/
        model.safetensors
        run_meta.json
    trend/
      summary.json
      artifacts/
        model.safetensors
        run_meta.json
```

Each run’s `summary.json` contains the final training metrics, validation loss, device
choice, and step count emitted by the sandbox entry point.【F:scripts/local_cpu_demo.py†L135-L175】

## 3. Inspect the results

1. **Review the JSON summaries** to confirm both submissions trained and evaluated:

   ```bash
   python -m json.tool local_demo/runs/moving_average/summary.json
   python -m json.tool local_demo/runs/trend/summary.json
   ```

2. **Inspect the saved checkpoints**. The sandbox serialises the trained weights as a
   Safetensors file for each submission. You can, for example, verify file metadata or
   load the weights into a Python shell for manual probing:

   ```bash
   ls -lh local_demo/runs/moving_average/artifacts/
   ```

3. **Iterate on the submissions**. The generated templates live under
   `local_demo/submissions/`. You can edit them and rerun the script (pass `--force` to
   overwrite the templates) to see how changes impact training metrics or validation
   scores.【F:scripts/local_cpu_demo.py†L107-L127】

## 4. Customise the sandbox exercise

The script exposes several knobs for deeper experimentation:

- `--context-length` / `--prediction-length` control how large the synthetic sequences
  are when staging batches and evaluation payloads.【F:scripts/local_cpu_demo.py†L186-L204】
- `--seed` ensures reproducibility across staging, submissions, and evaluation.
- `--force` rewrites the submission templates even if you have modified them locally.

Combine these switches with edits to the generated submissions to stress different
parts of the validator contract before packaging your own miner.
