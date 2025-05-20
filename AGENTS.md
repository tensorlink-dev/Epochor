# Agents

> Developer guide to integrating OpenAI Codex agents into this repository for code generation, testing, and Bittensor subnet simulation.

---

## Table of Contents

1. [Overview](#overview)
2. [Code Generation with Codex](#code-generation-with-codex)

   * [Prerequisites](#prerequisites)
   * [Prompting Best Practices](#prompting-best-practices)
   * [Integration Example](#integration-example)
3. [Unit Test Generation](#unit-test-generation)

   * [Prompting Tests](#prompting-tests)
   * [Test Framework Integration](#test-framework-integration)
   * [Example Tests](#example-tests)
4. [Mock Miners for Bittensor Subnet](#mock-miners-for-bittensor-subnet)

   * [Overview](#overview-1)
   * [Creating Wallets](#creating-wallets)
   * [Axon Server Setup](#axon-server-setup)
   * [Dummy Forward Function](#dummy-forward-function)
   * [Validator Configuration](#validator-configuration)
   * [Simulation Best Practices](#simulation-best-practices)
5. [Best Practices and Tips](#best-practices-and-tips)
6. [References](#references)

---

## Overview

This document outlines how to use **OpenAI Codex** as an AI coding agent within our development workflow:

* **Generate** and insert Python code chunks with correct syntax and project cohesion.
* **Write** unit tests via Codex, integrating with `pytest`.
* **Simulate** Bittensor network components by creating **mock miners** for quick testing of subnet interactions.

Follow the patterns and examples below to integrate AI-generated content safely and effectively.

---

## Code Generation with Codex

### Prerequisites

* Python ≥ 3.8
* `openai` Python package installed:

  ```bash
  pip install openai python-dotenv
  ```
* `.env` file in repo root containing:

  ```bash
  OPENAI_API_KEY=sk-<your-key>
  ```

### Prompting Best Practices

1. **Include Context**: Supply function signatures, imports, and docstrings in the prompt.
2. **Be Specific**: Clearly state the task and expected behavior.
3. **Chunk It**: Generate small functions or classes one at a time.
4. **Review & Refine**: Always lint, test for syntax, and manually inspect.
5. **Maintain Style**: Use existing linters/formatters (e.g., Black, flake8) on generated code.

### Integration Example

````python
import os
import openai
from dotenv import load_dotenv

# Load API key
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
openai.api_key = os.getenv('OPENAI_API_KEY')

prompt = '''
# Function: update_user
'''
'''Updates a user's email in the database. Raises ValueError if not found.'''
's'
'def update_user(user_id, new_email):
    # Implementation here
'''
'''
response = openai.Completion.create(
    model='code-davinci-002',
    prompt=prompt,
    max_tokens=150,
    temperature=0,
    stop=['```']
)
generated = response.choices[0].text
print(generated)
````

1. Insert `generated` into your codebase.
2. Run `python -m py_compile path/to/file.py` to check syntax.
3. Lint with `flake8` and format with `black`.

---

## Unit Test Generation

### Prompting Tests

* Provide the **function implementation** or signature in the prompt.
* Specify **`pytest`** style tests.
* Ask for **edge cases** and **error conditions**.

### Test Framework Integration

1. Install `pytest`:

   ```bash
   pip install pytest
   ```
2. Place tests in a `/tests` directory.
3. Name test files with the pattern `test_*.py`.
4. Run with:

   ```bash
   pytest --maxfail=1 --disable-warnings -q
   ```

### Example Tests

*Function under test (`user.py`):*

```python
# user.py

def divide(a, b):
    '''Returns a / b, raises ZeroDivisionError if b == 0.'''
    return a / b
```

*Codex-generated tests (`tests/test_user.py`):*

```python
import pytest
from user import divide


def test_divide_positive():
    assert divide(6, 3) == 2


def test_divide_negative():
    assert divide(-6, 3) == -2


def test_divide_zero_divisor():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
```

1. Review and adjust any imports or fixtures.
2. Run `pytest` and ensure tests pass.

---

## Mock Miners for Bittensor Subnet

### Overview

Mock miners simulate real Bittensor nodes (miners) to test validators and network interactions without full models or on-chain registration.

### Creating Wallets

```python
import bittensor
# Create a wallet (coldkey + hotkey)
wallet = bittensor.wallet(path='~/.bittensor', name='mockminer1', hotkey='hk1')
wallet.create_new_coldkey(overwrite=True)
wallet.create_new_hotkey(overwrite=True)
```

> *Option:* Use a testnet or local Subtensor node for on-chain registration, or allow non-registered miners.

### Axon Server Setup

```python
import bittensor
import torch

# Assuming `wallet` is created as above
axon = bittensor.axon(wallet=wallet, port=8091)
```

### Dummy Forward Function

```python
# Define a dummy forward callback matching your subnet's modality

def dummy_forward(input_tensor):
    # Return zeros of the same shape
    return torch.zeros_like(input_tensor)

axon.attach_forward_callback(dummy_forward, modality=bittensor.proto.Modality.TENSOR)
axon.start()  # Begins serving requests on port 8091
```

### Validator Configuration

* Use the same `netuid` and chain endpoint as your mock miner.
* For quick tests, allow querying non-registered miners:

  ```bash
  bittensor-miner --axon_port 8091 --allow_non_registered
  ```
* Or configure your local validator script to call the Axon directly via `bittensor.dendrite`.

### Simulation Best Practices

* **Latency Simulation:** Add `time.sleep(delay)` in `dummy_forward` to mimic inference time.
* **Concurrency:** Configure `axon.max_workers` to test parallel requests.
* **Logging:** Enable debug logging (`--logging.debug`) to trace requests and responses.

---

## Best Practices and Tips

* **Generate → Verify → Refine**: Always review AI-generated code and docs.
* **Maintain Style**: Run linters and formatters on all new code.
* **CI Integration**: Include generated tests in your CI pipeline.
* **Documentation**: Update this file when your Codex or Bittensor usage patterns change.

---

## References

* OpenAI Codex overview: [https://openai.com/index/introducing-codex/](https://openai.com/index/introducing-codex/)
* Bittensor docs: [https://bittensor.com/docs](https://bittensor.com/docs)
* PyTest documentation: [https://docs.pytest.org](https://docs.pytest.org)
