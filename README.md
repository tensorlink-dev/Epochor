<div align="center">
    <img src="docs/assets/epochor_logo.png#gh-dark-mode-only" width="300px">
    <img src="docs/assets/epochor_logo.png#gh-light-mode-only" width="300px">
</div>

# Epochor Subnet | SN13

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Our Mission
Our mission is to incentivize and democratize temporal intelligence. We are building an open-source, decentralized platform for time-series models that empowers anyone, anywhere, to develop and share state-of-the-art predictive models. By fostering collective innovation, we aim to ensure transparent, reproducible, and incentive-aligned progress towards robust, generalist temporal reasoning for the economic benefit of all.

---

## 📜 Overview

Epochor is a Bittensor subnet that **incentivizes the creation of foundational time-series models**. Validators now **own the entire training loop**: they choose the datasets, number of steps, hardware, and evaluation routine. Miners provide lightweight Python submissions that implement a small contract (model construction, optimizer selection, and the per-batch training step). Validators load those submissions, train the models inside a sandboxed loop, and score the resulting checkpoints on synthetic and real-world datasets (currently with **CRPS**).

Competitions rotate dynamically, and **the top-performing miner in each competition earns the largest share of rewards**, while other miners still receive proportionally smaller allocations. This **winner-makes-most** structure creates strong incentives for innovation while maintaining fairness across participants.

The repository builds upon the work of the [Pretrain Subnet](https://github.com/opentensor/pretrain) while retooling the validator workflow around submission-driven training.

---

## Subnet Flow

```
 Miner submission (miner_submission.py)
        │
        ▼
 Remote submission store (e.g. private HF repo)
        │
        ▼
 Validator (neurons/validator.py)
        │
 ┌─────────────── Services ────────────────┐
 │ ModelManager      → sync miner submissions    │
 │ Training Loop     → run validator-owned steps │
 │ EvaluationService → score trained checkpoints │
 │ ScoringService    → compute CRPS + weights    │
 │ State/EMA         → smooth scores             │
 │ WeightSetter      → submit set_weights        │
 └───────────────────────────────────────────────┘
        │
        ▼
 Subtensor (on-chain weights & rewards)
        │
        ▼
 Rewards distributed via validator-managed scoring
```

---

## ✨ Key Features & Design Principles

Epochor's design incorporates several key features to ensure a fair, competitive, and exploitation-resistant environment.

### 🏆 Winner-Makes-Most Incentives
Rewards are weighted so the **#1 ranked miner receives the majority of emissions** for each competition. Other miners receive smaller proportional shares. This ensures competitiveness while still rewarding participation.

### 🛡️ Sybil Resistance
The winner-makes-most mechanism makes Sybil attacks unprofitable. Running many mediocre nodes yields minimal returns — miners must focus resources into building genuinely competitive models.

### 💡 Innovation Over Imitation
Validator-run safeguards discourage duplicate or plagiarized models. Challengers must demonstrate **clear improvement in the scoring metric** to overtake the leader, forcing true innovation.

### 🧠 Zero-Shot Generalization
Validators draw from a **broad and rotating set of datasets** (synthetic + real). Miners never know which competition comes next, ensuring that rewarded models are **generalist** rather than overfit.

### ⚙️ Standardized Interface
Miners target a compact API: subclasses of `epochor.model.base.BaseTemporalModel` plus the `MinerSubmissionProtocol` hooks (`build_model`, `build_optimizer`, `train_step`). Validators run those hooks inside an audited training harness, guaranteeing apples-to-apples comparisons regardless of the underlying architecture.

---

## ⚙️ Scoring Mechanism

All competitions use a **consistent scoring pipeline**, with the **current primary metric being CRPS** (Continuous Ranked Probability Score). Future competitions may introduce additional or alternative metrics as needed.

1. **Data Generation** – Fresh datasets (synthetic GP kernels, financial returns, etc.) are created or loaded each round.  
2. **Forecasting** – Validators fetch miner models from Hugging Face and run them on unseen data.  
3. **Evaluation** – Forecasts are scored using **CRPS** (ensemble CRPS when probabilistic sampling is available).  
4. **Smoothing** – Scores are tracked with an **Exponential Moving Average (EMA)** for stability.  
5. **Clone Assessment** – Duplicate detection prevents trivial copies from gaming rewards.  
6. **Reward Allocation** – The **winner receives the majority share**, others get smaller proportional weights.  

---

## 📂 Project Structure

```
epochor/
 ├─ datasets/       # dataset loaders & IDs
 ├─ evaluation/     # CRPS scoring pipeline
 ├─ generators/     # synthetic time-series generators
 ├─ model/          # submission tracker, stores, constraints
 ├─ training/       # validator-run training harness & contract
 ├─ validation/     # EMA tracker & stats helpers
 ├─ utils/          # logging, hashing, misc helpers
neurons/validator/
 ├─ competition_manager.py  # schedules datasets
 ├─ evaluation_service.py   # trains + evaluates submissions
 ├─ model_manager.py        # syncs miner submissions
 ├─ scoring_service.py      # turns metrics into weights
 ├─ state.py                # persisted validator state
 ├─ weight_setter.py        # submits weights on-chain
 └─ validator.py            # orchestrates the validator
```

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/tensorlink-dev/epochor-subnet.git
   cd epochor-subnet
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp .env.example .env
   # Add your WANDB_API_KEY and HF_TOKEN
   ```

4. **Run a neuron**
   - Validator:
     ```bash
     python examples/run_validator.py \
       --wallet.name <validator_wallet> \
       --wallet.hotkey <validator_hotkey> \
       --subtensor.network <network_name> \
       --netuid <epochor_netuid>
     ```
   - Miner:
     1. Implement `miner_submission.py` exposing `get_submission()` with the `MinerSubmissionProtocol` hooks.
     2. Package and upload the submission to your configured remote store (e.g. a private Hugging Face repo).
     3. Run the lightweight heartbeat miner (see `neurons/miner.py`) to stay registered on the subnet.

---

## 🛡️ Safety and Best Practices
- **Secure Your Keys** – Never commit wallet keys.  
- **Resource Management** – Monitor GPU/CPU usage.  
- **Testing** – Validate in testnet before mainnet deployment.  
- **Stay Updated** – Sync with latest Bittensor + Epochor changes.  
- **Monitor Logs** – Track metrics in WandB and logs.  

---
