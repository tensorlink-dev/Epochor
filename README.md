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

Epochor is a Bittensor subnet that **incentivizes the creation of foundational time-series models**. Miners train models and publish them to **Hugging Face Hub**, while validators fetch these models, evaluate them on **synthetic and real-world datasets**, and score them with competition-specific metrics (currently **CRPS**).  

Competitions are rotated dynamically, and **the top-performing miner in each competition earns the largest share of rewards**, while other miners still receive proportionally smaller allocations. This **winner-makes-most** structure creates strong incentives for innovation while maintaining fairness across participants.

This codebase builds upon the work of the [Pretrain Subnet](https://github.com/opentensor/pretrain).

---

## Subnet Flow

```
 Miner (train + push to HF) 
        │
        ▼
 HF Model Store (base_hf_model_store.py)
        │
        ▼
 Validator (validator.py)
        │
 ┌─────────────── Services ────────────────┐
 │ CompetitionManager → select dataset     │
 │ ModelManager → fetch model from HF      │
 │ EvaluationService → run forecasts       │
 │ ScoringService → compute metric (CRPS)  │
 │ State/EMA → smooth scores               │
 │ WeightSetter → submit set_weights       │
 │ Clone Assessment → detect duplicates    │
 └─────────────────────────────────────────┘
        │
        ▼
 Subtensor (on-chain weights & rewards)
        │
        ▼
 Rewards distributed (validation/rewards.py)
```

---

## ✨ Key Features & Design Principles

Epochor's design incorporates several key features to ensure a fair, competitive, and exploitation-resistant environment.

### 🏆 Winner-Makes-Most Incentives
Rewards are weighted so the **#1 ranked miner receives the majority of emissions** for each competition. Other miners receive smaller proportional shares. This ensures competitiveness while still rewarding participation.

### 🛡️ Sybil Resistance
The winner-makes-most mechanism makes Sybil attacks unprofitable. Running many mediocre nodes yields minimal returns — miners must focus resources into building genuinely competitive models.

### 💡 Innovation Over Imitation
`assess_clones.py` enforces penalties on duplicate or plagiarized models. A challenger must demonstrate **clear improvement in the scoring metric** to overtake the leader, forcing true innovation.

### 🧠 Zero-Shot Generalization
Validators draw from a **broad and rotating set of datasets** (synthetic + real). Miners never know which competition comes next, ensuring that rewarded models are **generalist** rather than overfit.

### ⚙️ Standardized Architecture
All models must be implemented with **[Temporal](https://github.com/your-repo/temporal)** for compatibility and fairness. This ensures seamless evaluation and reproducibility.

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
 ├─ evaluation/     # eval tasks, scoring methods
 ├─ generators/     # synthetic time-series kernels
 ├─ model/          # model stores, tracker, updater
 ├─ validation/     # EMA tracker, rewards, clone detection
 ├─ utils/          # helper functions
neurons/validator/
 ├─ competition_manager.py  # schedules datasets
 ├─ evaluation_service.py   # runs model inference
 ├─ model_manager.py        # handles HF model pulls
 ├─ scoring_service.py      # scoring logic (currently CRPS)
 ├─ state.py                # EMA + state tracking
 ├─ weight_setter.py        # submits weights
 └─ validator.py            # main validator loop
```

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/badger-fi/epochor-subnet.git
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
     python examples/run_validator.py        --wallet.name <validator_wallet>        --wallet.hotkey <validator_hotkey>        --subtensor.network <network_name>        --netuid <epochor_netuid>
     ```
   - Miner:
     ```bash
     python your_miner_script.py        --wallet.name <miner_wallet>        --wallet.hotkey <miner_hotkey>        --subtensor.network <network_name>        --netuid <epochor_netuid>
     ```

---

## 🛡️ Safety and Best Practices
- **Secure Your Keys** – Never commit wallet keys.  
- **Resource Management** – Monitor GPU/CPU usage.  
- **Testing** – Validate in testnet before mainnet deployment.  
- **Stay Updated** – Sync with latest Bittensor + Epochor changes.  
- **Monitor Logs** – Track metrics in WandB and logs.  

---

This README now reflects the actual validator/miner flow, the **winner-makes-most** reward design, and that while **CRPS is currently used**, the framework is flexible to future metrics.
