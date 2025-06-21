<div align="center">
    <img src="docs/assets/epochor_logo.png#gh-dark-mode-only" width="300px">
    <img src="docs/assets/epochor_logo.png#gh-light-mode-only" width="300px">
</div>

# Epochor Subnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Our Mission
Our mission is to democratize temporal intelligence by building an open-source, decentralized platform for time-series modeling and forecasting—empowering anyone, anywhere, to develop, share, and deploy state-of-the-art predictive models across industries and domains. We strive to catalyze collective innovation in time-series AI, ensuring transparent, reproducible, and incentive-aligned progress toward robust, generalist temporal reasoning for the benefit of all.

## 📜 Overview

Epochor is a Bittensor subnet that evaluates and ranks time-series models on both synthetic and real-world datasets, benchmarking them against competition-specific metrics—beginning with probabilistic forecasting via CRPS—and aggregating results to identify the leader. Inspired by Pretrain’s winner-takes-all scoring, it awards tokens exclusively to the top models, driving continual innovation and refinement of robust, general-purpose temporal forecasting systems.

Think of Epochor as Pretrain for time series models.

## 🚀 Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/badger-fi/epochor-subnet.git
    cd epochor-subnet
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment:**
    Create a `.env` file from the example and add your API keys:
    ```bash
    cp .env.example .env
    # Edit .env with your WANDB_API_KEY and HF_TOKEN
    ```

4.  **Run a neuron:**
    *   **Validator:** See the [How Validators Operate](#how-validators-operate) section.
    *   **Miner:** See the [How Miners Operate](#how-miners-operate) section.

## ⚙️ Subnet Architecture & Scoring Mechanism

The Epochor subnet uses a multi-stage process to evaluate and reward miners for their time-series forecasting models. The core of this mechanism is designed to be fair, robust, and to incentivize the development of high-quality models.

### Scoring Process Breakdown:

1.  **Data Generation**: Validators generate synthetic time-series datasets. This ensures that evaluation tasks are new and unpredictable for each scoring cycle, preventing miners from simply memorizing or overfitting to a static dataset.

2.  **Task Execution**: Miners push their models to Hugging Face. Validators then pull the models and use them to generate predictions for future time points in the series.

3.  **Evaluation (CRPS)**: The primary metric for our first competition is the **Continuous Ranked Probability Score (CRPS)**. CRPS is a sophisticated metric that assesses both the accuracy and the probabilistic nature of a forecast. A lower CRPS score indicates a better forecast.

4.  **Score Aggregation & Ranking**: To create a stable and reliable measure of a miner's long-term performance, raw CRPS scores are used to calculate a model's **Win Rate** and **Gap Score**. These are then combined into a composite score, which is smoothed using an **Exponential Moving Average (EMA)**.

    *   **Win Rate**: Measures how frequently a model outperforms others in head-to-head comparisons on the same datasets. A model "wins" if its CRPS score is lower than a competitor's.
    
    *   **Gap Score**: Measures the *statistical significance* of the performance differences between models. It answers the question: "Is a model's high Win Rate due to genuine superiority or just random chance?" This is done by analyzing the statistical separation between models' CRPS confidence intervals.

    *   **Composite Score**: The final score is a combination of these two metrics, rewarding models that both win frequently and win decisively.
        ```
        Composite Score = Win Rate * Gap Score
        ```

5.  **Weight Allocation**: The final, smoothed scores are converted into on-chain "weights". These weights determine the proportion of network rewards (TAO) each miner receives. This process encourages continuous improvement and the development of genuinely powerful and generalizable time-series forecasting models.

## 🧠 Why Time-Series Foundational Models?

### Endogenous Benefits (within the Bittensor Ecosystem)

*   **Model-First Time-Series Backbone**: Establishes a powerful, generalist temporal representation engine that downstream prediction subnets can fine-tune.
*   **Robust Models for Prediction-First Subnets**: Provides high-quality pretrained weights, allowing specialized subnets to achieve top performance with less data.
*   **Ecosystem Strengthening**: Acts as a shared temporal substrate that aligns reward signals and accelerates research cycles.
*   **Foundation for Multimodal AGI**: Lays the time-series “pillar” necessary for integrating temporal reasoning alongside vision and language.

### Exogenous Benefits & Industry Impact

*   **Universal Forecasting & Classification**: Powers critical tasks in finance, sales, climate modeling, and IoT.
*   **Access to a $10B+ Market**: Addresses a massive combined industry opportunity where accurate forecasts drive value.
*   **Vertical-Specific Model Extensions**: Enables rapid customization for enterprise use cases (e.g., retail, energy).
*   **Edge & Embedded Deployments**: Supports lightweight variants for on-device inference in industrial and consumer IoT.

## How Validators Operate

Validators are responsible for maintaining the integrity of the network.

**Responsibilities:**
*   Stay synchronized with the Bittensor network.
*   Generate evaluation tasks for miners.
*   Pull miner models and evaluated them on the evaluation tasks.
*   Track miner performance using an EMA.
*   Allocate rewards and set weights on the blockchain.
*   Log operations and metrics to Weights & Biases (WandB).

**Running a Validator:**
```bash
python examples/run_validator.py \
    --wallet.name <your_validator_wallet> \
    --wallet.hotkey <your_validator_hotkey> \
    --subtensor.network <network_name> \
    --netuid <epochor_netuid>
```

## How Miners Operate

Miners are the core contributors to the network.

**Responsibilities:**
*   Develop and train time-series forecasting models.
*   Push models to Hugging Face.
*   Serve models via a Bittensor Axon.
*   Update models or push new models to stay competitive.

**Running a Miner:**
```bash
python your_miner_script.py \
    --wallet.name <your_miner_wallet> \
    --wallet.hotkey <your_miner_hotkey> \
    --subtensor.network <network_name> \
    --netuid <epochor_netuid>
```

## Project Structure

*   `epochor/`: Core logic for the subnet.
    *   `config.py`: Configuration for parameters like EMA span and reward strategy.
    *   `validator.py`: Main validator implementation.
    *   `rewards.py`: Reward allocation strategies.
    *   `ema_tracker.py`: Exponential Moving Average tracking utility.
*   `examples/`: Example scripts for running neurons.
*   `template/`: Bittensor neuron templates.
*   `tests/`: Unit and integration tests.

## 🛡️ Safety and Best Practices

*   **Secure Your Keys:** Never share your wallet keys or commit them to version control.
*   **Resource Management:** Monitor the computational resources your neuron consumes.
*   **Test Thoroughly:** Test your neurons in a local or testnet environment before deploying.
*   **Stay Updated:** Keep your Bittensor and subnet code up to date.
*   **Monitor Your Neuron:** Regularly check your neuron's logs and performance.

---

This README provides a foundational understanding of the Epochor subnet. For the most up-to-date information, please refer to the latest code and official announcements.
