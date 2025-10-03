# TrailKeeper

*A modular framework for ML experimentation and experiment tracking*

## Overview

TrailKeeper is a framework designed to **structure, track, and reproduce machine learning experiments**.
It provides a clean architecture inspired by **SOLID principles**, enabling modularity, extensibility, and reproducibility across the full ML workflow.

The framework helps researchers and practitioners to:

* Organize experiments in a reproducible way
* Configure pipelines with ease (YAML configs)
* Track metrics, logs, and artifacts
* Integrate with experiment tracking tools (Weights & Biases, etc.)
* Compare and visualize results

---

## Features

* **Config-driven workflows**: Define experiments using YAML configs
* **Modular data loaders**: Easily swap datasets (e.g., MNIST, custom datasets)
* **Preprocessing components**: Standard normalization, transformations
* **Flexible modeling**: Training, inference, and structure separation
* **Optimizer integration**: Support for custom optimizers & Optuna for hyperparameter search
* **Metrics factory**: Plug-and-play metrics for evaluation
* **Logging and tracking**: Base logger + W&B tracker for experiment visualization
* **Reproducibility**: Save runs, metrics, predictions, and configs

---

## Project Structure

```
configs/           # YAML configs for experiments
data/              # Datasets
data_loader/       # Data loaders (e.g. MNISTLoader)
data_pre_processing/ # Preprocessing modules
logger/            # Logging utilities
metrics/           # Evaluation metrics
modeling/          # Model structure, training, inference
optimizer/         # Base + Optuna optimizers
runs/              # Saved experiment runs
tracker/           # Experiment tracking (base + W&B)
utils/             # Helpers (config, device, IO, etc.)
wandb/             # W&B integration scripts
```

---

## Getting Started

### 1. Installation

Clone the repository:

```bash
git clone https://github.com/MauricioSight/TrailKeeper.git
cd TrailKeeper
```

Create and activate a local virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Linux/Mac
source .venv/bin/activate
# On Windows (PowerShell)
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running an Experiment

```bash
python execute_train_validation.py --config <config_file (i.e. mlp)>
```

### 3. Running Hyperparameter Tuning

```bash
python execute_tunning.py --config <config_file (i.e. mlp)>
```

### 4. Evaluating Results

```bash
python execute_get_metrics.py --run_id <run_id>
```

---

## Experiment Tracking

TrailKeeper integrates with **Weights & Biases** [W&B](https://github.com/MauricioSight/TrailKeeper) for experiment logging and visualization:

* Track metrics in real-time
* Compare experiments
* Reproduce runs with saved configs and logs

---

## Design Principles

TrailKeeper is inspired by **SOLID design principles**:

* **S**ingle Responsibility → Clear separation of concerns
* **O**pen/Closed → Easy to extend with new components
* **L**iskov Substitution → Interchangeable modules
* **I**nterface Segregation → Minimal, focused interfaces
* **D**ependency Inversion → Flexible architecture

---

## Contributing

Contributions are welcome! Please open issues or PRs to suggest improvements.

