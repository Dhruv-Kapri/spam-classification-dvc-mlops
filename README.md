# Spam Classification using DVC & MLOps

AAn **end-to-end spam classification ML pipeline** built with **modular Python code**, **DVC for data & experiment versioning**, and **Git-based workflows**.
  
The project emphasizes **reproducibility, automation, and MLOps practices** over raw model performance.

<!-- --- -->

## Project Overview

This repository demonstrates a **production-style ML workflow** including:

- Data ingestion and splitting
- Text preprocessing and normalization
- Feature engineering using **TF-IDF**
- Model training with **RandomForest**
- Model evaluation with standard classification metrics
- Parameterized pipelines using `params.yaml`
- Automated execution and experiment tracking using **DVC**
- Experiment tracking and visualization using **dvclive**

<!-- --- -->

## Project Structure

  ```
  spam-classification-dvc-mlops
  ├── pipeline/                        # Core ML pipeline code
  │ ├── data_ingestion.py
  │ ├── data_preprocessing.py
  │ ├── feature_engineering.py
  │ ├── model_building.py
  │ ├── model_evaluation.py
  │ └── utils/                         # Shared utilities
  │  ├── data.py
  │  ├── logger.py
  │  ├── model.py
  │  ├── metrics.py
  │  ├── params.py
  │  └── paths.py
  │
  ├── data/                            # DVC-tracked data
  │ ├── raw/
  │ ├── interim/
  │ └── processed/
  │
  ├── models/                          # Trained models (DVC tracked)
  │ └── model.pkl
  │
  ├── reports/                         # Evaluation outputs
  │ └── metrics.json
  │
  ├── dvclive/                         # Experiment tracking & metrics
  │ ├── params.yaml
  │ ├── metrics.json
  │ └── plots/
  │ └── metrics/
  │
  ├── experiments/                     # Notebooks / exploratory work
  ├── logs/                            # Per-module log files
  │
  ├── dvc.yaml                         # DVC pipeline definition
  ├── params.yaml                      # Pipeline parameters
  ├── dvc.lock                         # DVC pipeline lockfile
  │
  ├── requirements.txt
  ├── requirements-dev.txt
  └── README.md
  ```

<!-- --- -->

## DVC Pipeline Stages
1. Data Ingestion  
2. Data Preprocessing  
3. Feature Engineering (TF-IDF)  
4. Model Building  
5. Model Evaluation (with `dvclive` tracking) 

All stages are automated and reproducible using DVC.

<!-- --- -->

## How to Run

1. Set up venv:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

2. Install requirements:
  ```bash
  pip install -r requirements.txt
  ```

3. Run the full pipeline:
  ```bash
  dvc repro
  ```

4. Run experiments
  ```bash
  dvc exp run
  dvc exp show
  ```

<!-- --- -->

## Experiments & dvclive
- Parameters are controlled via `params.yaml` or `dvclive/params.yaml`.
- Metrics are automatically logged to `dvclive/metrics.json`.
- Plots of accuracy, precision, recall, and AUC are saved in `dvclive/plots/metrics`.
- Every `dvc exp run` creates a reproducible experiment with tracked metrics and parameters.
- Previous experiments can be reproduced or removed with:
  ```bash
  dvc exp apply <exp-name>
  dvc exp remove <exp-name>
  ```
