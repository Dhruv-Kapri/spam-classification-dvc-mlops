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
5. Model Evaluation  

All stages are automated and reproducible using DVC.

<!-- --- -->

## How to Run

1. Set up venv
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

2. Install requirements
  ```bash
  pip install -r requirements.txt
  ```

3. Run project
  ```bash
  dvc repro
  ```

<!-- --- -->

## Experiments
- Pipeline parameters are controlled via `params.yaml`.
- Changing parameters and running `dvc repro` or `dvc exp run` creates reproducible experiments.

