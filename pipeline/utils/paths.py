from pathlib import Path

# Project root (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Common directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
