from pathlib import Path
import logging


def get_logger(
        file_path: str,
        level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Returns a configured logger instance with console and file handlers.

    Args:
        name (str): Logger name (usually __name__)
        level (int): Logging level (default: logging.INFO)

    Returns:
        logging.Logger
    """

    # Resolve project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_DIR.mkdir(exist_ok=True)

    # Module name from file
    module_name = Path(file_path).stem  # data_ingestion.py → data_ingestion
    log_file = LOG_DIR / f"{module_name}.log"

    logger = logging.getLogger(module_name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
