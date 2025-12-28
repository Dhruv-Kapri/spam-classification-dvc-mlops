import json

from pipeline.utils.logger import get_logger
from pipeline.utils.paths import REPORTS_DIR


def save_metrics(
    metrics: dict,
    metrics_name: str
) -> None:
    """
    Evaluate the model and return the evaluation metrics.
    """
    logger = get_logger(__file__)

    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        if not metrics_name.endswith(".pkl"):
            metrics_name += ".pkl"
        output_path = REPORTS_DIR / metrics_name

        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', output_path)

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise
