import pickle
from pathlib import Path
from sklearn.base import BaseEstimator

from pipeline.utils.logger import get_logger
from pipeline.utils.paths import MODELS_DIR


def load_model(model_path: Path) -> BaseEstimator:
    """Load the trained model from a file."""
    logger = get_logger(__file__)

    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model

    except FileNotFoundError:
        logger.error('File not found: %s', model_path)
        raise

    except Exception as e:
        logger.error(
            "Unexpected error occurred while loading the model: %s", e)
        raise


def save_model(
    model: BaseEstimator,
    model_name: str,
) -> None:
    """
    Save the trained model to a file.

    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    logger = get_logger(__file__)

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not model_name.endswith(".pkl"):
            model_name += ".pkl"
        output_path = MODELS_DIR / model_name

        with open(output_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', output_path)

    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise
