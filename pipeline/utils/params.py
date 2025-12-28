import yaml
from pathlib import Path

from pipeline.utils.logger import get_logger
from pipeline.utils.paths import PROJECT_ROOT


def load_params(
        params_path: Path = PROJECT_ROOT / "params.yaml"
) -> dict:
    """
    Load parameters from a YAML file.
    """
    logger = get_logger(__file__)

    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters retrieved from %s", params_path)
        return params

    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise

    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise

    except Exception as e:
        logger.error("Unexpected error while loading params: %s", e)
        raise
