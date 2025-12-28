import pandas as pd
from pathlib import Path
from typing import Optional

from pipeline.utils.logger import get_logger
from pipeline.utils.paths import DATA_DIR


def load_data(data_url: str | Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    logger = get_logger(__file__)

    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df

    except pd.errors.ParserError as e:
        logger.error('Failed to parse csv file: %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def clean_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw dataset schema (column removal/renaming)."""
    logger = get_logger(__file__)

    try:
        df = df.drop(
            columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],
            errors="ignore",
        )
        df = df.rename(
            columns={'v1': 'target', 'v2': 'text'},
        )
        df["text"] = df["text"].astype(str).fillna("")
        logger.debug("Data preprocessing completed")
        return df

    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise


def save_data(
        data: pd.DataFrame,
        data_name: str,
        sub_folder: Optional[str] = None,
) -> None:
    """Save a dataframe to the data directory."""
    logger = get_logger(__file__)

    try:
        data_dir = (
            DATA_DIR / sub_folder
            if sub_folder
            else DATA_DIR
        )

        data_dir.mkdir(parents=True, exist_ok=True)

        output_path = data_dir / data_name
        data.to_csv(output_path, index=False)

        logger.debug('Data saved to %s', output_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
