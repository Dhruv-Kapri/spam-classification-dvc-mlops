import pandas as pd
import nltk
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder

from pipeline.utils import get_logger, load_data, save_data, RAW_DATA_DIR

ps = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))


def transform_text(text: str) -> str:
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and (
        t not in STOPWORDS) and (t not in string.punctuation)]
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)


def process_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = 'text',
    target_column: str = 'target',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    logger = get_logger(__file__)

    try:
        logger.debug('Starting preprocessing for DataFrame')

        # Encode the target column
        encoder = LabelEncoder()

        # Remove duplicate rows
        train_df = train_df.drop_duplicates(keep='first')
        test_df = test_df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        train_df[target_column] = encoder.fit_transform(
            train_df[target_column])
        test_df[target_column] = encoder.transform(test_df[target_column])
        logger.debug('Target column encoded')

        # Apply text transformation to the specified text column
        train_df.loc[:, text_column] = train_df[text_column].fillna(
            "").apply(transform_text)
        test_df.loc[:, text_column] = test_df[text_column].fillna(
            "").apply(transform_text)
        logger.debug('Text column transformed')

        return train_df, test_df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise

    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def data_preprocessing(
    text_column: str = 'text',
    target_column: str = 'target',
) -> None:
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    logger = get_logger(__file__)

    try:
        train_data = load_data(RAW_DATA_DIR / "train.csv")
        test_data = load_data(RAW_DATA_DIR / "test.csv")
        logger.debug('Loaded data for Preprocessing')

        # Transform the data
        train_processed_data, test_processed_data = process_data(
            train_data, test_data)

        # Save data to interim folder
        save_data(train_processed_data, "train.csv", sub_folder="interim")
        save_data(test_processed_data, "test.csv", sub_folder="interim")

        logger.debug('Data Processed')

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise

    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
        raise

    except Exception as e:
        logger.error('Encountered unexpected error: %s', e)
        raise


if __name__ == "__main__":
    data_preprocessing()
