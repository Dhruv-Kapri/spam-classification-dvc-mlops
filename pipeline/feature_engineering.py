import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.utils import get_logger, load_data, save_data, load_params, INTERIM_DATA_DIR


def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply TfIdf to the data.
    """
    logger = get_logger(__file__)

    try:
        train_data["text"] = train_data["text"].fillna("")
        test_data["text"] = test_data["text"].fillna("")

        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        # NOTE: Dense conversion is NOT scalable.
        X_train_bow = vectorizer.fit_transform(X_train)
        train_df = pd.DataFrame(X_train_bow.todense())
        train_df['label'] = y_train

        X_test_bow = vectorizer.transform(X_test)
        test_df = pd.DataFrame(X_test_bow.todense())
        test_df['label'] = y_test

        logger.debug('tfidf applied and data transformed')

        return train_df, test_df

    except Exception as e:
        logger.error(
            "Error during Bag of Words transformation: %s", e)
        raise


def feature_engineering() -> None:
    logger = get_logger(__file__)

    try:
        params = load_params()
        max_features = params['feature_engineering']['max_features']
        # max_features = 50

        train_data = load_data(INTERIM_DATA_DIR / "train.csv")
        test_data = load_data(INTERIM_DATA_DIR / "test.csv")
        logger.debug('Loaded data for Feature Engineering')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        logger.debug('Tfidf transformation complete')

        save_data(train_df, "train_tfidf.csv", sub_folder="processed")
        save_data(test_df, "test_tfidf.csv", sub_folder="processed")
        logger.debug('Saved Feature Engineered data')

    except Exception as e:
        logger.error(
            "Failed to complete the feature engineering process: %s", e)
        raise


if __name__ == "__main__":
    feature_engineering()
