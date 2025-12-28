import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from pipeline.utils import get_logger, load_data, save_model, load_params, PROCESSED_DATA_DIR


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict
) -> RandomForestClassifier:
    """
    Train the RandomForest model.

    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    logger = get_logger(__file__)

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "The number of samples in X_train and y_train must be the same.")

        logger.debug(
            'Initializing RandomForest model with parameters: %s', params)
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug('Model training started with %d samples',
                     X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')

        return clf

    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise

    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def model_building() -> None:
    logger = get_logger(__file__)

    try:
        params = load_params()['model_building']
        # params = {
        #     "n_estimators":  22,
        #     "random_state":  2
        # }

        train_data = load_data(PROCESSED_DATA_DIR / "train_tfidf.csv")
        X_train = train_data.iloc[:, :-1].to_numpy()
        y_train = train_data.iloc[:, -1].to_numpy()

        clf = train_model(X_train, y_train, params)

        save_model(clf, "model.pkl")
        logger.debug("Model Building Completed")

    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        raise


if __name__ == "__main__":
    model_building()
