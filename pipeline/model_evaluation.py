import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


from pipeline.utils import get_logger, load_model, MODELS_DIR, PROCESSED_DATA_DIR, load_data, save_metrics


def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate the model and return the evaluation metrics.
    """
    logger = get_logger(__file__)

    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred,)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def model_evaluation() -> None:
    logger = get_logger(__file__)

    try:
        clf = load_model(MODELS_DIR / "model.pkl")
        test_data = load_data(PROCESSED_DATA_DIR / "test_tfidf.csv")

        X_test = test_data.iloc[:, :-1].to_numpy()
        y_test = test_data.iloc[:, -1].to_numpy()

        metrics = evaluate_model(clf, X_test, y_test)

        save_metrics(metrics, "metrics.json")

    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s", e)
        raise


if __name__ == "__main__":
    model_evaluation()
