import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from pipeline.utils import get_logger, load_data, preprocess_data, save_data


def main():
    logger = get_logger(__file__)

    try:
        test_size = 0.2

        # data_path = '../experiments/spam.csv'
        project_root = Path(__file__).resolve().parents[1]
        data_path = project_root / "experiments" / "spam.csv"

        df = load_data(data_path)
        final_df = preprocess_data(df)

        train_df, test_df = train_test_split(
            final_df,
            test_size=test_size,
            random_state=2
        )

        save_data(train_df, "train.csv", sub_folder='raw')
        save_data(test_df, "test.csv", sub_folder='raw')

        logger.info("Data ingestion stage completed successfully")

    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s", e)
        raise


if __name__ == '__main__':
    main()
