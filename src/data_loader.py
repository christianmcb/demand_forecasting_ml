import pandas as pd

from src.config import TRAIN_PATH, STORE_PATH
from src.logger import get_logger

logger = get_logger(__name__)


def load_train_data(train_path=TRAIN_PATH, store_path=STORE_PATH) -> pd.DataFrame:
    logger.info("Loading train data from %s", train_path)
    train = pd.read_csv(train_path)

    logger.info("Loading store data from %s", store_path)
    store = pd.read_csv(store_path)

    logger.info("Merging train and store data")
    df = train.merge(store, on="Store", how="left")

    logger.info("Loaded merged dataframe with shape %s", df.shape)
    return df