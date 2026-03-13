import numpy as np
import pandas as pd

from src.config import DATE_COL
from src.logger import get_logger

logger = get_logger(__name__)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    logger.info("Initial shape: %s", df.shape)

    # Remove closed-store and zero-sales rows
    df = df.loc[df["Open"] != 0].copy()
    df = df.loc[df["Sales"] != 0].copy()

    # Customers should not be used for real forecasting
    if "Customers" in df.columns:
        df = df.drop(columns=["Customers"])

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Ensure categorical-style columns are strings
    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)

    # Fill missing values
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    df = df.sort_values(["Store", DATE_COL]).reset_index(drop=True)

    logger.info("Post-preprocessing shape: %s", df.shape)
    return df