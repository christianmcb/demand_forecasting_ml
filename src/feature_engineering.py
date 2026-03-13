import pandas as pd

from src.config import DATE_COL, STORE_COL, TARGET_COL, LAG_WINDOWS
from src.logger import get_logger

logger = get_logger(__name__)


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["day"] = df[DATE_COL].dt.day
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df[DATE_COL].dt.is_month_start.astype(int)
    df["is_month_end"] = df[DATE_COL].dt.is_month_end.astype(int)

    return df


def create_lag_features(
    df: pd.DataFrame,
    group_col: str = STORE_COL,
    target_col: str = TARGET_COL,
    lags: list[int] = LAG_WINDOWS,
) -> pd.DataFrame:
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    group_col: str = STORE_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    df = df.copy()

    shifted = df.groupby(group_col)[target_col].shift(1)

    df["rolling_mean_7"] = (
        shifted.groupby(df[group_col])
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_mean_14"] = (
        shifted.groupby(df[group_col])
        .rolling(14)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_std_7"] = (
        shifted.groupby(df[group_col])
        .rolling(7)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating calendar features")
    df = create_calendar_features(df)

    logger.info("Creating lag features")
    df = create_lag_features(df)

    logger.info("Creating rolling features")
    df = create_rolling_features(df)

    logger.info("Dropping rows with missing lag/rolling history")
    df = df.dropna().reset_index(drop=True)

    logger.info("Feature dataframe shape: %s", df.shape)
    return df