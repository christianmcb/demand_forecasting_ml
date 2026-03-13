import pandas as pd
from src.feature_engineering import create_calendar_features


def test_calendar_features():

    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"]
    })

    df["Date"] = pd.to_datetime(df["Date"])

    result = create_calendar_features(df)

    assert "month" in result.columns
    assert "day_of_week" in result.columns