import pandas as pd
from src.feature_engineering import build_features

def test_feature_generation():

    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=10),
        "sales": range(10),
        "store": 1,
        "item": 1
    })

    features = build_features(df)

    assert "month" in features.columns
    assert "day_of_week" in features.columns