import pandas as pd
from src.predict import predict_from_dataframe


class DummyModel:
    def predict(self, X):
        return [123.45] * len(X)


def test_predict_shape():
    df = pd.DataFrame({
        "Store": [1],
        "Promo": [0],
        "SchoolHoliday": [0],
        "StateHoliday": ["0"],
        "StoreType": ["a"],
        "Assortment": ["a"],
        "CompetitionDistance": [500.0],
        "Promo2": [0],
        "month": [1],
        "day": [15],
        "week_of_year": [3],
        "day_of_week": [2],
        "is_weekend": [0],
        "is_month_start": [0],
        "is_month_end": [0],
        "lag_1": [5100.0],
        "lag_7": [5000.0],
        "lag_14": [4900.0],
        "lag_28": [4950.0],
        "rolling_mean_7": [4800.0],
        "rolling_mean_14": [4700.0],
        "rolling_std_7": [150.0],
    })

    model = DummyModel()
    preds = predict_from_dataframe(df, model)

    assert len(preds) == 1