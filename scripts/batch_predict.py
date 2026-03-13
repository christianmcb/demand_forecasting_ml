import pandas as pd
import joblib

from src.config import MODEL_PATH
from src.data_loader import load_train_data
from src.preprocessing import preprocess_data
from src.feature_engineering import build_features
from src.predict import predict_from_dataframe


def main():

    model = joblib.load(MODEL_PATH)

    df = load_train_data()
    df = preprocess_data(df)
    df = build_features(df)

    # Take most recent row per store
    df = df.sort_values(["Store", "Date"]).groupby("Store").tail(1)

    preds = predict_from_dataframe(df, model)

    df["prediction"] = preds

    df.to_csv("data/predictions/forecast.csv", index=False)

    print("Saved forecasts to data/predictions/forecast.csv")


if __name__ == "__main__":
    main()