from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.predict import load_model, predict_from_dataframe

app = FastAPI(title="Demand Forecasting API")

model = load_model()


class PredictionRequest(BaseModel):
    Store: int
    Promo: int
    SchoolHoliday: int
    StateHoliday: str
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    Promo2: int
    month: int
    day: int
    week_of_year: int
    day_of_week: int
    is_weekend: int
    is_month_start: int
    is_month_end: int
    lag_1: float
    lag_7: float
    lag_14: float
    lag_28: float
    rolling_mean_7: float
    rolling_mean_14: float
    rolling_std_7: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest):
    df = pd.DataFrame([payload.model_dump()])

    for col in ["Store", "StateHoliday", "StoreType", "Assortment"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    pred = predict_from_dataframe(df, model=model)
    return {"predicted_sales": float(pred[0])}