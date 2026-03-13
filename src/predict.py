import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.config import FEATURE_COLS, CATEGORICAL_COLS


def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)

def predict_from_dataframe(df: pd.DataFrame, model):

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    X = df[feature_cols].copy()

    # Ensure categorical dtype matches training
    for col in cat_cols:
        X[col] = X[col].astype("category")

    preds = model.predict(X)

    return preds