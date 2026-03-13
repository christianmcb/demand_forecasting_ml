from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOK_OUTPUTS_DIR = BASE_DIR / "notebooks" / "outputs"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
STORE_PATH = RAW_DATA_DIR / "store.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "lightgbm_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = NOTEBOOK_OUTPUTS_DIR / "feature_importance.csv"
TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "test_predictions.csv"
METRICS_SUMMARY_PATH = NOTEBOOK_OUTPUTS_DIR / "metrics_summary.csv"

TARGET_COL = "Sales"
DATE_COL = "Date"
STORE_COL = "Store"

TRAIN_END_DATE = "2015-01-01"
VAL_END_DATE = "2015-06-01"

LAG_WINDOWS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14]

FEATURE_COLS = [
    "Store",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
    "month",
    "day",
    "week_of_year",
    "day_of_week",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
]

CATEGORICAL_COLS = ["Store", "StateHoliday", "StoreType", "Assortment"]

MODEL_PARAMS = {
    "objective": "regression",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}