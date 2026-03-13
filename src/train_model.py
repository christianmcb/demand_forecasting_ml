from pathlib import Path
import joblib
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import mlflow
import mlflow.lightgbm

from src.config import (
    CATEGORICAL_COLS,
    DATE_COL,
    FEATURE_COLS,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    METRICS_SUMMARY_PATH,
    MODEL_PARAMS,
    MODEL_PATH,
    TARGET_COL,
    TEST_PREDICTIONS_PATH,
    TRAIN_END_DATE,
    VAL_END_DATE,
    LAG_WINDOWS,
    ROLLING_WINDOWS,
)
from src.data_loader import load_train_data
from src.evaluate_model import (
    build_feature_importance,
    build_results_table,
    build_test_results,
    regression_metrics,
    save_json,
)
from src.feature_engineering import build_features
from src.logger import get_logger
from src.preprocessing import preprocess_data
from src.metadata import save_metadata

logger = get_logger(__name__)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df[DATE_COL] < TRAIN_END_DATE].copy()
    val_df = df[(df[DATE_COL] >= TRAIN_END_DATE) & (df[DATE_COL] < VAL_END_DATE)].copy()
    test_df = df[df[DATE_COL] >= VAL_END_DATE].copy()

    logger.info("Train shape: %s", train_df.shape)
    logger.info("Validation shape: %s", val_df.shape)
    logger.info("Test shape: %s", test_df.shape)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more chronological splits are empty. Check split dates.")

    return train_df, val_df, test_df


def prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    feature_cols = [col for col in FEATURE_COLS if col in train_df.columns]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in feature_cols]

    if not feature_cols:
        raise ValueError("No feature columns found in training dataframe.")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[TARGET_COL].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, categorical_cols


def _ensure_output_dirs() -> None:
    output_paths = [
        MODEL_PATH,
        METRICS_PATH,
        METRICS_SUMMARY_PATH,
        TEST_PREDICTIONS_PATH,
        FEATURE_IMPORTANCE_PATH,
    ]
    for path in output_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def train_model(experiment_name: str = "demand_forecasting_ml") -> dict:
    logger.info("Starting training pipeline")
    _ensure_output_dirs()

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("target_col", TARGET_COL)
        mlflow.log_param("date_col", DATE_COL)
        mlflow.log_param("train_end_date", TRAIN_END_DATE)
        mlflow.log_param("val_end_date", VAL_END_DATE)
        mlflow.log_param("lag_windows", ",".join(map(str, LAG_WINDOWS)))
        mlflow.log_param("rolling_windows", ",".join(map(str, ROLLING_WINDOWS)))
        mlflow.log_param("n_configured_features", len(FEATURE_COLS))

        for param_name, param_value in MODEL_PARAMS.items():
            mlflow.log_param(f"model__{param_name}", param_value)

        df = load_train_data()
        mlflow.log_param("raw_rows", len(df))
        mlflow.log_param("raw_columns", len(df.columns))

        df = preprocess_data(df)
        mlflow.log_metric("rows_after_preprocessing", len(df))

        df = build_features(df)
        mlflow.log_metric("rows_after_feature_engineering", len(df))

        train_df, val_df, test_df = split_data(df)

        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("val_rows", len(val_df))
        mlflow.log_metric("test_rows", len(test_df))

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols,
            categorical_cols,
        ) = prepare_xy(train_df, val_df, test_df)

        mlflow.log_param("used_features", ",".join(feature_cols))
        mlflow.log_param("categorical_features", ",".join(categorical_cols))
        mlflow.log_metric("n_used_features", len(feature_cols))

        baseline_col = "lag_7" if "lag_7" in X_test.columns else "lag_1"
        if baseline_col not in X_test.columns:
            raise ValueError("Neither lag_7 nor lag_1 is available for baseline prediction.")

        baseline_pred = X_test[baseline_col].values
        baseline_metrics = regression_metrics(y_test, baseline_pred)

        logger.info("Baseline (%s) metrics: %s", baseline_col, baseline_metrics)
        mlflow.log_param("baseline_model", baseline_col)
        mlflow.log_metric("baseline_mae", baseline_metrics["mae"])
        mlflow.log_metric("baseline_rmse", baseline_metrics["rmse"])

        model = LGBMRegressor(**MODEL_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            categorical_feature=categorical_cols,
            callbacks=[
                early_stopping(stopping_rounds=100),
                log_evaluation(100),
            ],
        )

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = regression_metrics(y_val, val_pred)
        test_metrics = regression_metrics(y_test, test_pred)

        logger.info("Validation metrics: %s", val_metrics)
        logger.info("Test metrics: %s", test_metrics)

        mlflow.log_metric("val_mae", val_metrics["mae"])
        mlflow.log_metric("val_rmse", val_metrics["rmse"])
        mlflow.log_metric("test_mae", test_metrics["mae"])
        mlflow.log_metric("test_rmse", test_metrics["rmse"])

        results_df = build_results_table(
            baseline_name=f"Baseline ({baseline_col})",
            baseline_metrics=baseline_metrics,
            model_metrics=test_metrics,
        )

        test_results = build_test_results(
            test_data=test_df,
            date_col=DATE_COL,
            target_col=TARGET_COL,
            baseline_pred=baseline_pred,
            model_pred=test_pred,
        )

        importance_df = build_feature_importance(feature_cols, model.feature_importances_)

        joblib.dump(model, MODEL_PATH)
        results_df.to_csv(METRICS_SUMMARY_PATH, index=False)
        test_results.to_csv(TEST_PREDICTIONS_PATH, index=False)
        importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

        metrics_payload = {
            "baseline_model": baseline_col,
            "baseline_mae": baseline_metrics["mae"],
            "baseline_rmse": baseline_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "n_train_rows": int(len(train_df)),
            "n_val_rows": int(len(val_df)),
            "n_test_rows": int(len(test_df)),
            "features": feature_cols,
            "categorical_features": categorical_cols,
        }
        save_json(metrics_payload, METRICS_PATH)
        save_metadata("models/model_metadata.json")
        

        mlflow.log_artifact(str(METRICS_SUMMARY_PATH))
        mlflow.log_artifact(str(TEST_PREDICTIONS_PATH))
        mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH))
        mlflow.log_artifact(str(METRICS_PATH))

        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
        )

        logger.info("Training complete")

        return {
            "model": model,
            "results_df": results_df,
            "test_results": test_results,
            "importance_df": importance_df,
            "metrics": metrics_payload,
        }