import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

from src.logger import get_logger

logger = get_logger(__name__)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def build_results_table(baseline_name: str, baseline_metrics: dict, model_metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": [baseline_name, "LightGBM"],
            "MAE": [baseline_metrics["mae"], model_metrics["mae"]],
            "RMSE": [baseline_metrics["rmse"], model_metrics["rmse"]],
        }
    )


def build_test_results(test_data: pd.DataFrame, date_col: str, target_col: str, baseline_pred, model_pred) -> pd.DataFrame:
    test_results = test_data[[date_col, "Store", target_col]].copy()
    test_results["baseline_pred"] = baseline_pred
    test_results["model_pred"] = model_pred
    test_results["abs_error"] = np.abs(test_results[target_col] - test_results["model_pred"])
    return test_results


def build_feature_importance(feature_cols: list[str], importances) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved JSON to %s", path)