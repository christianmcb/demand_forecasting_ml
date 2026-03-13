import json
import platform
import sklearn
import lightgbm
import pandas as pd
from datetime import datetime


def build_metadata():

    return {
        "training_timestamp": datetime.utcnow().isoformat(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lightgbm.__version__
    }


def save_metadata(path):

    meta = build_metadata()

    with open(path, "w") as f:
        json.dump(meta, f, indent=4)