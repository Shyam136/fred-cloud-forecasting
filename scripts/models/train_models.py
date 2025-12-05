<<<<<<< HEAD
# ============================================================
# TRAIN MODELS SCRIPT (FINAL CLEAN VERSION)
# File: scripts/models/train_models.py
# ============================================================

import argparse
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
def load_dataset(target):
    """
    Load the correct feature dataset from the /data folder.
    """

    if target == "CPI":
        path = "data/features_cpi.parquet"
    elif target == "GDP":
        path = "data/features_gdp.parquet"
    elif target == "FEDFUNDS":
        path = "data/features_fedfunds.parquet"
    else:
        raise ValueError("Invalid target name")

    print(f"[INFO] Loading dataset for {target}: {path}")
    return pd.read_parquet(path)


# ------------------------------------------------------------
# TRAIN MODELS
# ------------------------------------------------------------
def train_models(df, target):
    """
    Train Linear Regression and Random Forest models.
    Compute R2, MAE, RMSE metrics.
    """

    # Drop missing rows
    df = df.dropna()

    # -----------------------------------------
    # FIX: Remove datetime columns
    # -----------------------------------------
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns
    if len(datetime_cols) > 0:
        print("[INFO] Dropping datetime columns:", list(datetime_cols))
        df = df.drop(columns=datetime_cols)

    # Features & label
    X = df.drop(columns=[target])
    y = df[target]

    # Time-ordered split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ---------------------------
    # Linear Regression
    # ---------------------------
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    pred_lr = lin.predict(X_test)

    # ---------------------------
    # Random Forest
    # ---------------------------
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # ---------------------------
    # Metrics
    # ---------------------------
    metrics = {
        "target": target,
        "linear_regression": {
            "r2": r2_score(y_test, pred_lr),
            "mae": mean_absolute_error(y_test, pred_lr),
            "rmse": np.sqrt(mean_squared_error(y_test, pred_lr)),
        },
        "random_forest": {
            "r2": r2_score(y_test, pred_rf),
            "mae": mean_absolute_error(y_test, pred_rf),
            "rmse": np.sqrt(mean_squared_error(y_test, pred_rf)),
        },
    }

    return metrics


# ------------------------------------------------------------
# SAVE METRICS REPORT
# ------------------------------------------------------------
def save_report(metrics):
    """
    Save model metrics to reports/model_report_<date>.json
    """

    today = datetime.now().strftime("%Y%m%d")

    os.makedirs("reports", exist_ok=True)

    out_path = f"reports/model_report_{today}.json"

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Saved report: {out_path}")


# ------------------------------------------------------------
# MAIN SCRIPT ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train baseline economic models.")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["CPI", "GDP", "FEDFUNDS"],
        help="Which target variable to train?",
    )

    args = parser.parse_args()
    target = args.target

    # 1. Load dataset
    df = load_dataset(target)

    # 2. Train models
    metrics = train_models(df, target)

    # 3. Save report
    save_report(metrics)

    print("[INFO] Training completed successfully!")
=======
# ============================================================
# TRAIN MODELS SCRIPT (FINAL CLEAN VERSION)
# File: scripts/models/train_models.py
# ============================================================

import argparse
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
def load_dataset(target):
    """
    Load the correct feature dataset from the /data folder.
    """

    if target == "CPI":
        path = "data/features_cpi.parquet"
    elif target == "GDP":
        path = "data/features_gdp.parquet"
    elif target == "FEDFUNDS":
        path = "data/features_fedfunds.parquet"
    else:
        raise ValueError("Invalid target name")

    print(f"[INFO] Loading dataset for {target}: {path}")
    return pd.read_parquet(path)


# ------------------------------------------------------------
# TRAIN MODELS
# ------------------------------------------------------------
def train_models(df, target):
    """
    Train Linear Regression and Random Forest models.
    Compute R2, MAE, RMSE metrics.
    """

    # Drop missing rows
    df = df.dropna()

    # -----------------------------------------
    # FIX: Remove datetime columns
    # -----------------------------------------
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns
    if len(datetime_cols) > 0:
        print("[INFO] Dropping datetime columns:", list(datetime_cols))
        df = df.drop(columns=datetime_cols)

    # Features & label
    X = df.drop(columns=[target])
    y = df[target]

    # Time-ordered split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ---------------------------
    # Linear Regression
    # ---------------------------
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    pred_lr = lin.predict(X_test)

    # ---------------------------
    # Random Forest
    # ---------------------------
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # ---------------------------
    # Metrics
    # ---------------------------
    metrics = {
        "target": target,
        "linear_regression": {
            "r2": r2_score(y_test, pred_lr),
            "mae": mean_absolute_error(y_test, pred_lr),
            "rmse": np.sqrt(mean_squared_error(y_test, pred_lr)),
        },
        "random_forest": {
            "r2": r2_score(y_test, pred_rf),
            "mae": mean_absolute_error(y_test, pred_rf),
            "rmse": np.sqrt(mean_squared_error(y_test, pred_rf)),
        },
    }

    return metrics


# ------------------------------------------------------------
# SAVE METRICS REPORT
# ------------------------------------------------------------
def save_report(metrics):
    """
    Save model metrics to reports/model_report_<date>.json
    """

    today = datetime.now().strftime("%Y%m%d")

    os.makedirs("reports", exist_ok=True)

    out_path = f"reports/model_report_{today}.json"

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Saved report: {out_path}")


# ------------------------------------------------------------
# MAIN SCRIPT ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train baseline economic models.")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["CPI", "GDP", "FEDFUNDS"],
        help="Which target variable to train?",
    )

    args = parser.parse_args()
    target = args.target

    # 1. Load dataset
    df = load_dataset(target)

    # 2. Train models
    metrics = train_models(df, target)

    # 3. Save report
    save_report(metrics)

    print("[INFO] Training completed successfully!")
>>>>>>> bfa3c79cd36f2304b130ad6fb01f8b8563988586
