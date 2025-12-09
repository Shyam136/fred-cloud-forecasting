# ============================================================
# PIPELINE SCRIPT - WEEK 5 (Feature Selection + Training)
# File: scripts/pipeline/run_pipeline.py
#
# What this script does:
# 1. Load feature dataset for CPI / GDP / FEDFUNDS
# 2. Do feature selection using Random Forest importance
# 3. Retrain Linear Regression & Random Forest using selected features
# 4. Save metrics to reports/pipeline_report_<date>.json
# ============================================================

import argparse
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ------------------------------------------------------------
# HELPER: Load dataset (same logic as train_models.py)
# ------------------------------------------------------------
def load_dataset(target: str) -> pd.DataFrame:
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
        raise ValueError(f"Invalid target name: {target}")

    print(f"[INFO] Loading dataset for {target}: {path}")
    df = pd.read_parquet(path)

    # Drop missing rows
    df = df.dropna()

    # Drop datetime columns (cannot go into sklearn directly)
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns
    if len(datetime_cols) > 0:
        print(f"[INFO] Dropping datetime columns: {list(datetime_cols)}")
        df = df.drop(columns=datetime_cols)

    return df


# ------------------------------------------------------------
# HELPER: Split data into train / test (time-ordered)
# ------------------------------------------------------------
def make_train_test(df: pd.DataFrame, target: str, test_size: float = 0.2):
    """
    Create X_train, X_test, y_train, y_test with time-ordered split.
    """

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# STEP 1: Feature Selection using Random Forest
# ------------------------------------------------------------
def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 10,
    n_estimators: int = 300,
    random_state: int = 42,
):
    """
    Train a Random Forest on the full feature set and return the names
    of the top_k most important features.
    """

    print(f"[INFO] Running feature selection (top_k={top_k})")

    rf_fs = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state
    )
    rf_fs.fit(X_train, y_train)

    importances = rf_fs.feature_importances_
    importance_series = pd.Series(importances, index=X_train.columns)

    # Sort descending by importance
    importance_sorted = importance_series.sort_values(ascending=False)

    # Select top_k features (or all if fewer than top_k)
    top_k = min(top_k, len(importance_sorted))
    selected_features = importance_sorted.iloc[:top_k].index.tolist()

    print("[INFO] Top features selected:")
    for name, val in importance_sorted.iloc[:top_k].items():
        print(f"       {name}: {val:.4f}")

    return selected_features, importance_sorted.to_dict()


# ------------------------------------------------------------
# STEP 2: Train models on selected features
# ------------------------------------------------------------
def train_models_with_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Train Linear Regression and Random Forest using the given
    feature subset. Return metrics.
    """

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
        "linear_regression": {
            "r2": r2_score(y_test, pred_lr),
            "mae": mean_absolute_error(y_test, pred_lr),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_lr))),
        },
        "random_forest": {
            "r2": r2_score(y_test, pred_rf),
            "mae": mean_absolute_error(y_test, pred_rf),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_rf))),
        },
    }

    return metrics


# ------------------------------------------------------------
# STEP 3: Run full pipeline for a single target
# ------------------------------------------------------------
def run_pipeline_for_target(target: str, top_k: int = 10):
    """
    End-to-end pipeline for one target:
    1. Load dataset
    2. Train/test split
    3. Feature selection (Random Forest importance)
    4. Train models with selected features
    5. Return all metrics and importance
    """

    print("=" * 60)
    print(f"[PIPELINE] Running pipeline for target: {target}")
    print("=" * 60)

    # 1. Load dataset
    df = load_dataset(target)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = make_train_test(df, target)

    # 3. Feature selection
    selected_features, importance_dict = select_top_features(
        X_train, y_train, top_k=top_k
    )

    # 4. Restrict to selected features
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # 5. Train models on selected features
    metrics_selected = train_models_with_features(
        X_train_sel, X_test_sel, y_train, y_test
    )

    # Build a combined result dict for this target
    result = {
        "target": target,
        "selected_features": selected_features,
        "feature_importance": importance_dict,
        "metrics_selected_features": metrics_selected,
    }

    return result


# ------------------------------------------------------------
# SAVE PIPELINE REPORT
# ------------------------------------------------------------
def save_pipeline_report(results_dict):
    """
    Save pipeline results to reports/pipeline_report_<date>.json
    """

    os.makedirs("reports", exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    out_path = f"reports/pipeline_report_{today}.json"

    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"[INFO] Saved pipeline report: {out_path}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: feature selection + model training."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="ALL",
        choices=["ALL", "CPI", "GDP", "FEDFUNDS"],
        help="Which target to run the pipeline for? Use ALL to run for all three.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top features to keep based on feature importance.",
    )

    args = parser.parse_args()

    # Determine which targets to run
    if args.target == "ALL":
        targets = ["CPI", "GDP", "FEDFUNDS"]
    else:
        targets = [args.target]

    all_results = {"targets": {}}

    for tgt in targets:
        result = run_pipeline_for_target(tgt, top_k=args.top_k)
        all_results["targets"][tgt] = result

    # Save one combined report
    save_pipeline_report(all_results)

    print("[INFO] Pipeline completed successfully for:", ", ".join(targets))
