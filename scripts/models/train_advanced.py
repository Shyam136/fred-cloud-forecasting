#!/usr/bin/env python3
"""
Advanced Model Training Script - Week 5
Train gradient boosting models (XGBoost or LightGBM) and compare with baselines.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import utility functions
from utils import (
    load_baseline_metrics,
    plot_pred_vs_actual,
    plot_residuals,
    plot_feature_importance,
    plot_metric_compare,
    ensure_directory_exists
)


def check_dependencies(model_type: str) -> bool:
    """
    Check if required ML libraries are installed.
    
    Args:
        model_type: Either 'xgboost' or 'lightgbm'
        
    Returns:
        True if dependencies are available, False otherwise
    """
    if model_type == 'xgboost':
        try:
            import xgboost
            print(f"[INFO] XGBoost version: {xgboost.__version__}")
            return True
        except ImportError:
            print("[ERROR] XGBoost is not installed. Please run:")
            print("    pip install xgboost")
            return False
    elif model_type == 'lightgbm':
        try:
            import lightgbm
            print(f"[INFO] LightGBM version: {lightgbm.__version__}")
            return True
        except ImportError:
            print("[ERROR] LightGBM is not installed. Please run:")
            print("    pip install lightgbm")
            return False
    else:
        print(f"[ERROR] Unknown model type: {model_type}")
        return False


def load_data(input_path: str, features_path: Optional[str], 
              date_col: str, target: str) -> Tuple[pd.DataFrame, str]:
    """
    Load and merge datasets.
    
    Args:
        input_path: Path to main dataset
        features_path: Optional path to feature dataset
        date_col: Name of date column
        target: Target variable name
        
    Returns:
        Tuple of (merged dataframe, actual date column name)
    """
    print(f"\n[INFO] Loading main dataset: {input_path}")
    
    # Load main dataset
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    print(f"[INFO] Main dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    
    # Check if date column exists
    actual_date_col = date_col
    if date_col not in df.columns:
        print(f"[WARNING] Date column '{date_col}' not found in dataset")
        if 'date' in df.columns:
            actual_date_col = 'date'
            print(f"[INFO] Using 'date' column instead")
        elif df.index.name and 'date' in df.index.name.lower():
            df = df.reset_index()
            actual_date_col = df.columns[0]
            print(f"[INFO] Reset index, using '{actual_date_col}' as date column")
        else:
            # Create a date column from index
            df = df.reset_index()
            actual_date_col = 'date'
            df.rename(columns={df.columns[0]: actual_date_col}, inplace=True)
            print(f"[INFO] Created '{actual_date_col}' column from index")
    
    # Load and merge features if provided
    if features_path and os.path.exists(features_path):
        print(f"\n[INFO] Loading features from: {features_path}")
        
        if features_path.endswith('.parquet'):
            features_df = pd.read_parquet(features_path)
        elif features_path.endswith('.csv'):
            features_df = pd.read_csv(features_path)
        else:
            raise ValueError(f"Unsupported file format: {features_path}")
        
        print(f"[INFO] Features dataset shape: {features_df.shape}")
        
        # Check for date column in features
        if actual_date_col not in features_df.columns:
            if features_df.index.name and 'date' in features_df.index.name.lower():
                features_df = features_df.reset_index()
            else:
                features_df = features_df.reset_index()
                if actual_date_col not in features_df.columns:
                    features_df.rename(columns={features_df.columns[0]: actual_date_col}, inplace=True)
        
        # Merge on date
        print(f"[INFO] Merging on column: {actual_date_col}")
        df = pd.merge(df, features_df, on=actual_date_col, how='inner', suffixes=('', '_feat'))
        print(f"[INFO] Merged dataset shape: {df.shape}")
    
    # Check if target exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset. Available: {list(df.columns)}")
    
    return df, actual_date_col


def prepare_features(df: pd.DataFrame, target: str, date_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target, handling missing values.
    
    Args:
        df: Input dataframe
        target: Target variable name
        date_col: Date column name
        
    Returns:
        Tuple of (X, y)
    """
    print("\n[INFO] Preparing features and target...")
    
    # Separate target
    y = df[target].copy()
    
    # Select features (exclude target and date)
    feature_cols = [col for col in df.columns if col not in [target, date_col]]
    X = df[feature_cols].copy()
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    
    print(f"[INFO] Number of features: {X.shape[1]}")
    print(f"[INFO] Target: {target}")
    
    # Check for missing values
    missing_y = y.isna().sum()
    missing_X = X.isna().sum().sum()
    
    if missing_y > 0:
        print(f"[WARNING] Target has {missing_y} missing values - will be dropped")
    
    if missing_X > 0:
        print(f"[WARNING] Features have {missing_X} missing values total")
        # Drop rows where target is missing
        valid_mask = ~y.isna()
        y = y[valid_mask]
        X = X[valid_mask]
        
        # Impute remaining missing values in X with median
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"[INFO] Imputed {col} with median: {median_val:.4f}")
    
    print(f"[INFO] Final dataset shape: X={X.shape}, y={y.shape}")
    
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                X_test: pd.DataFrame, y_test: pd.Series,
                model_type: str, seed: int) -> Tuple[object, Dict]:
    """
    Train gradient boosting model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_type: Either 'xgboost' or 'lightgbm'
        seed: Random seed
        
    Returns:
        Tuple of (trained model, metrics dict)
    """
    print(f"\n[INFO] Training {model_type.upper()} model...")
    
    if model_type == 'xgboost':
        import xgboost as xgb
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1
        )
        
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
    elif model_type == 'lightgbm':
        import lightgbm as lgb
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1
        )
        
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    # Train model
    model.fit(X_train, y_train)
    print("[INFO] Training complete")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = {
        'model_type': model_type.upper(),
        'hyperparameters': hyperparams,
        'random_seed': seed,
        'train_r2': r2_score(y_train, y_train_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'n_features': X_train.shape[1],
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test)
    }
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"Training Set:")
    print(f"  R² Score:  {metrics['train_r2']:.6f}")
    print(f"  MAE:       {metrics['train_mae']:.6f}")
    print(f"  RMSE:      {metrics['train_rmse']:.6f}")
    print(f"\nTest Set:")
    print(f"  R² Score:  {metrics['test_r2']:.6f}")
    print(f"  MAE:       {metrics['test_mae']:.6f}")
    print(f"  RMSE:      {metrics['test_rmse']:.6f}")
    print("="*60 + "\n")
    
    return model, metrics, y_test_pred


def save_outputs(model, metrics: Dict, baseline_metrics: Optional[Dict],
                X_test: pd.DataFrame, y_test: pd.Series, y_test_pred: np.ndarray,
                target: str, out_metrics_path: str, out_figs_dir: str,
                timestamp: str) -> None:
    """
    Save all outputs: metrics JSON and figures.
    
    Args:
        model: Trained model
        metrics: Advanced model metrics
        baseline_metrics: Baseline model metrics (if available)
        X_test: Test features
        y_test: Test target
        y_test_pred: Test predictions
        target: Target variable name
        out_metrics_path: Output path for metrics JSON
        out_figs_dir: Output directory for figures
        timestamp: Timestamp string for filenames
    """
    print("\n[INFO] Saving outputs...")
    
    # Ensure output directories exist
    ensure_directory_exists(os.path.dirname(out_metrics_path))
    ensure_directory_exists(out_figs_dir)
    
    # Prepare comparison data
    comparison = {
        'target': target,
        'timestamp': timestamp,
        'advanced_model': metrics,
        'baseline_models': baseline_metrics if baseline_metrics else {}
    }
    
    # Save metrics JSON
    with open(out_metrics_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"[SAVED] Metrics JSON: {out_metrics_path}")
    
    # Generate figures
    residuals = y_test.values - y_test_pred
    
    # 1. Predicted vs Actual
    pred_vs_actual_path = os.path.join(
        out_figs_dir, 
        f"{target}_pred_vs_actual_{timestamp}.png"
    )
    plot_pred_vs_actual(y_test.values, y_test_pred, pred_vs_actual_path, target)
    
    # 2. Residuals
    residuals_path = os.path.join(
        out_figs_dir,
        f"{target}_residuals_{timestamp}.png"
    )
    plot_residuals(residuals, residuals_path, target)
    
    # 3. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        feat_importance_path = os.path.join(
            out_figs_dir,
            f"{target}_feat_importance_{timestamp}.png"
        )
        plot_feature_importance(
            X_test.columns.tolist(),
            model.feature_importances_,
            feat_importance_path,
            top=15,
            target_name=target
        )
    else:
        print("[INFO] Model does not expose feature_importances_")
    
    # 4. Metric Comparison
    metric_compare_path = os.path.join(
        out_figs_dir,
        f"{target}_metric_compare_{timestamp}.png"
    )
    plot_metric_compare(baseline_metrics, metrics, metric_compare_path, target)
    
    print("[INFO] All outputs saved successfully!\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train advanced gradient boosting models for economic forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost model for CPI
  python train_advanced.py --input processed/master_transformed_20251203.parquet --target CPI --model xgboost
  
  # Train LightGBM model for GDP with features
  python train_advanced.py --input processed/master_transformed_20251203.parquet \\
    --features processed/features_gdp_20251126.parquet --target GDP --model lightgbm
  
  # Train with custom test split
  python train_advanced.py --input processed/master_transformed_20251203.parquet \\
    --target FEDFUNDS --model xgboost --test-size 0.25 --seed 42
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to processed dataset (parquet or csv)')
    parser.add_argument('--target', type=str, required=True,
                       choices=['CPI', 'GDP', 'FEDFUNDS'],
                       help='Target variable to predict')
    parser.add_argument('--model', type=str, required=True,
                       choices=['xgboost', 'lightgbm'],
                       help='Model type to train')
    parser.add_argument('--features', type=str, default=None,
                       help='Optional path to features file (will be merged on date)')
    parser.add_argument('--date-col', type=str, default='date',
                       help='Name of date column (default: date)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--out-metrics', type=str, default=None,
                       help='Output path for metrics JSON (default: reports/model_comparison_<date>.json)')
    parser.add_argument('--out-figs', type=str, default='docs/figures/week5/',
                       help='Output directory for figures (default: docs/figures/week5/)')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Generate timestamp
    timestamp = datetime.utcnow().strftime('%Y%m%d')
    
    # Set default output paths
    if args.out_metrics is None:
        args.out_metrics = f'reports/model_comparison_{timestamp}.json'
    
    # Print configuration
    print("\n" + "="*60)
    print("ADVANCED MODEL TRAINING - WEEK 5")
    print("="*60)
    print(f"Input dataset:    {args.input}")
    print(f"Features file:    {args.features or 'None'}")
    print(f"Target variable:  {args.target}")
    print(f"Model type:       {args.model}")
    print(f"Date column:      {args.date_col}")
    print(f"Test size:        {args.test_size}")
    print(f"Random seed:      {args.seed}")
    print(f"Output metrics:   {args.out_metrics}")
    print(f"Output figures:   {args.out_figs}")
    print("="*60 + "\n")
    
    # Check dependencies
    if not check_dependencies(args.model):
        sys.exit(1)
    
    # Load data
    try:
        df, actual_date_col = load_data(
            args.input, 
            args.features, 
            args.date_col, 
            args.target
        )
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)
    
    # Prepare features
    try:
        X, y = prepare_features(df, args.target, actual_date_col)
    except Exception as e:
        print(f"[ERROR] Failed to prepare features: {e}")
        sys.exit(1)
    
    # Split data
    print(f"\n[INFO] Splitting data (test_size={args.test_size}, seed={args.seed})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, shuffle=False
    )
    print(f"[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set:  {X_test.shape[0]} samples")
    
    # Train model
    try:
        model, metrics, y_test_pred = train_model(
            X_train, y_train,
            X_test, y_test,
            args.model,
            args.seed
        )
    except Exception as e:
        print(f"[ERROR] Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load baseline metrics
    baseline_candidates = [
        f'reports/model_report_{args.target}.json',
        f'reports/model_report_{args.target}_{timestamp}.json',
        'reports/model_report_latest.json'
    ]
    baseline_metrics = load_baseline_metrics(baseline_candidates)
    
    # Save outputs
    try:
        save_outputs(
            model, metrics, baseline_metrics,
            X_test, y_test, y_test_pred,
            args.target, args.out_metrics, args.out_figs,
            timestamp
        )
    except Exception as e:
        print(f"[ERROR] Failed to save outputs: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
