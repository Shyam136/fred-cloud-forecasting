"""
Utility functions for model training and evaluation.
Week 5 - Advanced Modeling
"""

import json
import os
from typing import List, Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_baseline_metrics(path_candidates: List[str]) -> Optional[Dict]:
    """
    Load baseline metrics from JSON files.
    
    Args:
        path_candidates: List of possible paths to baseline metric files
        
    Returns:
        Dictionary with baseline metrics or None if not found
    """
    for path in path_candidates:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                print(f"[INFO] Loaded baseline metrics from: {path}")
                return data
            except Exception as e:
                print(f"[WARNING] Could not load {path}: {e}")
                continue
    
    print("[WARNING] No baseline metrics found")
    return None


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                       out_path: str, target_name: str = "Target") -> None:
    """
    Create predicted vs actual scatter plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        out_path: Output file path
        target_name: Name of target variable
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel(f'Actual {target_name}', fontsize=12)
    plt.ylabel(f'Predicted {target_name}', fontsize=12)
    plt.title(f'Predicted vs Actual: {target_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Prediction plot: {out_path}")


def plot_residuals(residuals: np.ndarray, out_path: str, 
                   target_name: str = "Target") -> None:
    """
    Create residuals histogram.
    
    Args:
        residuals: Array of residuals (y_true - y_pred)
        out_path: Output file path
        target_name: Name of target variable
    """
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    
    plt.xlabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Residuals Distribution: {target_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Residuals plot: {out_path}")


def plot_feature_importance(names: List[str], importances: np.ndarray, 
                           out_path: str, top: int = 15,
                           target_name: str = "Target") -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        names: List of feature names
        importances: Array of importance values
        out_path: Output file path
        top: Number of top features to display
        target_name: Name of target variable
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top]
    top_names = [names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_importances)))
    plt.barh(range(len(top_importances)), top_importances, color=colors)
    plt.yticks(range(len(top_importances)), top_names)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top} Feature Importances: {target_name}', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Feature importance plot: {out_path}")


def plot_metric_compare(baseline_dict: Optional[Dict], advanced_dict: Dict,
                       out_path: str, target_name: str = "Target") -> None:
    """
    Create comparison bar chart for baseline vs advanced models.
    
    Args:
        baseline_dict: Dictionary with baseline model metrics
        advanced_dict: Dictionary with advanced model metrics
        out_path: Output file path
        target_name: Name of target variable
    """
    metrics_to_plot = ['r2', 'mae', 'rmse']
    
    # Prepare data
    models = []
    r2_vals = []
    mae_vals = []
    rmse_vals = []
    
    # Add baseline models if available
    if baseline_dict:
        if 'linear_regression' in baseline_dict:
            models.append('Linear Regression\n(Baseline)')
            r2_vals.append(baseline_dict['linear_regression'].get('r2', 0))
            mae_vals.append(baseline_dict['linear_regression'].get('mae', 0))
            rmse_vals.append(baseline_dict['linear_regression'].get('rmse', 0))
        
        if 'random_forest' in baseline_dict:
            models.append('Random Forest\n(Baseline)')
            r2_vals.append(baseline_dict['random_forest'].get('r2', 0))
            mae_vals.append(baseline_dict['random_forest'].get('mae', 0))
            rmse_vals.append(baseline_dict['random_forest'].get('rmse', 0))
    
    # Add advanced model
    model_type = advanced_dict.get('model_type', 'Advanced')
    models.append(f'{model_type}\n(Advanced)')
    r2_vals.append(advanced_dict.get('test_r2', 0))
    mae_vals.append(advanced_dict.get('test_mae', 0))
    rmse_vals.append(advanced_dict.get('test_rmse', 0))
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(models))
    width = 0.6
    
    # R² plot
    bars1 = axes[0].bar(x, r2_vals, width, color=['skyblue'] * (len(models) - 1) + ['orange'])
    axes[0].set_ylabel('R² Score', fontsize=11)
    axes[0].set_title('R² Score (higher is better)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # MAE plot
    bars2 = axes[1].bar(x, mae_vals, width, color=['skyblue'] * (len(models) - 1) + ['orange'])
    axes[1].set_ylabel('MAE', fontsize=11)
    axes[1].set_title('Mean Absolute Error (lower is better)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # RMSE plot
    bars3 = axes[2].bar(x, rmse_vals, width, color=['skyblue'] * (len(models) - 1) + ['orange'])
    axes[2].set_ylabel('RMSE', fontsize=11)
    axes[2].set_title('Root Mean Squared Error (lower is better)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle(f'Model Comparison: {target_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Metric comparison plot: {out_path}")


def ensure_directory_exists(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)
