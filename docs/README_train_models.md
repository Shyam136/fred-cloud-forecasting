# Model Training & Report Generation

This note explains how to train baseline models (CPI, GDP, FEDFUNDS) and how reports are generated.

## 1. Training Models

Run the training script from the project root:

```bash
python scripts/models/train_models.py --target CPI
python scripts/models/train_models.py --target GDP
python scripts/models/train_models.py --target FEDFUNDS
```

Each command:
- Loads the corresponding processed feature file
- Trains **Linear Regression** and **Random Forest**
- Computes metrics (**R², MAE, RMSE**)
- Saves a JSON report

## 2. Required Input Files

Place these files inside the `data/` folder:

```
data/features_cpi.parquet
data/features_gdp.parquet
data/features_fedfunds.parquet
```

## 3. Output Reports

Each run generates a file:

```
reports/model_report_YYYYMMDD.json
```

This JSON contains:

```json
{
  "target": "CPI",
  "linear_regression": {
    "r2": ...,
    "mae": ...,
    "rmse": ...
  },
  "random_forest": {
    "r2": ...,
    "mae": ...,
    "rmse": ...
  }
}
```

## 4. Notes

- Datetime columns are automatically removed before training.
- The script uses a **time-series friendly** 80/20 split.
- Reports are overwritten daily but stored by date.



---

This file summarizes the training and reporting workflow for your Cloud Forecasting project.
