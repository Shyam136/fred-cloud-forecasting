# FRED Cloud Forecasting Runbook

## 1. Data Refresh
- Trigger: schedule or manual run
- Command: `make preprocess`
- Steps:
  1. Pull latest FRED series via ingestion script.
  2. Upload raw datasets to `s3://<bucket>/raw/`.
  3. Log fetch summary in `logs/bulk_fetch_<date>.md`.

## 2. Feature Build
- Command: `make features`
- Steps:
  1. Generate lag, rolling, and pct_change features.
  2. Save artifacts to `s3://<bucket>/processed/features_*_<date>.parquet`.
  3. Validate using `scripts/checks/validate_features.py`.

## 3. Model Training
- Command: `make train`
- Steps:
  1. Train models for CPI, GDP, FEDFUNDS.
  2. Save metrics → `reports/model_report_<date>.json`.

## 4. Model Validation
- Command: `make validate`
- Steps:
  1. Run unit tests.
  2. Execute CI linting (Black/Flake8, pytest).
  3. Confirm outputs and report in PR.

## 5. Automation
- CI workflow (.github/workflows/ci.yml)
- Nightly ingestion trigger (optional cron job)

## 6. End-to-End Workflow
The forecasting workflow is fully automated:
1. make preprocess
2. make features
3. make train
4. make train-advanced
5. make validate
6. make pipeline

## 7. Reproducibility and CI/CD
Every step can be re-run reproducibly using the Makefile.
- Lint + tests run on pull requests.
- The full pipeline runs automatically on push to main.
- This ensures continuous integration and delivery of fresh forecasts.
---