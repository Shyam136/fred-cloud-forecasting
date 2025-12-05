# ================================================================
# Makefile for FRED Cloud Forecasting Project
# Automates preprocessing, feature engineering, model training,
# and validation steps for reproducibility and CI integration.
# ================================================================

PYTHON := python
SCRIPTS_DIR := scripts
LOGS_DIR := logs
REPORTS_DIR := reports

$(shell mkdir -p $(LOGS_DIR) $(REPORTS_DIR))

.PHONY: preprocess
preprocess:
	@echo ">>> Starting preprocessing..."
	$(PYTHON) $(SCRIPTS_DIR)/preprocess/run_preprocess.py
	@echo ">>> Preprocessing complete. Outputs saved to S3 and logs."

.PHONY: features
features:
	@echo ">>> Generating features..."
	$(PYTHON) $(SCRIPTS_DIR)/features/feature_builder.py
	$(PYTHON) $(SCRIPTS_DIR)/checks/validate_features.py
	@echo ">>> Feature generation complete. Outputs validated and saved to S3."

.PHONY: train
train:
	@echo ">>> Training forecasting models..."
	$(PYTHON) $(SCRIPTS_DIR)/models/train_models.py --target all
	@echo ">>> Training complete. Metrics saved under $(REPORTS_DIR)/."

.PHONY: validate
validate:
	@echo ">>> Running validation suite..."
	pytest -v
	@echo ">>> Validation complete. All tests passed!"

.PHONY: lint
lint:
	@echo ">>> Checking code formatting..."
	black --check .
	flake8 .
	@echo ">>> Linting complete. No issues found."

.PHONY: help
help:
	@echo ""
	@echo "============================"
	@echo "FRED Cloud Forecasting Makefile"
	@echo "============================"
	@echo "Available commands:"
	@echo "  make preprocess   - Run data preprocessing pipeline"
	@echo "  make features     - Build features + validate outputs"
	@echo "  make train        - Train models (CPI, GDP, FEDFUNDS)"
	@echo "  make validate     - Run tests and validation checks"
	@echo "  make lint         - Run code formatting checks"
	@echo ""