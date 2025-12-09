.PHONY: train-advanced
train-advanced:
	@echo ">>> Training advanced models (XGBoost / LightGBM)..."
	python scripts/models/train_advanced.py --target all
	@echo ">>> Advanced training complete."

.PHONY: pipeline
pipeline:
	@echo ">>> Running full data-to-model pipeline..."
	python scripts/pipeline/run_pipeline.py
	@echo ">>> Pipeline run complete. Outputs saved to S3 and reports."