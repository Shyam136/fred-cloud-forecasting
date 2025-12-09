#!/bin/bash
# Week 5 - Run all advanced models
# Usage: bash scripts/models/run_all_models.sh

set -e  # Exit on error

echo "=================================================="
echo "WEEK 5: Training All Advanced Models"
echo "=================================================="
echo ""

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$SCRIPT_DIR"

# Date for output files
DATE=$(date -u +%Y%m%d)

echo "Output date: $DATE"
echo "Project root: $PROJECT_ROOT"
echo ""

# Configuration
INPUT="${PROJECT_ROOT}/processed/master_transformed_20251203.parquet"
METRICS="${PROJECT_ROOT}/reports/model_comparison_${DATE}.json"
FIGS="${PROJECT_ROOT}/docs/figures/week5/"

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file not found: $INPUT"
    echo "Please ensure the processed data file exists."
    exit 1
fi

echo "=================================================="
echo "1/6: Training CPI with XGBoost"
echo "=================================================="
python train_advanced.py \
    --input "$INPUT" \
    --target CPI \
    --model xgboost \
    --out-metrics "$METRICS" \
    --out-figs "$FIGS"

echo ""
echo "=================================================="
echo "2/6: Training CPI with LightGBM"
echo "=================================================="
python train_advanced.py \
    --input "$INPUT" \
    --target CPI \
    --model lightgbm \
    --out-metrics "${PROJECT_ROOT}/reports/model_comparison_cpi_lgb_${DATE}.json" \
    --out-figs "$FIGS"

echo ""
echo "=================================================="
echo "3/6: Training GDP with XGBoost + Features"
echo "=================================================="
GDP_FEATURES="${PROJECT_ROOT}/processed/features_gdp_20251126.parquet"
if [ -f "$GDP_FEATURES" ]; then
    python train_advanced.py \
        --input "$INPUT" \
        --features "$GDP_FEATURES" \
        --target GDP \
        --model xgboost \
        --out-metrics "$METRICS" \
        --out-figs "$FIGS"
else
    echo "SKIP: GDP features file not found"
    python train_advanced.py \
        --input "$INPUT" \
        --target GDP \
        --model xgboost \
        --out-metrics "$METRICS" \
        --out-figs "$FIGS"
fi

echo ""
echo "=================================================="
echo "4/6: Training GDP with LightGBM + Features"
echo "=================================================="
if [ -f "$GDP_FEATURES" ]; then
    python train_advanced.py \
        --input "$INPUT" \
        --features "$GDP_FEATURES" \
        --target GDP \
        --model lightgbm \
        --out-metrics "${PROJECT_ROOT}/reports/model_comparison_gdp_lgb_${DATE}.json" \
        --out-figs "$FIGS"
else
    echo "SKIP: GDP features file not found"
    python train_advanced.py \
        --input "$INPUT" \
        --target GDP \
        --model lightgbm \
        --out-metrics "${PROJECT_ROOT}/reports/model_comparison_gdp_lgb_${DATE}.json" \
        --out-figs "$FIGS"
fi

echo ""
echo "=================================================="
echo "5/6: Training FEDFUNDS with XGBoost"
echo "=================================================="
python train_advanced.py \
    --input "$INPUT" \
    --target FEDFUNDS \
    --model xgboost \
    --test-size 0.25 \
    --out-metrics "$METRICS" \
    --out-figs "$FIGS"

echo ""
echo "=================================================="
echo "6/6: Training FEDFUNDS with LightGBM"
echo "=================================================="
python train_advanced.py \
    --input "$INPUT" \
    --target FEDFUNDS \
    --model lightgbm \
    --test-size 0.25 \
    --out-metrics "${PROJECT_ROOT}/reports/model_comparison_fedfunds_lgb_${DATE}.json" \
    --out-figs "$FIGS"

echo ""
echo "=================================================="
echo "✓ ALL MODELS TRAINED SUCCESSFULLY"
echo "=================================================="
echo ""
echo "Outputs:"
echo "  - Metrics: ${PROJECT_ROOT}/reports/model_comparison_*.json"
echo "  - Figures: ${FIGS}"
echo ""
echo "Generated files:"
ls -lh "${FIGS}" | grep "${DATE}" || echo "  (Using default timestamp)"
echo ""
