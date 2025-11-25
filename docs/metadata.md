# Metadata and Provenance Log
Project: FRED Cloud Forecasting  
Author: Sandeep Kang  
Last Updated: November 2025  

---

## 1. Purpose

This document records metadata for both raw and processed datasets in the project.  
It includes timestamps, script names and versions, storage paths, and instructions for generating provenance.

---

## 2. Data Pull Metadata (Raw Data)

This table logs when data was pulled from the FRED API, which script was used, and where the raw data was stored.

| Series ID | Pulled At | Script Used | Script Version | Raw S3 Path |
|-----------|-----------|--------------|----------------|--------------|
| CPIAUCSL | To be recorded | fred_bulk_fetch.py | v1 | s3://your-bucket-name/raw/CPIAUCSL/ |
| GDP | To be recorded | fred_bulk_fetch.py | v1 | s3://your-bucket-name/raw/GDP/ |
| FEDFUNDS | To be recorded | fred_bulk_fetch.py | v1 | s3://your-bucket-name/raw/FEDFUNDS/ |
| UNRATE | To be recorded | fred_bulk_fetch.py | v1 | s3://your-bucket-name/raw/UNRATE/ |
| Additional series | To be recorded | fred_bulk_fetch.py | v1 | s3://your-bucket-name/raw/<series>/ |

---

## 3. Processed Artifact Metadata

After feature engineering and preprocessing, all datasets should be documented as follows:

| Artifact Name | Generated At | Script Used | Notes | Processed S3 Path |
|---------------|--------------|--------------|--------|---------------------|
| master_wide_<date>.parquet | To be recorded | preprocess.ipynb or preprocess.py | Fully merged aligned dataset | s3://your-bucket-name/processed/ |
| features_cpi_<date>.parquet | To be recorded | feature_builder.py | CPI model features | s3://your-bucket-name/processed/features_cpi/ |
| features_gdp_<date>.parquet | To be recorded | feature_builder.py | GDP model features | s3://your-bucket-name/processed/features_gdp/ |
| features_fedfunds_<date>.parquet | To be recorded | feature_builder.py | FEDFUNDS model features | s3://your-bucket-name/processed/features_fedfunds/ |

---

## 4. Provenance Requirements

Each processed dataset must include the following provenance fields:

- source  
- generated_at  
- script  
- version  
- series_included  
- row_count  
- date_range  

These fields may be stored inside the Parquet metadata or in a JSON sidecar file.

All JSON metadata files must be placed in:

processed/metadata/

Filename format:

eatures_cpi_.json
features_gdp_.json
features_fedfunds_.json

---

## 5. Provenance Header Instructions

The following instructions describe how to attach provenance information to processed datasets.

### Option A: Store provenance inside DataFrame attributes

This method attaches metadata to the DataFrame before saving as Parquet.

Example:

df.attrs[“source”] = “FRED API”
df.attrs[“generated_at”] = timestamp
df.attrs[“script”] = “feature_builder.py”
df.attrs[“version”] = git_hash
df.attrs[“series_included”] = series_list

df.to_parquet(“processed/features_cpi_.parquet”)

### Option B: Store provenance in a JSON sidecar file (recommended)

This method writes a separate JSON file matching each processed dataset.

Example content:

{
“target”: “CPIAUCSL”,
“generated_at”: “”,
“script”: “feature_builder.py”,
“version”: “”,
“source”: “FRED API”,
“series_used”: [“CPIAUCSL”, “UNRATE”, “WTISPLC”, “PPIACO”, “GDP”],
“row_count”: “”,
“date_range”: { “start”: “”, “end”: “” }
}

All JSON files must be placed in:

processed/metadata/

This approach keeps metadata separate, easier to maintain, and readable for auditing.

---

## 6. Folder Structure Requirements

The following folders must exist in the repository:

processed/
processed/metadata/

The processed folder stores final model-ready datasets.  
The metadata folder stores all JSON provenance files.

---

## 7. Version History

| Version | Date | Notes |
|---------|--------|--------|
| 1.0 | November 2025 | Initial structure created |
| 1.1 | November 2025 | Added provenance header instructions |

