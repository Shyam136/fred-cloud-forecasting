# Data Dictionary and Provenance
Project: FRED Cloud Forecasting  
Author: Sandeep Kang  
Last Updated: November 2025  

---

## 1. Data Dictionary

This table lists all economic indicators used in the forecasting pipeline, along with frequency, units, transformation notes, and expected historical coverage.

| Series ID | Name | Frequency | Units | Transformations Used or Planned | Expected Coverage | Notes | Usage |
|-----------|------|-----------|--------|----------------------------------|-------------------|--------|--------|
| CPIAUCSL | Consumer Price Index | Monthly | Index (1982-84=100) | log, log-diff, pct_change, rolling means | 1947 to present | Main inflation indicator | Target for CPI models |
| UNRATE | Unemployment Rate | Monthly | Percent | pct_change | 1948 to present | Labor force pressure on prices | Regressor for CPI |
| WTISPLC or DCOILWTICO | Crude Oil Prices | Daily resampled to monthly | Dollars per barrel | pct_change, 3 to 12 month rolling means | 1986 to present | Energy cost and fuel inflation | Regressor for CPI |
| PPIACO | Producer Price Index | Monthly | Index | log, pct_change | 1913 to present | Producer-side inflation | Regressor for CPI |
| GDP | Gross Domestic Product | Quarterly resampled to monthly | Billions of USD | pct_change representing GDP growth | 1947 to present | Economic output | Target for GDP and regressor for CPI and FEDFUNDS |
| PCE | Personal Consumption Expenditures | Monthly | Billions of USD | pct_change | 1959 to present | Household spending | Regressor for GDP |
| FGCEC1 | Government Expenditures | Quarterly resampled to monthly | Billions of USD | pct_change | 1947 to present | Fiscal spending measure | Regressor for GDP |
| PAYEMS | Nonfarm Payroll Employment | Monthly | Thousands of persons | pct_change | 1939 to present | Employment and labor strength | Regressor for GDP |
| FEDFUNDS | Federal Funds Rate | Monthly | Percent | pct_change | 1954 to present | Monetary policy rate | Target for FEDFUNDS |
| GS10 | 10-Year Treasury Yield | Monthly | Percent | pct_change | 1962 to present | Long-term interest rates | Regressor for FEDFUNDS |
| PCEPILFE | Core PCE Inflation | Monthly | Index | log, pct_change | 1959 to present | Underlying inflation pressure | Regressor for FEDFUNDS |

---

## 2. Target to Regressor Mapping with Rationale

| Target Variable | Regressors Used | Rationale |
|-----------------|----------------|-----------|
| CPIAUCSL | UNRATE, WTISPLC or DCOILWTICO, PPIACO, GDP | Inflation influenced by labor market pressure, fuel prices, producer input costs, and total economic activity. |
| GDP | PCE, FGCEC1, PAYEMS | GDP is driven by consumption, government spending, and employment. |
| FEDFUNDS | CPIAUCSL, GDP, GS10, PCEPILFE | The Federal Funds Rate responds to inflation, GDP growth, long-term interest rates, and core inflation indicators. |

---

## 3. Provenance Summary

Data Source: Federal Reserve Economic Data (FRED)  
Access Method: fredapi Python library  

Pull Scripts:
- scripts/ingest/fred_ping.py  
- scripts/ingest/fred_bulk_fetch.py  

Raw Storage Path:  
s3://your-bucket-name/raw/<series_id>/  

Processed Storage Path:  
s3://your-bucket-name/processed/master_<yyyymmdd>.parquet  

Provenance Notes:
- All transformations such as resampling, interpolation, log transforms, and percent changes are applied in preprocessing.
- The feature engineering script adds lag features, rolling statistics, and growth indicators.

---

## 4. Version History

| Version | Date | Notes |
|---------|-------|--------|
| 1.0 | November 11, 2025 | Initial version |
| 2.0 | November 24, 2025 | Expanded with units, transformations, date coverage, and mapping table |