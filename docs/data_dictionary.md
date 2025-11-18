# Data Dictionary and Provenance
Project: Cloud-Based Forecasting of U.S. Macroeconomic Indicators Using FRED API and Distributed Cloud Tools  
Author: Sandeep Kang  
Last Updated: November 11, 2025  

---

## 1. Data Dictionary

| Series ID | Human Readable Name | Frequency | Units | Notes | Usage |
|------------|--------------------|------------|--------|--------|--------|
| CPIAUCSL | Consumer Price Index (All Urban Consumers, Seasonally Adjusted) | Monthly | Index (1982–84=100) | Measures changes in consumer prices and inflation | Target (Forecast CPI) |
| UNRATE | Unemployment Rate | Monthly | Percent | Indicator of labor market conditions; affects inflation | Regressor for CPI |
| WTISPLC / DCOILWTICO | Crude Oil Prices (West Texas Intermediate) | Daily (Resampled Monthly) | Dollars per Barrel | Energy price indicator influencing inflation trends | Regressor for CPI |
| PPIACO | Producer Price Index (All Commodities) | Monthly | Index (1982=100) | Measures average change in prices received by domestic producers | Regressor for CPI |
| GDP | Gross Domestic Product | Quarterly (Resampled Monthly) | Billions of USD | Measures total economic output; primary target variable | Target (Forecast GDP) and Regressor for CPI and FEDFUNDS |
| PCE | Personal Consumption Expenditures | Monthly | Billions of USD | Measures household spending and consumption | Regressor for GDP |
| FGCEC1 | Government Consumption Expenditures and Gross Investment | Quarterly (Resampled Monthly) | Billions of USD | Proxy for government spending | Regressor for GDP |
| PAYEMS | Total Nonfarm Payroll Employment | Monthly | Thousands of Persons | Employment measure related to economic activity | Regressor for GDP |
| FEDFUNDS | Federal Funds Effective Rate | Monthly | Percent | Benchmark interest rate for monetary policy | Target (Forecast FEDFUNDS) |
| GS10 | 10-Year Treasury Constant Maturity Rate | Monthly | Percent | Long-term interest rate used for yield curve analysis | Regressor for FEDFUNDS |
| PCEPILFE | Core Personal Consumption Expenditures (Excluding Food and Energy) | Monthly | Index (2012=100) | Core inflation measure used to gauge underlying inflation trends | Regressor for FEDFUNDS |

---

## 2. Regressor Relationships

| Forecast Target | External Regressors |
|------------------|---------------------|
| CPI (CPIAUCSL) | UNRATE, WTISPLC or DCOILWTICO, PPIACO, GDP |
| GDP | PCE, FGCEC1, PAYEMS, and optionally trade balance data |
| FEDFUNDS | CPIAUCSL, GDP, GS10, PCEPILFE |

Lag features (for example, CPI t-1, GDP t-1) and moving averages of target variables will also be created during feature engineering.

---

## 3. Provenance

Data Source: Federal Reserve Economic Data (FRED)  
URL: https://fred.stlouisfed.org/  
Access Method: fredapi Python library (version 0.5.0)  

Pull Scripts:  
- scripts/ingest/fred_ping.py (connectivity test)  
- scripts/ingest/fred_bulk_fetch.py (to be implemented for bulk ingestion)  

Storage Locations:  
- Raw Data: s3://your-bucket-name/raw/<series_id>/  
- Processed Data: s3://your-bucket-name/processed/master_<yyyymmdd>.parquet  

Date Pulled: November 2025  
Frequency of Update: Monthly (can be automated using AWS Lambda or a cron job)  

Attribution: Data provided by the Federal Reserve Bank of St. Louis (FRED API).  

---

## 4. Version History

| Version | Date | Description |
|----------|------|-------------|
| 1.0 | November 11, 2025 | Initial version created by Sandeep Kang |
| 1.1 | To be updated | Update with confirmed S3 paths, script versions, and provenance logs |