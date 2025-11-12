# FRED Cloud Forecasting

## 📌 Project Overview
This project forecasts U.S. macroeconomic indicators (CPI, GDP, Federal Funds Rate) using data from the FRED API.  
The pipeline uses **Apache Spark** for distributed processing and deploys models on **AWS EC2**, storing data in **AWS S3**.

## ✅ Key Features
- Automated data ingestion from FRED API
- Scalable preprocessing and feature engineering using Spark
- Time series forecasting models (Prophet, Regression, optional LSTM)
- Deployment of API endpoints on AWS EC2
- Visualization dashboards for real-time insights

## ⚙️ Setup Instructions
1. **Clone the repository**  
   `git clone https://github.com/<your-username>/fred-cloud-forecasting.git`

2. **Set up environment variables**
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your credentials:
     ```
     FRED_API_KEY=your_api_key_here
     AWS_REGION=your_aws_region
     S3_BUCKET=your_s3_bucket_name
     ```
   - Get a FRED API key from: [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

3. **Install dependencies**  
   ```bash
   pip install pandas requests boto3 python-dotenv pyarrow
   ```
   Or use the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the ingestion script**  
   ```bash
   python scripts/ingest/fred_ping.py  # Test API access
   python scripts/ingest/fred_fetch.py  # Run bulk data fetch
   ```

## 🚀 Quick Start (Bulk Fetch)

### Fetching Data in Bulk
Use the bulk fetch script to download and process multiple economic indicators from FRED:

```bash
# Example 1: Fetch specific series
python scripts/ingest/fred_fetch.py --series_id CPIAUCSL,GDP,FEDFUNDS --start_date 2000-01-01 --output_format csv

# Example 2: Fetch and save to S3 (requires AWS credentials configured)
python scripts/ingest/fred_fetch.py --series_id UNRATE --start_date 2010-01-01 --output_format parquet --s3_upload

# Example 3: Get help with available options
python scripts/ingest/fred_fetch.py --help
```

### Output Formats
- CSV: Human-readable format for local analysis
- Parquet: Optimized for big data processing and storage
- S3: Direct upload to configured AWS S3 bucket

### Required Environment Variables
- `FRED_API_KEY`: Your FRED API key
- `AWS_REGION`: AWS region (e.g., us-east-1)
- `S3_BUCKET`: Target S3 bucket for data storage (if using S3 upload)

### 🔧 Troubleshooting

#### Authentication Issues
- **Missing Credentials**: Ensure your AWS credentials are configured via:
  - Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
  - AWS credentials file (`~/.aws/credentials`)
  - IAM role (if running on AWS)

- **Invalid FRED API Key**: Verify your API key is correctly set in `.env` and has proper permissions

#### Common Errors
- **AccessDenied when accessing S3**:
  - Check IAM user permissions (needs `s3:PutObject` and `s3:PutObjectAcl`)
  - Verify the S3 bucket name is correct and exists
  - Ensure your IAM user/role has access to the specified S3 bucket

- **InvalidRegionError**:
  - Verify `AWS_REGION` in `.env` matches your S3 bucket's region
  - Common regions: `us-east-1`, `us-west-2`, `eu-west-1`

- **NoSuchBucket**:
  - Check for typos in the `S3_BUCKET` name
  - Ensure the bucket exists in the specified region

## 👥 Team
- Shyam Patel
- Jesmin Sultana
- Noor Hassuneh
- Sandeep Kang

---

