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
   Add your FRED API key as `FRED_API_KEY` in your `.env` file or system environment.

3. **Install dependencies**  
   Use `requirements.txt` (to be added) or set up a virtual environment.

4. **Run the ingestion script**  
   Navigate to `scripts/ingest/` and run `fred_ping.py` to test API access.

## 👥 Team
- Shyam Patel – Lead, ML, AWS EC2, Data Analysis
- Jesmin Sultana – ML, OpenAI/LLM, Causal Inference
- Noor Hassuneh – Frontend, Python, EDA
- Sandeep Singh – Spark, Data Analytics

---

