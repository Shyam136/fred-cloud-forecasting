# Infrastructure Decision Document

## 1. Purpose
This document describes our cloud setup for the Cloud Group Project. The infrastructure supports:
- Data ingestion from the FRED API  
- Storage in AWS S3  
- Processing with Apache Spark  
- Serving results through a Flask API hosted on EC2.

## 2. EC2 Instance Type
- **Chosen Type:** t2.micro (Free Tier)
- **Specs:** 1 vCPU, 1 GB RAM
- **Reasoning:** Cost-efficient and sufficient for testing Spark jobs and Flask API under light load.
- **Scalability Plan:** Upgrade to t3.medium for production-level processing.

## 3. Operating System
- **Chosen OS:** Ubuntu 22.04 LTS on AWS EC2
- **Reasoning:** Stable, lightweight, compatible with Spark, Python, and AWS CLI.

## 4. Cost Considerations
- **EC2 Free Tier:** 750 hours/month for one t2.micro instance.  
- **S3 Storage:** Minimal for small datasets (<5 GB).  
- **Data Transfer:** Negligible during testing phase.  
- Estimated monthly cost ≈ **$0 - $3**, within Free Tier limits.

## 5. Expected Usage
- Used mainly for development and demonstration.  
- EC2 instance hosts both Spark and Flask.  
- Expected uptime: ~30 hrs/week.

## 6. Security Notes
- Use IAM roles with least privileges for S3 access.  
- Store credentials securely using environment variables (`.env` file, never commit keys).
