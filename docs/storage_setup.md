# Storage Setup (AWS S3)

> **Status (11/10/2025)**
> - **Bucket:** `fred-forecasting-cloud-us-east-2`
> - **Region:** `us-east-2`
> - **Public access:** **Blocked** (default)
> - **Default encryption:** **SSE-S3 (AES-256)**
> - **Versioning:** Disabled (can enable later if needed)
> - **Lifecycle:** (optional) expire `logs/` after 30 days

## Naming Convention
We use a single private bucket per environment. For this course project:fred-forecasting-cloud-us-east-2