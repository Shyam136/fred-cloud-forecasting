#!/usr/bin/env python3
"""
scripts/ingest/fred_ping.py

Smoke test for FRED + S3 (GitHub Codespaces friendly):
- Reads FRED_API_KEY, AWS_REGION, S3_BUCKET from environment (or .env if present)
- Downloads observations for a series (default CPIAUCSL) using requests
- Saves a small CSV locally (./data/tmp) and uploads it to s3://<bucket>/raw/<series>/
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd

try:
    from dotenv import load_dotenv  # optional, if available
    load_dotenv()
except Exception:
    pass

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as e:
    print("boto3 is required for S3 upload. Add it to requirements.txt and pip install.", file=sys.stderr)
    raise

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, api_key: str, freq: str | None = None, limit: int = 25) -> pd.DataFrame:
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if freq:
        params["frequency"] = freq

    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    obs = payload.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        raise RuntimeError(f"No observations returned for series {series_id}")
    df = df[["date", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def upload_to_s3(local_path: Path, bucket: str, s3_key: str) -> None:
    s3 = boto3.client("s3")
    try:
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"✅ Uploaded to s3://{bucket}/{s3_key}")
    except ClientError as e:
        print(f"❌ S3 upload failed: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="CPIAUCSL", help="FRED series id")
    parser.add_argument("--freq", default=None, help="Optional frequency (e.g., m, q)")
    parser.add_argument("--limit", type=int, default=25, help="Tail sample size to keep")
    parser.add_argument("--outdir", default="data/tmp", help="Local output directory")
    args = parser.parse_args()

    api_key = os.getenv("FRED_API_KEY")
    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("AWS_REGION", "us-east-1")

    missing = [k for k, v in [("FRED_API_KEY", api_key), ("S3_BUCKET", bucket)] if not v]
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    df = fetch_series(args.series, api_key, args.freq, args.limit)

    # local write
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{args.series}_{ts}.csv"
    local_path = outdir / filename
    df.to_csv(local_path, index=False)
    print(df.head())
    print(f"💾 Saved local file: {local_path}")

    # s3 path: raw/<series>/<file>
    s3_key = f"raw/{args.series}/{filename}"
    upload_to_s3(local_path, bucket, s3_key)


if __name__ == "__main__":
    main()