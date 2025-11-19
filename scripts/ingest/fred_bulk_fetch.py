#!/usr/bin/env python3
"""
Bulk fetch FRED series → S3 as Parquet/CSV.

Robust env handling:
- Loads .env if present (without overriding existing env)
- Reads FRED_API_KEY, S3_BUCKET, AWS_REGION/AWS_DEFAULT_REGION
- Allows CLI overrides for keys/bucket/region
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd

# --- dotenv: load .env if present, but do not override existing env ---
try:
    from dotenv import load_dotenv, find_dotenv
    # find .env in the CURRENT working directory (repo root)
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # python-dotenv not installed or .env not found; script can still run with CLI overrides
    pass

# Optional dependencies for Parquet & S3
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAVE_BOTO = True
except Exception:
    HAVE_BOTO = False


def get_env_or_cli(args):
    """Resolve configuration from CLI / environment / defaults."""
    api_key = args.fred_api_key or os.getenv("FRED_API_KEY")
    bucket  = args.s3_bucket   or os.getenv("S3_BUCKET")
    region  = args.region      or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-2"

    # Normalize values
    api_key = api_key.strip() if api_key else api_key
    bucket  = bucket.strip()  if bucket  else bucket
    region  = region.strip()  if region  else region

    missing = [name for name, val in [("FRED_API_KEY", api_key), ("S3_BUCKET", bucket)] if not val]
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        print("Tips:")
        print("  • Ensure you are in the repo root so .env is found")
        print("  • Check .env values are NOT empty (no quotes)")
        print("  • Or pass --fred-api-key and --s3-bucket on the CLI")
        sys.exit(1)

    if not HAVE_BOTO:
        print("ERROR: boto3 is required for S3 upload. Install with: pip install boto3", file=sys.stderr)
        sys.exit(1)

    return api_key, bucket, region


def fetch_series(series_id, api_key):
    """Fetch a single FRED series (date, value) as a DataFrame."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("observations", [])
    df = pd.DataFrame(data)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def save_and_upload(df, series, bucket, s3, outdir, fmt="parquet"):
    """Write to local file (Parquet/CSV) and upload to s3://bucket/raw/<series>/."""
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if fmt == "csv":
        local_path = outdir / f"{series}_{ts}.csv"
        df.to_csv(local_path, index=False)
        s3_key = f"raw/{series}/{series}_{ts}.csv"
    else:
        # Prefer Parquet; fall back to CSV if pyarrow is missing
        local_path = outdir / f"{series}_{ts}.parquet"
        try:
            df.to_parquet(local_path, index=False)
            s3_key = f"raw/{series}/{series}_{ts}.parquet"
        except Exception as e:
            print(f"Parquet write failed ({e}); falling back to CSV")
            local_path = outdir / f"{series}_{ts}.csv"
            df.to_csv(local_path, index=False)
            s3_key = f"raw/{series}/{series}_{ts}.csv"

    try:
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"✅ Uploaded {series} to s3://{bucket}/{s3_key}")
    except NoCredentialsError:
        print("❌ AWS credentials not found. Add Codespaces secrets or run `aws configure`.", file=sys.stderr)
        sys.exit(2)
    except ClientError as e:
        print(f"❌ S3 upload failed: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Bulk fetch FRED series → S3")
    parser.add_argument("--series-list", default="CPIAUCSL,GDP,FEDFUNDS",
                        help="Comma-separated list of FRED series IDs")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                        help="Output format. Parquet preferred; falls back to CSV if parquet engine missing.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Tail sample size; 0 = all rows")
    # CLI overrides for env vars
    parser.add_argument("--fred-api-key", default=None, help="Override FRED_API_KEY")
    parser.add_argument("--s3-bucket", default=None, help="Override S3_BUCKET")
    parser.add_argument("--region", default=None, help="Override AWS_REGION")
    args = parser.parse_args()

    api_key, bucket, region = get_env_or_cli(args)
    s3 = boto3.client("s3", region_name=region)

    # Default list; feel free to expand later
    series_list = [s.strip() for s in args.series_list.split(",") if s.strip()]
    outdir = Path("data/tmp")

    for series in series_list:
        df = fetch_series(series, api_key)
        if args.limit and args.limit > 0:
            df = df.tail(args.limit).reset_index(drop=True)
        save_and_upload(df, series, bucket, s3, outdir, fmt=args.format)


if __name__ == "__main__":
    main()