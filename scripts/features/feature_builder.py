
#!/usr/bin/env python3
"""
Feature Builder (Pandas / Spark-friendly)
=========================================

This module creates time-series features for a single target column from an
already aligned monthly dataset (produced in preprocessing).

Features:
- Lags: t-1, t-3, t-6, t-12 (configurable)
- Moving averages: 3, 6, 12 months (configurable)
- Percent change (month-over-month)
- Rolling standard deviation (configurable windows)

CLI supports reading a local CSV/Parquet input, generating features, writing a
versioned output under `processed/`, and (optionally) uploading to S3.
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import pandas as pd

# Optional dependencies
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    HAVE_BOTO = True
except Exception:
    HAVE_BOTO = False

def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError("Unsupported input format. Use .parquet or .csv")

def _write_table(df: pd.DataFrame, path: Path, fmt: str = "parquet") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        final_path = path.with_suffix(".csv")
        df.to_csv(final_path, index=False)
        return final_path
    try:
        final_path = path.with_suffix(".parquet")
        df.to_parquet(final_path, index=False)
        return final_path
    except Exception as e:
        sys.stderr.write(f"Parquet write failed ({e}); falling back to CSV\n")
        final_path = path.with_suffix(".csv")
        df.to_csv(final_path, index=False)
        return final_path

def _upload_s3(local_path: Path, bucket: str, key: str, region: Optional[str] = None) -> None:
    if not HAVE_BOTO:
        raise RuntimeError("boto3 not available. Install boto3 to enable S3 upload.")
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.upload_file(str(local_path), bucket, key)
    except NoCredentialsError:
        raise RuntimeError("AWS credentials not found. Configure via environment or AWS CLI.")
    except ClientError as e:
        raise RuntimeError(f"S3 upload failed: {e}")

def build_target_features(
    df: pd.DataFrame,
    target_col: str,
    date_col: str = "date",
    lags: List[int] = [1, 3, 6, 12],
    ma_windows: List[int] = [3, 6, 12],
    std_windows: List[int] = [3, 6, 12],
    add_pct_change: bool = True,
) -> pd.DataFrame:
    if date_col not in df.columns:
        raise KeyError(f"date column '{date_col}' not found")
    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' not found")

    out = df[[date_col, target_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(by=date_col).reset_index(drop=True)
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")

    for L in sorted(set(lags)):
        out[f"{target_col}_lag{L}"] = out[target_col].shift(L)

    for W in sorted(set(ma_windows)):
        out[f"{target_col}_ma{W}"] = out[target_col].rolling(window=W, min_periods=1).mean()

    for W in sorted(set(std_windows)):
        out[f"{target_col}_std{W}"] = out[target_col].rolling(window=W, min_periods=1).std()

    if add_pct_change:
        out[f"{target_col}_pct_change"] = out[target_col].pct_change()

    return out

def _compose_output_name(target: str, outdir: Path, tag_date: Optional[str] = None) -> Path:
    if tag_date is None:
        tag_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    basename = f"features_{target.lower()}_{tag_date}"
    return outdir / basename

def run_pipeline(
    input_path: str,
    target: str,
    date_col: str = "date",
    outdir: str = "processed",
    fmt: str = "parquet",
    lags: List[int] = [1, 3, 6, 12],
    ma_windows: List[int] = [3, 6, 12],
    std_windows: List[int] = [3, 6, 12],
    add_pct_change: bool = True,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "processed/",
    tag_date: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[pd.DataFrame, Path, Optional[str]]:
    df = _read_table(input_path)
    feats = build_target_features(
        df,
        target_col=target,
        date_col=date_col,
        lags=lags,
        ma_windows=ma_windows,
        std_windows=std_windows,
        add_pct_change=add_pct_change,
    )
    outdir_p = Path(outdir)
    local_base = _compose_output_name(target, outdir_p, tag_date=tag_date)
    local_path = _write_table(feats, local_base, fmt=fmt)

    s3_uri = None
    if s3_bucket:
        if not HAVE_BOTO:
            raise RuntimeError("boto3 not available for S3 upload")
        key = f"{s3_prefix.rstrip('/')}/{local_base.name}{local_path.suffix}"
        _upload_s3(local_path, s3_bucket, key, region=region)
        s3_uri = f"s3://{s3_bucket}/{key}"
    return feats, local_path, s3_uri

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build target features and (optionally) upload to S3")
    p.add_argument("--input", required=True, help="Input CSV/Parquet path (aligned monthly dataset)")
    p.add_argument("--target", required=True, help="Target column (e.g., CPIAUCSL)")
    p.add_argument("--date-col", default="date", help="Date column name (default: date)")
    p.add_argument("--outdir", default="processed", help="Local output directory (default: processed)")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output format")
    p.add_argument("--lags", default="1,3,6,12", help="Comma-separated lag periods")
    p.add_argument("--ma", default="3,6,12", help="Comma-separated moving average windows")
    p.add_argument("--std", default="3,6,12", help="Comma-separated rolling std windows")
    p.add_argument("--no-pct-change", action="store_true", help="Disable percent change feature")
    # S3 options
    p.add_argument("--s3-bucket", default=None, help="S3 bucket for upload (optional)")
    p.add_argument("--s3-prefix", default="processed/", help="S3 key prefix (default: processed/)")
    p.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or None,
                   help="AWS region (default from env)")
    p.add_argument("--tag-date", default=None, help="Override yyyymmdd tag for output filename")
    return p.parse_args()

def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]

def main() -> None:
    args = _parse_args()
    feats, local_path, s3_uri = run_pipeline(
        input_path=args.input,
        target=args.target,
        date_col=args.date_col,
        outdir=args.outdir,
        fmt=args.format,
        lags=_parse_int_list(args.lags),
        ma_windows=_parse_int_list(args.ma),
        std_windows=_parse_int_list(args.std),
        add_pct_change=not args.no_pct_change,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        tag_date=args.tag_date,
        region=args.region,
    )
    print(f"\n✅ Features built for {args.target}: {local_path}")
    if s3_uri:
        print(f"✅ Uploaded to {s3_uri}")

if __name__ == "__main__":
    main()