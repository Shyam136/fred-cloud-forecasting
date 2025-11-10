#!/usr/bin/env python3
"""
scripts/ingest/fred_ping.py

Minimal FRED API smoke test:
- Reads FRED_API_KEY from environment
- Downloads latest observations for a series (default CPIAUCSL)
- Prints a tiny sample and writes a small CSV/Parquet to ./data/tmp
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def fetch_series(series_id: str, api_key: str, freq: str = None, limit: int = 25) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if freq:
        params["frequency"] = freq

    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    obs = payload.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        raise RuntimeError(f"No observations returned for series {series_id}")
    # Keep only date and value; coerce value to float where possible
    df = df[["date", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if limit:
        df = df.tail(limit).reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="CPIAUCSL", help="FRED series id (default: CPIAUCSL)")
    parser.add_argument("--freq", default=None, help="Optional frequency (e.g., m, q)")
    parser.add_argument("--outdir", default="data/tmp", help="Output directory")
    args = parser.parse_args()

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("ERROR: FRED_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)

    df = fetch_series(args.series, api_key, args.freq)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_path = Path(args.outdir) / f"{args.series}_{ts}.csv"
    df.to_csv(csv_path, index=False)

    print(df.head())
    print(f"\nWrote sample to {csv_path}")

if __name__ == "__main__":
    main()