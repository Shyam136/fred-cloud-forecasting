
#!/usr/bin/env python3
"""
Validate Feature Artifacts
==========================

Checks:
- Row count > 0
- Null ratios per column
- Date range (min/max)
- Monotonic date order

Writes a Markdown summary log to logs/feature_checks_<date>.md
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == '.parquet':
        return pd.read_parquet(p)
    elif p.suffix.lower() == '.csv':
        return pd.read_csv(p)
    else:
        raise ValueError('Unsupported format. Use .parquet or .csv')

def summarize(df: pd.DataFrame, date_col: str = 'date') -> dict:
    if date_col not in df.columns:
        raise KeyError(f"date column '{date_col}' not found")
    out = {}
    out['row_count'] = int(len(df))
    d = pd.to_datetime(df[date_col])
    out['date_min'] = d.min().strftime('%Y-%m-%d') if len(d) else None
    out['date_max'] = d.max().strftime('%Y-%m-%d') if len(d) else None
    out['date_monotonic'] = bool(d.is_monotonic_increasing)
    nulls = df.isna().mean().to_dict()
    out['null_ratios'] = {k: float(v) for k, v in nulls.items()}
    return out

def write_markdown(summary: dict, input_path: str, log_dir: str = 'logs') -> Path:
    ts = datetime.now(timezone.utc).strftime('%Y%m%d')
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    out_path = p / f'feature_checks_{ts}.md'
    lines = [
        f"# Feature Validation Report ({ts})",
        "",
        f"**File:** {input_path}",
        f"**Rows:** {summary['row_count']}",
        f"**Date Range:** {summary['date_min']} → {summary['date_max']}",
        f"**Monotonic Dates:** {summary['date_monotonic']}",
        "",
        "## Null Ratios",
    ]
    for k, v in summary['null_ratios'].items():
        lines.append(f"- {k}: {v:.4f}")
    content = '\n'.join(lines)
    out_path.write_text(content)
    return out_path

def main():
    ap = argparse.ArgumentParser(description='Validate feature artifact and write Markdown report')
    ap.add_argument('--input', required=True, help='Feature file (.parquet or .csv)')
    ap.add_argument('--date-col', default='date', help='Date column name (default: date)')
    ap.add_argument('--log-dir', default='logs', help='Directory to write report (default: logs)')
    args = ap.parse_args()
    df = _read(args.input)
    summary = summarize(df, date_col=args.date_col)
    out_path = write_markdown(summary, args.input, log_dir=args.log_dir)
    print(f"✅ Validation written to {out_path}")

if __name__ == '__main__':
    main()