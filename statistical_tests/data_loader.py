"""
data_loader.py
--------------
Shared data loading and preparation utilities for all statistical tests.
Loads individual run-level data from all_runs_master.json.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_run_data() -> pd.DataFrame:
    """
    Load individual run data from all_runs_master.json.
    Returns a flat DataFrame with one row per (model, size, quant, domain, run).
    """
    json_path = DATA_DIR / "all_runs_master.json"
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for entry in raw:
        row = {
            "model": entry["model"],
            "size": entry["size"],
            "quant": entry["quant"],
            "domain": entry["domain"],
            "run": entry["run"],
        }
        row.update(entry["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Numeric size (in billions)
    size_map = {"1.5B": 1.5, "3B": 3.0, "4B": 4.0, "7B": 7.0, "8B": 8.0}
    df["size_num"] = df["size"].map(size_map)

    # Configuration identifier
    df["config"] = df["model"] + "-" + df["size"] + "-" + df["quant"]
    df["model_size"] = df["model"] + "-" + df["size"]

    # Replace Infinity with NaN for safe computation
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def get_failure_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract failure-type counts for compositional analysis.
    Returns a DataFrame with columns for each failure category.
    """
    failure_cols = ["tle_failures", "if_failures", "ia_failures",
                    "completed_failures", "task_errors", "errors"]
    keep_cols = ["model", "size", "quant", "domain", "run",
                 "size_num", "config", "model_size"] + failure_cols
    out = df[keep_cols].copy()
    # Rename for clarity
    out.rename(columns={
        "tle_failures": "TLE",
        "if_failures": "IF",
        "ia_failures": "IA",
        "completed_failures": "CF",
        "task_errors": "TE",
        "errors": "SysErr",
    }, inplace=True)
    return out


def get_failure_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract failure-type rates for compositional analysis.
    """
    rate_cols = ["tle_rate", "if_rate", "ia_rate",
                 "completed_failure_rate", "task_error_rate", "error_rate"]
    keep_cols = ["model", "size", "quant", "domain", "run",
                 "size_num", "config", "model_size"] + rate_cols
    out = df[keep_cols].copy()
    out.rename(columns={
        "tle_rate": "TLE",
        "if_rate": "IF",
        "ia_rate": "IA",
        "completed_failure_rate": "CF",
        "task_error_rate": "TE",
        "error_rate": "SysErr",
    }, inplace=True)
    return out


def get_matched_data(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Pivot data so each row is a (model, size, domain, run) 'subject'
    and columns are quantization levels. Used for paired/matched tests.
    """
    pivoted = df.pivot_table(
        index=["model", "size", "domain", "run"],
        columns="quant",
        values=value_col,
        aggfunc="first"
    ).dropna()
    return pivoted


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_results(results: dict, filename: str):
    """Save results dict to JSON in the output directory."""
    out_dir = ensure_output_dir()
    out_path = out_dir / filename

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if pd.isna(obj):
            return None
        return obj

    import json as _json

    class NumpyEncoder(_json.JSONEncoder):
        def default(self, obj):
            converted = convert(obj)
            if converted is not obj:
                return converted
            return super().default(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    df = load_run_data()
    print(f"Loaded {len(df)} run-level observations")
    print(f"Models: {df['model'].unique()}")
    print(f"Sizes: {df['size'].unique()}")
    print(f"Quants: {df['quant'].unique()}")
    print(f"Domains: {df['domain'].unique()}")
    print(f"\nConfigurations ({df['config'].nunique()}):")
    for cfg in sorted(df["config"].unique()):
        n = len(df[df["config"] == cfg])
        print(f"  {cfg}: {n} runs")
