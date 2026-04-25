"""
Agent 01 — Data Analyzer
========================
Responsible for: loading the raw dataset, producing a full inspection
report (schema, stats, missing values, duplicates, class balance) and
logging any anomalies that downstream agents need to handle.

Run standalone: python agents/agent_01_data_analyzer.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw, inspection_report
from src.config import RAW_DATA_PATH
import pandas as pd
import numpy as np
from scipy import stats


def run():
    print("=" * 60)
    print("AGENT 01 — DATA ANALYZER")
    print("=" * 60)

    df = load_raw(RAW_DATA_PATH)
    rep = inspection_report(df)

    print(f"\n{'─'*50}")
    print("DATASET OVERVIEW")
    print(f"{'─'*50}")
    print(f"  Rows        : {rep['n_rows']}")
    print(f"  Columns     : {rep['n_cols']}")
    print(f"  Label 0     : {rep['label_counts'].get(0, 0)} ({rep['label_balance_pct'].get(0, 0):.1f}%)")
    print(f"  Label 1     : {rep['label_counts'].get(1, 0)} ({rep['label_balance_pct'].get(1, 0):.1f}%)")
    print(f"  Missing     : {sum(rep['missing_values'].values())} cells")
    print(f"  Duplicates  : {rep['duplicate_rows']}")
    print(f"  Age range   : {rep['age_range'][0]}–{rep['age_range'][1]} years")

    print(f"\n{'─'*50}")
    print("DESCRIPTIVE STATISTICS")
    print(f"{'─'*50}")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    desc = df[numeric_cols].describe().round(3)
    print(desc.to_string())

    print(f"\n{'─'*50}")
    print("OUTLIER ANALYSIS (IQR METHOD)")
    print(f"{'─'*50}")
    for col in [c for c in numeric_cols if c != "label"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
        pct = n_out / len(df) * 100
        print(f"  {col:35s}: {n_out:3d} outliers ({pct:.1f}%)")

    print(f"\n{'─'*50}")
    print("NORMALITY TESTS (Shapiro-Wilk, p-value)")
    print(f"{'─'*50}")
    for col in [c for c in numeric_cols if c != "label"]:
        sample = df[col].dropna().sample(min(100, len(df)), random_state=42)
        _, p = stats.shapiro(sample)
        dist = "Normal" if p > 0.05 else "Non-normal"
        print(f"  {col:35s}: p={p:.4f}  -> {dist}")

    print(f"\n[Agent 01 Complete]")


if __name__ == "__main__":
    run()
