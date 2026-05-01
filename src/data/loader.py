"""
src/data/loader.py — Raw Data Loading and Inspection
=====================================================
Loads the combined clinical dataset, validates its schema, and produces
a structured inspection report consumed by Phase 1 of the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COL

REQUIRED_COLUMNS: tuple[str, ...] = (
    "age_years", "gender", "eye",
    "astig_value_D", "astig_axis_deg",
    "kmax_value_D", "kmax_axis_deg",
    "pachy_central_um", "pachy_thinnest_um",
    "pachy_thinnest_x", "pachy_thinnest_y",
    "asphericity_anterior", "asphericity_posterior",
    "label",
)


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw clinical dataset and validate the schema."""
    csv_path = Path(path) if path else RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")

    label_values = set(df[TARGET_COL].unique())
    if not label_values.issubset({0, 1}):
        raise ValueError(f"Label column must be binary; got {label_values}")

    return df


def inspection_report(df: pd.DataFrame) -> dict[str, Any]:
    """Build an inspection report describing dataset shape and quality."""
    label_counts = df[TARGET_COL].value_counts().sort_index().to_dict()
    total = len(df)

    return {
        "n_rows": total,
        "n_cols": df.shape[1],
        "label_counts": label_counts,
        "label_balance_pct": {
            int(k): round(100.0 * v / total, 2) for k, v in label_counts.items()
        },
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "age_range": (int(df["age_years"].min()), int(df["age_years"].max())),
        "gender_counts": df["gender"].value_counts().to_dict(),
        "eye_counts": df["eye"].value_counts().to_dict(),
    }
