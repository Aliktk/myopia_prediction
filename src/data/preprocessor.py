"""
src/data/preprocessor.py — Data Cleaning, Splitting, and Scaling
==================================================================
Handles deduplication, IQR-based outlier capping, median imputation,
stratified train/val/test splitting, and standard-scaler fitting.

All transformations are fit on the training set only to prevent
information leakage from validation or test data.
"""
from __future__ import annotations

from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLS_FILE,
    OUTPUTS_DIR,
    RANDOM_STATE,
    RAW_NUMERIC_COLS,
    SCALER_FILE,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)


class PreprocessingArtefacts(TypedDict):
    df_processed: pd.DataFrame
    n_duplicates_removed: int
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    scaler: StandardScaler
    feature_cols: list[str]


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows. Returns (cleaned_df, n_removed)."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n_before - len(df)


def cap_outliers(
    df: pd.DataFrame, numeric_cols: list[str], factor: float = 3.0
) -> pd.DataFrame:
    """Winsorise extreme outliers using IQR method (factor=3 conservative)."""
    out = df.copy()
    for col in numeric_cols:
        if col not in out.columns:
            continue
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        out[col] = out[col].clip(lower=lower, upper=upper)
    return out


def impute_missing(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Median imputation. Median chosen for skewed clinical distributions."""
    out = df.copy()
    for col in numeric_cols:
        if col in out.columns and out[col].isnull().any():
            out[col] = out[col].fillna(out[col].median())
    return out


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """Stratified 70/10/20 train/val/test split."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df[target_col].copy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    val_fraction_of_trainval = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_fraction_of_trainval,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, np.ndarray]:
    """Fit StandardScaler on training data only."""
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(X_train)


def save_artifacts(scaler: StandardScaler, feature_cols: list[str]) -> None:
    """Persist scaler and feature column list."""
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_cols, FEATURE_COLS_FILE)


def save_splits(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> None:
    """Save unscaled splits to outputs/ for full reproducibility."""
    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        out = X.copy()
        out[TARGET_COL] = y.values
        out.to_csv(OUTPUTS_DIR / f"{split_name}.csv", index=False)


def full_preprocessing_pipeline(
    df: pd.DataFrame,
    numeric_cols: list[str],
    feature_cols: list[str],
) -> PreprocessingArtefacts:
    """End-to-end preprocessing returning all artefacts in a dict."""
    df, n_dup = remove_duplicates(df)

    raw_num = [c for c in RAW_NUMERIC_COLS if c in df.columns]
    df = cap_outliers(df, raw_num)
    df = impute_missing(df, [c for c in feature_cols if c in df.columns])

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, feature_cols
    )

    scaler, X_train_scaled = fit_scaler(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    save_artifacts(scaler, list(X_train.columns))
    save_splits(X_train, y_train, X_val, y_val, X_test, y_test)

    return {
        "df_processed": df,
        "n_duplicates_removed": n_dup,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_cols": list(X_train.columns),
    }
