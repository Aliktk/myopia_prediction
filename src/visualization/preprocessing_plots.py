"""
src/visualization/preprocessing_plots.py — Preprocessing Visualisations
========================================================================
Visualises every preprocessing step so reviewers and readers can see
exactly what was done to the raw data: missing values, outliers,
capping effect, SMOTE rebalancing, splits, and scaling.

All figures use the publication PUB_RC styling for clarity.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    COLOR_NEG,
    COLOR_POS,
    COLORS,
    DPI_PUBLICATION,
    DPI_SCREEN,
    FIG_PREPROCESSING,
    PUB_RC,
)


def _save(fig: plt.Figure, path: Path, dpi: int = DPI_SCREEN) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# 1. Class distribution before vs. after SMOTE
# ──────────────────────────────────────────────────────────────────────────

def plot_class_distribution_smote(
    y_before: np.ndarray,
    y_after: np.ndarray,
    out_dir: Path = FIG_PREPROCESSING,
) -> None:
    labels = ["Non-Progressive (0)", "Progressive (1)"]
    counts_before = [int((y_before == 0).sum()), int((y_before == 1).sum())]
    counts_after = [int((y_after == 0).sum()), int((y_after == 1).sum())]

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, counts, title in [
            (axes[0], counts_before, "Before SMOTE"),
            (axes[1], counts_after, "After SMOTE"),
        ]:
            bars = ax.bar(
                labels, counts, color=COLORS,
                alpha=0.92, width=0.55, edgecolor="black", linewidth=0.8,
            )
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{count:,}",
                    ha="center", va="bottom",
                    fontsize=14, fontweight="bold",
                )
            ax.set_title(title)
            ax.set_ylabel("Sample Count")
            ax.set_ylim(0, max(counts) * 1.18)

        fig.suptitle("Class Distribution: Before vs After SMOTE")
        plt.tight_layout()
        _save(fig, out_dir / "01_class_distribution_smote.png")
    print("  [OK] Class distribution (SMOTE)")


# ──────────────────────────────────────────────────────────────────────────
# 2. Missing values heatmap + counts
# ──────────────────────────────────────────────────────────────────────────

def plot_missing_values(
    df_raw: pd.DataFrame, out_dir: Path = FIG_PREPROCESSING,
) -> None:
    missing = df_raw.isnull()
    total_missing = missing.sum()

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))

        if missing.any().any():
            sns.heatmap(missing, cbar=False, cmap="Reds", ax=axes[0])
            axes[0].set_title("Missing Values Heatmap (red = missing)")
        else:
            axes[0].text(
                0.5, 0.5, "No Missing Values Detected",
                ha="center", va="center",
                fontsize=20, color="green",
                fontweight="bold", transform=axes[0].transAxes,
            )
            axes[0].set_title("Missing Values Heatmap")
            axes[0].axis("off")

        axes[1].bar(
            total_missing.index, total_missing.values,
            color=COLOR_NEG, alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        axes[1].set_title("Missing Values per Column")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=40)
        for label in axes[1].get_xticklabels():
            label.set_horizontalalignment("right")

        fig.suptitle("Missing Value Analysis")
        plt.tight_layout()
        _save(fig, out_dir / "02_missing_values.png")
    print("  [OK] Missing values")


# ──────────────────────────────────────────────────────────────────────────
# 3. Outlier detection (IQR scatter)
# ──────────────────────────────────────────────────────────────────────────

def plot_outlier_detection(
    df_raw: pd.DataFrame,
    numeric_cols: list[str],
    out_dir: Path = FIG_PREPROCESSING,
) -> None:
    ncols = 4
    nrows = int(np.ceil(len(numeric_cols) / ncols))

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.5 * nrows))
        axes = axes.ravel()

        for i, col in enumerate(numeric_cols):
            q1, q3 = df_raw[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            is_out = (df_raw[col] < lower) | (df_raw[col] > upper)
            n_out = int(is_out.sum())

            axes[i].scatter(
                range(len(df_raw)), df_raw[col],
                c=[COLOR_POS if o else COLOR_NEG for o in is_out],
                alpha=0.45, s=10,
            )
            axes[i].set_title(f"{col}\n(outliers = {n_out})", fontsize=12)
            axes[i].axhline(upper, color="red", ls="--", lw=1.0, alpha=0.7)
            axes[i].axhline(lower, color="red", ls="--", lw=1.0, alpha=0.7)
            axes[i].tick_params(labelsize=10)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Outlier Detection (IQR Method) — red points = outliers, dashed lines = IQR fence", y=1.01)
        plt.tight_layout()
        _save(fig, out_dir / "03_outlier_detection.png")
    print("  [OK] Outlier detection")


# ──────────────────────────────────────────────────────────────────────────
# 4. Before vs after outlier capping
# ──────────────────────────────────────────────────────────────────────────

def plot_before_after_capping(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    cols: list[str],
    out_dir: Path = FIG_PREPROCESSING,
) -> None:
    display_cols = [c for c in cols[:8] if c in df_raw.columns and c in df_processed.columns]

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(2, len(display_cols), figsize=(3 * len(display_cols), 9))
        for j, col in enumerate(display_cols):
            for row, df_, title, color in [
                (0, df_raw, "Raw", COLOR_NEG),
                (1, df_processed, "Capped", COLOR_POS),
            ]:
                axes[row, j].boxplot(
                    df_[col].dropna(), patch_artist=True,
                    boxprops={"facecolor": color, "alpha": 0.75, "linewidth": 1.2},
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0},
                )
                axes[row, j].set_title(f"{title}\n{col}", fontsize=11)
                axes[row, j].tick_params(axis="y", labelsize=9)
                axes[row, j].tick_params(axis="x", labelbottom=False)

        fig.suptitle("Before vs After Outlier Capping (IQR × 3)")
        plt.tight_layout()
        _save(fig, out_dir / "04_outlier_capping_comparison.png")
    print("  [OK] Outlier capping comparison")


# ──────────────────────────────────────────────────────────────────────────
# 5. Train/Val/Test split distribution
# ──────────────────────────────────────────────────────────────────────────

def plot_split_distribution(
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    out_dir: Path = FIG_PREPROCESSING,
) -> None:
    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        for ax, y, name in [
            (axes[0], y_train, "Train"),
            (axes[1], y_val, "Validation"),
            (axes[2], y_test, "Test"),
        ]:
            unique, counts = np.unique(y, return_counts=True)
            wedges, texts, autotexts = ax.pie(
                counts, labels=["Non-Prog", "Progressive"],
                colors=COLORS,
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 13, "fontweight": "bold"},
                wedgeprops={"linewidth": 1.8, "edgecolor": "white"},
            )
            for at in autotexts:
                at.set_color("white")
                at.set_fontsize(13)
            ax.set_title(f"{name}  (n = {len(y):,})")

        fig.suptitle("Label Distribution Across Train / Validation / Test Splits")
        plt.tight_layout()
        _save(fig, out_dir / "05_split_distribution.png")
    print("  [OK] Split distribution")


# ──────────────────────────────────────────────────────────────────────────
# 6. Scaling comparison
# ──────────────────────────────────────────────────────────────────────────

def plot_scaling_comparison(
    X_train: pd.DataFrame, X_train_scaled: np.ndarray,
    feature_cols: list[str],
    out_dir: Path = FIG_PREPROCESSING,
) -> None:
    n_show = min(6, len(feature_cols))
    show_cols = feature_cols[:n_show]
    X_scaled_df = pd.DataFrame(X_train_scaled[:, :n_show], columns=show_cols)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        for j, col in enumerate(show_cols):
            axes[0, j].hist(
                X_train[col].dropna(), bins=30,
                color=COLOR_NEG, alpha=0.85, edgecolor="black", linewidth=0.4,
            )
            axes[0, j].set_title(f"Raw\n{col}", fontsize=11)
            axes[0, j].set_ylabel("Count" if j == 0 else "")

            axes[1, j].hist(
                X_scaled_df[col], bins=30,
                color=COLOR_POS, alpha=0.85, edgecolor="black", linewidth=0.4,
            )
            axes[1, j].set_title("After StandardScaler", fontsize=11)
            axes[1, j].set_ylabel("Count" if j == 0 else "")

        fig.suptitle("Feature Scaling: Raw vs. StandardScaler-Normalised")
        plt.tight_layout()
        _save(fig, out_dir / "06_scaling_comparison.png")
    print("  [OK] Scaling comparison")


# ──────────────────────────────────────────────────────────────────────────
# Master function
# ──────────────────────────────────────────────────────────────────────────

def run_all_preprocessing_plots(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    y_aug: np.ndarray,
    X_train: pd.DataFrame,
    X_train_scaled: np.ndarray,
    feature_cols: list[str],
    numeric_cols: list[str],
) -> None:
    print("\n" + "=" * 60)
    print("PREPROCESSING VISUALISATIONS")
    print("=" * 60)
    plot_missing_values(df_raw)
    plot_outlier_detection(df_raw, numeric_cols)
    plot_before_after_capping(df_raw, df_processed, numeric_cols)
    plot_class_distribution_smote(y_train, y_aug)
    plot_split_distribution(y_train, y_val, y_test)
    plot_scaling_comparison(X_train, X_train_scaled, feature_cols)
    print(f"\n[Preprocessing plots saved] -> {FIG_PREPROCESSING}")
