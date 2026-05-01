"""
src/config.py — Central Configuration
======================================
Single source of truth for all paths, constants, feature lists, and
publication-quality matplotlib styling. Every other module imports from here.

Path resolution uses pathlib so the project is portable across Windows,
macOS, and Linux without modification.
"""
from __future__ import annotations

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────

ROOT: Path = Path(__file__).resolve().parent.parent

# Data
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RAW_DATA_PATH: Path = RAW_DIR / "combined_clinical_data_and_labels.csv"

# Outputs
OUTPUTS_DIR: Path = ROOT / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
DOCUMENT_DIR: Path = OUTPUTS_DIR / "document"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
FIG_PREPROCESSING: Path = FIGURES_DIR / "preprocessing"
FIG_EDA: Path = FIGURES_DIR / "eda"
FIG_EVALUATION: Path = FIGURES_DIR / "evaluation"
FIG_PUBLICATION: Path = FIGURES_DIR / "publication"
FIG_DIAGRAMS: Path = FIGURES_DIR / "diagrams"

# Streamlit app
APP_DIR: Path = ROOT / "app"

# Reports
RESULTS_CSV: Path = REPORTS_DIR / "model_results_summary.csv"
CV_REPORT_CSV: Path = REPORTS_DIR / "cross_validation_report.csv"
STAT_TESTS_CSV: Path = REPORTS_DIR / "statistical_tests.csv"

# Model artifacts
BEST_MODEL_FILE: Path = MODELS_DIR / "best_model.joblib"
SCALER_FILE: Path = MODELS_DIR / "scaler.joblib"
FEATURE_COLS_FILE: Path = MODELS_DIR / "feature_cols.joblib"

# Ensure all output directories exist at import time
for _d in (
    OUTPUTS_DIR, MODELS_DIR, REPORTS_DIR, DOCUMENT_DIR, FIGURES_DIR,
    FIG_PREPROCESSING, FIG_EDA, FIG_EVALUATION, FIG_PUBLICATION, FIG_DIAGRAMS,
):
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Modelling constants
# ──────────────────────────────────────────────────────────────────────────

RANDOM_STATE: int = 42
TARGET_COL: str = "label"

TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.10
CV_FOLDS: int = 5

DPI_SCREEN: int = 150
DPI_PUBLICATION: int = 300


# ──────────────────────────────────────────────────────────────────────────
# Feature columns
# ──────────────────────────────────────────────────────────────────────────

RAW_NUMERIC_COLS: list[str] = [
    "age_years",
    "astig_value_D", "astig_axis_deg",
    "kmax_value_D", "kmax_axis_deg",
    "pachy_central_um", "pachy_thinnest_um",
    "pachy_thinnest_x", "pachy_thinnest_y",
    "asphericity_anterior", "asphericity_posterior",
]

ENCODED_CATEGORICAL_COLS: list[str] = ["gender_encoded", "eye_encoded"]
# Backwards-compatible alias used by the research notebook
RAW_CATEGORICAL_COLS: list[str] = ["gender", "eye"]

ENGINEERED_COLS: list[str] = [
    # Pachymetry
    "pachy_diff", "pachy_ratio", "pachy_thinnest_displacement",
    "pachy_thin_flag", "pachy_diff_flag",
    # Asphericity
    "asphericity_diff", "asphericity_ratio", "asphericity_abs_sum",
    "anterior_oblate_flag",
    # Astigmatism
    "astig_abs", "astig_axis_sin", "astig_axis_cos",
    "astig_wtr_flag", "astig_high_flag",
    # Keratometry
    "kmax_axis_sin", "kmax_axis_cos",
    "kmax_high_flag", "kmax_steep_flag",
    # Composite indices
    "corneal_power_index", "corneal_irregularity_index",
    "kisa_proxy", "cone_location_magnitude_index",
    # Interactions
    "kmax_astig_interaction", "age_kmax_interaction",
    "pachy_asph_interaction", "age_pachy_interaction",
    # Risk scores
    "corneal_risk_score", "ectasia_risk_score",
]

ALL_FEATURE_COLS: list[str] = (
    RAW_NUMERIC_COLS + ENCODED_CATEGORICAL_COLS + ENGINEERED_COLS
)


# ──────────────────────────────────────────────────────────────────────────
# Visual styling — publication quality, larger fonts for clarity
# ──────────────────────────────────────────────────────────────────────────

# Two-class colour palette (colourblind-friendly)
COLOR_NEG: str = "#1565C0"   # Non-progressive — strong blue
COLOR_POS: str = "#E65100"   # Progressive — strong orange
COLORS: list[str] = [COLOR_NEG, COLOR_POS]
PALETTE: dict = {0: COLOR_NEG, 1: COLOR_POS}

# Multi-model comparison palette (12 distinct colours)
MODEL_PALETTE: list[str] = [
    "#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#C62828", "#00838F",
    "#EF6C00", "#283593", "#558B2F", "#AD1457", "#4527A0", "#37474F",
]

# Publication rcParams — bigger fonts than journal default for clarity
PUB_RC: dict = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.0,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 19,
    "figure.titleweight": "bold",
    "figure.dpi": DPI_SCREEN,
    "savefig.dpi": DPI_SCREEN,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.2,
    "patch.linewidth": 1.0,
}
