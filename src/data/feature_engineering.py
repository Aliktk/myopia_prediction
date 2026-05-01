"""
src/data/feature_engineering.py — Clinical Feature Engineering
================================================================
Derives 21+ engineered features from raw corneal topography measurements.
Each engineering block is documented with its clinical rationale and
literature reference where applicable.

References
----------
- Rabinowitz, Y.S. (1998). Keratoconus. Survey of Ophthalmology 42(4).
- Rabinowitz, Y.S. (2002). Videokeratographic indices (KISA). J Refract Surg.
- Randleman, J.B. (2008). Risk Assessment for Ectasia (ERSS). J Refract Surg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────────────────────
# Categorical encoding
# ──────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode `gender` and `eye` columns deterministically."""
    out = df.copy()
    out["gender_encoded"] = LabelEncoder().fit_transform(out["gender"])
    out["eye_encoded"] = LabelEncoder().fit_transform(out["eye"])
    return out


# ──────────────────────────────────────────────────────────────────────────
# Pachymetry — corneal thickness derived features
# ──────────────────────────────────────────────────────────────────────────

def add_pachymetry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pachymetry asymmetry, ratio, and thinning displacement features."""
    out = df.copy()
    out["pachy_diff"] = out["pachy_central_um"] - out["pachy_thinnest_um"]
    out["pachy_ratio"] = out["pachy_thinnest_um"] / out["pachy_central_um"]
    out["pachy_thinnest_displacement"] = np.sqrt(
        out["pachy_thinnest_x"] ** 2 + out["pachy_thinnest_y"] ** 2
    )
    out["pachy_thin_flag"] = (out["pachy_thinnest_um"] < 500).astype(int)
    out["pachy_diff_flag"] = (out["pachy_diff"] > 30).astype(int)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Asphericity (Q-value) features
# ──────────────────────────────────────────────────────────────────────────

def add_asphericity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anterior/posterior asphericity differences, ratios, and oblate flag."""
    out = df.copy()
    out["asphericity_diff"] = (
        out["asphericity_anterior"] - out["asphericity_posterior"]
    )
    safe_denom = out["asphericity_posterior"].replace(0, np.nan)
    out["asphericity_ratio"] = (
        out["asphericity_anterior"] / safe_denom
    ).fillna(0).clip(-10, 10)
    out["asphericity_abs_sum"] = (
        out["asphericity_anterior"].abs() + out["asphericity_posterior"].abs()
    )
    out["anterior_oblate_flag"] = (out["asphericity_anterior"] > 0).astype(int)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Astigmatism — magnitude, cyclic axis encoding, axis-type flags
# ──────────────────────────────────────────────────────────────────────────

def _classify_axis(deg: float) -> str:
    """Clinical classification of astigmatism axis."""
    if 60 <= deg <= 120:
        return "ATR"  # against-the-rule
    if deg <= 30 or deg >= 150:
        return "WTR"  # with-the-rule
    return "Oblique"


def add_astigmatism_features(df: pd.DataFrame) -> pd.DataFrame:
    """Magnitude, cyclic encoding (sin/cos of 2θ), axis-type and high flags."""
    out = df.copy()
    out["astig_abs"] = out["astig_value_D"].abs()
    rad_2x = np.radians(2 * out["astig_axis_deg"])
    out["astig_axis_sin"] = np.sin(rad_2x)
    out["astig_axis_cos"] = np.cos(rad_2x)
    out["astig_axis_type"] = out["astig_axis_deg"].apply(_classify_axis)
    out["astig_wtr_flag"] = (out["astig_axis_type"] == "WTR").astype(int)
    out["astig_high_flag"] = (out["astig_abs"] > 2.5).astype(int)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Keratometry — Kmax cyclic axis and threshold flags
# ──────────────────────────────────────────────────────────────────────────

def add_keratometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kmax cyclic axis and clinical threshold flags."""
    out = df.copy()
    rad_2x = np.radians(2 * out["kmax_axis_deg"])
    out["kmax_axis_sin"] = np.sin(rad_2x)
    out["kmax_axis_cos"] = np.cos(rad_2x)
    out["kmax_high_flag"] = (out["kmax_value_D"] > 47.2).astype(int)
    out["kmax_steep_flag"] = (out["kmax_value_D"] > 46.0).astype(int)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Composite clinical indices
# ──────────────────────────────────────────────────────────────────────────

def add_composite_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Composite indices derived from keratoconus screening literature."""
    out = df.copy()

    # Corneal power adjusted by anterior asphericity
    out["corneal_power_index"] = (
        out["kmax_value_D"] * (1 + out["asphericity_anterior"])
    )

    # Combines astigmatic irregularity with thinning gradient
    out["corneal_irregularity_index"] = (
        out["astig_abs"] * out["pachy_diff"] / 100.0
    )

    # KISA-inspired index (Rabinowitz, 2002 — simplified proxy)
    excess_kmax = (out["kmax_value_D"] - 45).clip(lower=0)
    out["kisa_proxy"] = (
        excess_kmax
        * out["astig_abs"]
        * out["pachy_thinnest_displacement"]
        * 10
        / 100.0
    )

    # CLMI — cone location magnitude index proxy
    out["cone_location_magnitude_index"] = (
        out["pachy_thinnest_displacement"]
        * (1 - out["pachy_ratio"])
        * out["kmax_value_D"]
    )

    return out


# ──────────────────────────────────────────────────────────────────────────
# Interaction features
# ──────────────────────────────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise interactions of clinically known risk factors."""
    out = df.copy()
    out["kmax_astig_interaction"] = out["kmax_value_D"] * out["astig_abs"]
    out["age_kmax_interaction"] = out["age_years"] * out["kmax_value_D"]
    out["pachy_asph_interaction"] = (
        out["pachy_central_um"] * out["asphericity_anterior"].abs()
    )
    out["age_pachy_interaction"] = (
        out["age_years"] * out["pachy_central_um"]
    )
    return out


# ──────────────────────────────────────────────────────────────────────────
# Composite ordinal risk scores
# ──────────────────────────────────────────────────────────────────────────

def add_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Composite risk scores aggregating individual binary clinical flags."""
    out = df.copy()

    # Corneal Risk Score (0–4)
    out["corneal_risk_score"] = (
        (out["kmax_value_D"] > 46.0).astype(int)
        + (out["astig_abs"] > 2.5).astype(int)
        + (out["pachy_central_um"] < 510).astype(int)
        + (out["asphericity_anterior"] > 0).astype(int)
    )

    # Ectasia Risk Score — Randleman ERSS-inspired (0–7)
    out["ectasia_risk_score"] = (
        (out["kmax_value_D"] > 47.2).astype(int) * 2
        + (out["pachy_central_um"] < 500).astype(int) * 2
        + (out["pachy_diff"] > 30).astype(int)
        + (out["astig_abs"] > 3.0).astype(int)
        + (out["asphericity_anterior"] > 0.5).astype(int)
        + (out["pachy_thinnest_displacement"] > 1.0).astype(int)
        + (out["age_years"] < 25).astype(int)
    )

    return out


# ──────────────────────────────────────────────────────────────────────────
# Age groups (for stratified EDA, not modelled directly)
# ──────────────────────────────────────────────────────────────────────────

def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical age groups used in EDA stratification only."""
    out = df.copy()
    out["age_group"] = pd.cut(
        out["age_years"],
        bins=[0, 18, 25, 35, 50, 100],
        labels=["adolescent", "young_adult", "adult", "middle_age", "senior"],
    )
    return out


# ──────────────────────────────────────────────────────────────────────────
# Master pipeline
# ──────────────────────────────────────────────────────────────────────────

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in the correct order."""
    out = encode_categoricals(df)
    out = add_pachymetry_features(out)
    out = add_asphericity_features(out)
    out = add_astigmatism_features(out)
    out = add_keratometry_features(out)
    out = add_composite_indices(out)
    out = add_interaction_features(out)
    out = add_risk_scores(out)
    out = add_age_groups(out)
    return out
