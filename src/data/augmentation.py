"""
src/data/augmentation.py — Class Imbalance Correction (SMOTE family)
=====================================================================
Three SMOTE-based oversampling strategies for the minority (progressive)
class. Applied only to training data, never to validation or test sets.
"""
from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard SMOTE oversampling."""
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return sm.fit_resample(X, y)


def apply_smote_tomek(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE + Tomek link removal — cleans noisy borderline samples."""
    sm = SMOTETomek(random_state=RANDOM_STATE)
    return sm.fit_resample(X, y)


def apply_smoteenn(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE + Edited Nearest Neighbours — aggressive cleaning."""
    sm = SMOTEENN(random_state=RANDOM_STATE)
    return sm.fit_resample(X, y)


def augmentation_report(
    y_before: np.ndarray, y_after: np.ndarray
) -> dict:
    """Summarise class counts before and after oversampling."""
    before = dict(zip(*np.unique(y_before, return_counts=True)))
    after = dict(zip(*np.unique(y_after, return_counts=True)))
    return {
        "before": {int(k): int(v) for k, v in before.items()},
        "after": {int(k): int(v) for k, v in after.items()},
        "synthetic_added": int(len(y_after) - len(y_before)),
    }
