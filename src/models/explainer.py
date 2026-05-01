"""
src/models/explainer.py — SHAP Model Explainability
=====================================================
Generates SHAP summary plots, feature-importance bars, and dependence
plots for the best-performing model. Uses TreeExplainer for tree-based
models and KernelExplainer as fallback for SVM/MLP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import DPI_SCREEN, FIG_EVALUATION, PUB_RC


def build_explainer(
    model: Any,
    X_train: np.ndarray,
    feature_names: list[str] | None = None,
    sample_size: int = 200,
):
    """Choose the appropriate SHAP explainer and compute SHAP values.

    Returns
    -------
    explainer : shap.Explainer
        The instantiated SHAP explainer.
    shap_values : np.ndarray | None
        Computed SHAP values for `sample_size` background rows
        (None if computation fails). For binary classifiers the
        positive-class array is returned.
    """
    import shap
    name = type(model).__name__
    tree_models = (
        "RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
        "GradientBoostingClassifier", "DecisionTreeClassifier",
        "ExtraTreesClassifier",
    )
    if name in tree_models:
        explainer = shap.TreeExplainer(model)
    else:
        background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    sample = X_train[: min(sample_size, len(X_train))]
    try:
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    except Exception:
        shap_values = None
    return explainer, shap_values


def plot_shap_beeswarm(
    shap_values: Any, feature_names: list[str], out_dir: Path = FIG_EVALUATION,
) -> None:
    """SHAP beeswarm summary plot."""
    import shap
    plt.figure(figsize=(11, 9))
    shap.summary_plot(
        shap_values, features=None, feature_names=feature_names,
        show=False, plot_size=None, max_display=20,
    )
    fig = plt.gcf()
    fig.suptitle("SHAP Beeswarm Summary — Feature Impact on Prediction",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_beeswarm.png", dpi=DPI_SCREEN, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] SHAP beeswarm")


def plot_shap_bar(
    shap_values: Any, feature_names: list[str], out_dir: Path = FIG_EVALUATION,
) -> None:
    """SHAP mean-absolute bar chart (importance ranking)."""
    import shap
    plt.figure(figsize=(11, 9))
    shap.summary_plot(
        shap_values, plot_type="bar", feature_names=feature_names,
        show=False, max_display=20,
    )
    fig = plt.gcf()
    fig.suptitle("SHAP Global Feature Importance (mean |SHAP|)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_bar.png", dpi=DPI_SCREEN, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] SHAP bar chart")


def generate_shap_report(
    model: Any, X_train: np.ndarray, feature_names: list[str], model_name: str,
    out_dir: Path = FIG_EVALUATION,
) -> None:
    """Generate a complete SHAP explanation set for the best model."""
    try:
        import shap  # noqa: F401
    except ImportError:
        print("  [WARN] SHAP not installed — skipping explanations")
        return

    print(f"  Building SHAP explainer for {model_name}...")
    explainer, shap_values = build_explainer(model, X_train, feature_names)
    if shap_values is None:
        print("  [WARN] SHAP value computation failed; skipping plots.")
        return

    plot_shap_beeswarm(shap_values, feature_names, out_dir)
    plot_shap_bar(shap_values, feature_names, out_dir)
