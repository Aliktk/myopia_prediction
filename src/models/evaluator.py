"""
src/models/evaluator.py — Comprehensive Model Evaluation Visualisations
=========================================================================
All evaluation plots use the publication-quality PUB_RC styling defined
in src/config.py — larger fonts, serif typeface, and clear gridlines for
maximum legibility in journal figures and thesis appendices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)

from src.config import (
    COLOR_NEG,
    COLOR_POS,
    COLORS,
    DPI_PUBLICATION,
    DPI_SCREEN,
    FIG_EVALUATION,
    FIG_PUBLICATION,
    MODEL_PALETTE,
    PUB_RC,
)


def _save(fig: plt.Figure, path: Path, dpi: int = DPI_SCREEN) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# 1. ROC Curves
# ──────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: dict[str, dict[str, Any]],
    y_test: np.ndarray,
    out_dir: Path = FIG_EVALUATION,
) -> None:
    """ROC curves overlay for all models."""
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(11, 9))
        for i, (name, r) in enumerate(results.items()):
            if r["y_prob"] is None:
                continue
            fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
            ax.plot(
                fpr, tpr,
                label=f"{name} (AUC={r['auc_roc']:.3f})",
                color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                linewidth=2.5,
            )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1.4, label="Chance (AUC=0.5)")
        ax.set_xlabel("False Positive Rate (1 − Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title("ROC Curves — All Models")
        ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        plt.tight_layout()
        _save(fig, out_dir / "roc_curves.png")
        _save(fig, FIG_PUBLICATION / "fig_roc_curves.pdf", dpi=DPI_PUBLICATION)
    print("  [OK] ROC curves")


# ──────────────────────────────────────────────────────────────────────────
# 2. Precision–Recall Curves
# ──────────────────────────────────────────────────────────────────────────

def plot_precision_recall_curves(
    results: dict[str, dict[str, Any]],
    y_test: np.ndarray,
    out_dir: Path = FIG_EVALUATION,
) -> None:
    """Precision–Recall curves overlay for all models."""
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(11, 9))
        for i, (name, r) in enumerate(results.items()):
            if r["y_prob"] is None:
                continue
            precs, recs, _ = precision_recall_curve(y_test, r["y_prob"])
            ax.plot(
                recs, precs,
                label=f"{name} (AP={r['avg_precision']:.3f})",
                color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                linewidth=2.5,
            )
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision (PPV)")
        ax.set_title("Precision–Recall Curves — All Models")
        ax.legend(loc="lower left", fontsize=11, framealpha=0.95)
        plt.tight_layout()
        _save(fig, out_dir / "pr_curves.png")
        _save(fig, FIG_PUBLICATION / "fig_pr_curves.pdf", dpi=DPI_PUBLICATION)
    print("  [OK] Precision-Recall curves")


# ──────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrices Grid
# ──────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    results: dict[str, dict[str, Any]],
    out_dir: Path = FIG_EVALUATION,
) -> None:
    """Normalised confusion matrices for all models."""
    n_models = len(results)
    ncols = 4
    nrows = int(np.ceil(n_models / ncols))

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows))
        axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

        for i, (name, r) in enumerate(results.items()):
            cm = r["cm"]
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(
                cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=["Non-Prog", "Prog"],
                yticklabels=["Non-Prog", "Prog"],
                ax=axes[i], cbar=False, annot_kws={"size": 14, "weight": "bold"},
                vmin=0.0, vmax=1.0, linewidths=0.5,
            )
            axes[i].set_title(
                f"{name}\nAcc={r['accuracy']:.3f} | F1={r['f1']:.3f}",
                fontsize=13, fontweight="bold",
            )
            axes[i].set_ylabel("Actual", fontsize=12)
            axes[i].set_xlabel("Predicted", fontsize=12)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Confusion Matrices — Normalised by True Class", y=1.00)
        plt.tight_layout()
        _save(fig, out_dir / "confusion_matrices.png")
        _save(fig, FIG_PUBLICATION / "fig_confusion_matrices.pdf", dpi=DPI_PUBLICATION)
    print("  [OK] Confusion matrices")


# ──────────────────────────────────────────────────────────────────────────
# 4. Calibration Curves
# ──────────────────────────────────────────────────────────────────────────

def plot_calibration_curves(
    results: dict[str, dict[str, Any]],
    y_test: np.ndarray,
    out_dir: Path = FIG_EVALUATION,
) -> None:
    """Reliability diagrams with Brier scores for the top models."""
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.plot([0, 1], [0, 1], "k:", lw=1.5, label="Perfectly calibrated")

        ranked = sorted(
            results.items(), key=lambda kv: kv[1]["auc_roc"], reverse=True
        )[:6]
        for i, (name, r) in enumerate(ranked):
            if r["y_prob"] is None:
                continue
            try:
                prob_true, prob_pred = calibration_curve(
                    y_test, r["y_prob"], n_bins=10, strategy="quantile"
                )
                bs = brier_score_loss(y_test, r["y_prob"])
                ax.plot(
                    prob_pred, prob_true, "o-",
                    label=f"{name} (Brier={bs:.3f})",
                    color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                    linewidth=2.4, markersize=8,
                )
            except Exception:
                continue

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives (Observed)")
        ax.set_title("Calibration Curves (Reliability Diagram) — Top 6 Models")
        ax.legend(loc="upper left", fontsize=11)
        plt.tight_layout()
        _save(fig, out_dir / "calibration_curves.png")
    print("  [OK] Calibration curves")


# ──────────────────────────────────────────────────────────────────────────
# 5. Cross-Validation Score Distributions
# ──────────────────────────────────────────────────────────────────────────

def plot_cv_boxplots(
    results: dict[str, dict[str, Any]],
    out_dir: Path = FIG_EVALUATION,
) -> None:
    """Box plots of 5-fold CV AUC distributions per model."""
    rows = []
    for name, r in results.items():
        if "cv_roc_auc" in r:
            for fold_idx, score in enumerate(r["cv_roc_auc"]["scores"]):
                rows.append({"Model": name, "AUC": score, "Fold": fold_idx + 1})
    if not rows:
        return

    df_cv = pd.DataFrame(rows)
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.boxplot(
            data=df_cv, x="Model", y="AUC", ax=ax,
            palette=MODEL_PALETTE[: df_cv["Model"].nunique()],
            width=0.65, linewidth=1.2,
        )
        sns.stripplot(
            data=df_cv, x="Model", y="AUC", ax=ax,
            color="black", alpha=0.7, jitter=True, size=5,
        )
        ax.set_title("5-Fold Cross-Validation AUC-ROC Distributions")
        ax.set_ylabel("AUC-ROC")
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        plt.tight_layout()
        _save(fig, out_dir / "cv_boxplots.png")
    print("  [OK] CV box plots")


# ──────────────────────────────────────────────────────────────────────────
# 6. Model × Metric Heatmap
# ──────────────────────────────────────────────────────────────────────────

def plot_model_metric_heatmap(
    results: dict[str, dict[str, Any]],
    out_dir: Path = FIG_EVALUATION,
) -> None:
    """All-metrics × all-models colour-coded heatmap."""
    metrics = ["accuracy", "precision", "sensitivity", "specificity",
               "f1", "npv", "auc_roc", "avg_precision"]
    labels = ["Accuracy", "Precision", "Sensitivity", "Specificity",
              "F1-Score", "NPV", "AUC-ROC", "Avg Prec"]

    rows = []
    for name, r in results.items():
        rows.append([r.get(m, 0.0) for m in metrics])
    arr = np.array(rows)
    df_hm = pd.DataFrame(arr, index=list(results.keys()), columns=labels)
    df_hm = df_hm.sort_values("AUC-ROC", ascending=False)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(12, max(7, 0.55 * len(df_hm))))
        sns.heatmap(
            df_hm, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.85, vmax=1.0, ax=ax, linewidths=0.6,
            annot_kws={"size": 12, "weight": "bold"},
            cbar_kws={"shrink": 0.7, "label": "Score"},
        )
        ax.set_title("Model × Metric Performance Heatmap")
        ax.set_ylabel("Model")
        ax.set_xlabel("Metric")
        plt.tight_layout()
        _save(fig, out_dir / "model_metric_heatmap.png")
    print("  [OK] Model x Metric heatmap")


# ──────────────────────────────────────────────────────────────────────────
# 7. Model Comparison Bar Chart
# ──────────────────────────────────────────────────────────────────────────

def plot_model_comparison_bars(
    results: dict[str, dict[str, Any]],
    out_dir: Path = FIG_EVALUATION,
) -> None:
    """Grouped bar chart comparing key metrics across models."""
    metrics = ["accuracy", "f1", "sensitivity", "specificity", "auc_roc"]
    labels = ["Accuracy", "F1-Score", "Sensitivity", "Specificity", "AUC-ROC"]
    df = pd.DataFrame({m: [results[n].get(m, 0.0) for n in results]
                       for m in metrics}, index=list(results.keys()))
    df.columns = labels
    df = df.sort_values("AUC-ROC", ascending=False)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(16, 8))
        df.plot(
            kind="bar", ax=ax, width=0.8,
            color=["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#C62828"],
            edgecolor="white", linewidth=0.6,
        )
        ax.set_title("Model Performance Comparison — Key Metrics")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", ncol=5, fontsize=11, framealpha=0.95)
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        plt.tight_layout()
        _save(fig, out_dir / "model_comparison_bars.png")
        _save(fig, FIG_PUBLICATION / "fig_model_comparison.pdf", dpi=DPI_PUBLICATION)
    print("  [OK] Model comparison bars")


# ──────────────────────────────────────────────────────────────────────────
# 8. Feature Importance (top tree-based models)
# ──────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    results: dict[str, dict[str, Any]],
    feature_cols: list[str],
    out_dir: Path = FIG_EVALUATION,
    top_n: int = 15,
) -> None:
    """Feature importance from tree-based models."""
    tree_models = ["Random Forest", "XGBoost", "LightGBM", "Extra Trees", "Gradient Boosting"]
    tree_models = [m for m in tree_models if m in results]
    if not tree_models:
        return

    with plt.rc_context(PUB_RC):
        n = len(tree_models)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 9))
        if n == 1:
            axes = [axes]

        for i, name in enumerate(tree_models):
            model = results[name]["model"]
            if not hasattr(model, "feature_importances_"):
                axes[i].set_visible(False)
                continue
            importances = model.feature_importances_
            order = np.argsort(importances)[-top_n:]
            features = [feature_cols[j] for j in order]
            values = importances[order]

            axes[i].barh(
                features, values,
                color=COLORS[i % 2], edgecolor="black", linewidth=0.6,
            )
            axes[i].set_title(f"{name}\nTop {top_n} Features", fontsize=14)
            axes[i].set_xlabel("Importance")
            axes[i].tick_params(axis="y", labelsize=11)

        fig.suptitle("Feature Importance — Tree-Based Models", y=1.01)
        plt.tight_layout()
        _save(fig, out_dir / "feature_importance.png")
        _save(fig, FIG_PUBLICATION / "fig_feature_importance.pdf", dpi=DPI_PUBLICATION)
    print("  [OK] Feature importance")


# ──────────────────────────────────────────────────────────────────────────
# 9. Learning Curves (top 3 models)
# ──────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    results: dict[str, dict[str, Any]],
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    top_n: int = 3,
    out_dir: Path = FIG_EVALUATION,
    *,
    X_aug: np.ndarray | None = None,
    y_aug: np.ndarray | None = None,
) -> None:
    """Train vs validation AUC as a function of training set size."""
    from src.models.trainer import compute_learning_curve
    if X is None and X_aug is not None:
        X = X_aug
    if y is None and y_aug is not None:
        y = y_aug
    if X is None or y is None:
        raise ValueError("plot_learning_curves requires X/y (or X_aug/y_aug).")

    ranked = sorted(
        results.items(), key=lambda kv: kv[1]["auc_roc"], reverse=True
    )[:top_n]

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, top_n, figsize=(7 * top_n, 6.5))
        if top_n == 1:
            axes = [axes]

        for i, (name, r) in enumerate(ranked):
            try:
                sizes, train_scores, val_scores = compute_learning_curve(
                    r["model"], X, y, cv=5, scoring="roc_auc", n_points=8
                )
                t_mean, t_std = train_scores.mean(axis=1), train_scores.std(axis=1)
                v_mean, v_std = val_scores.mean(axis=1), val_scores.std(axis=1)

                axes[i].plot(sizes, t_mean, "o-", color=COLOR_NEG,
                             label="Training", linewidth=2.5, markersize=8)
                axes[i].fill_between(sizes, t_mean - t_std, t_mean + t_std,
                                      alpha=0.18, color=COLOR_NEG)
                axes[i].plot(sizes, v_mean, "s-", color=COLOR_POS,
                             label="Validation", linewidth=2.5, markersize=8)
                axes[i].fill_between(sizes, v_mean - v_std, v_mean + v_std,
                                      alpha=0.18, color=COLOR_POS)
                axes[i].set_title(f"{name}\nAUC={r['auc_roc']:.3f}", fontsize=14)
                axes[i].set_xlabel("Training Set Size")
                axes[i].set_ylabel("AUC-ROC")
                axes[i].legend(loc="lower right")
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Learning curve failed:\n{e}",
                             ha="center", va="center",
                             transform=axes[i].transAxes)

        fig.suptitle(f"Learning Curves — Top {top_n} Models", y=1.02)
        plt.tight_layout()
        _save(fig, out_dir / "learning_curves.png")
    print("  [OK] Learning curves")


# ──────────────────────────────────────────────────────────────────────────
# 10. Bootstrap Confidence Intervals
# ──────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
) -> dict[str, float]:
    """McNemar's test comparing two classifiers' errors on the same test set."""
    from statsmodels.stats.contingency_tables import mcnemar
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    n00 = int(((~a_correct) & (~b_correct)).sum())
    n01 = int(((~a_correct) & b_correct).sum())
    n10 = int((a_correct & (~b_correct)).sum())
    n11 = int((a_correct & b_correct).sum())
    table = [[n11, n10], [n01, n00]]
    res = mcnemar(table, exact=False, correction=True)
    # n10 = A correct, B wrong  ->  the canonical "b" cell of McNemar's table
    # n01 = A wrong, B correct  ->  the canonical "c" cell of McNemar's table
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "b": n10, "c": n01,
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    alpha: float | None = None,
    seed: int | None = None,
    *,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for AUC-ROC.

    Accepts both old-style (alpha, seed) and notebook-style (ci, random_state)
    keyword arguments. `ci` is the confidence level (default 0.95).
    """
    from sklearn.metrics import roc_auc_score
    if alpha is None:
        alpha = 1.0 - ci
    rng = np.random.default_rng(seed if seed is not None else random_state)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    aucs = np.array(aucs)
    return {
        "mean": float(aucs.mean()),
        "std": float(aucs.std()),
        "lower": float(np.percentile(aucs, 100 * (alpha / 2))),
        "upper": float(np.percentile(aucs, 100 * (1 - alpha / 2))),
        "n_valid_resamples": len(aucs),
    }
