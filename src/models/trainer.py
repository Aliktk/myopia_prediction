"""
src/models/trainer.py — Multi-Model Training and Cross-Validation
===================================================================
Trains all base models from definitions.py, computes 5-fold stratified
cross-validation metrics, fits on the full SMOTE-augmented training set,
and evaluates on the held-out test set.

Returns a unified results dictionary consumed by evaluator.py and
explainer.py downstream.
"""
from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE, RESULTS_CSV
from src.data.augmentation import (
    apply_smote,
    apply_smote_tomek,
    apply_smoteenn,
)
from src.models.definitions import get_base_models, get_stacking_model


def _safe_proba(model: Any, X: np.ndarray) -> np.ndarray | None:
    """Get positive-class probability or normalised decision score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return None


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_smote: bool = True,
    smote_strategy: str = "smote",
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Train all 11 base models + stacking ensemble. Return results dict."""
    if use_smote:
        aug_fn = {
            "smote": apply_smote,
            "smote_tomek": apply_smote_tomek,
            "smoteenn": apply_smoteenn,
        }[smote_strategy]
        X_aug, y_aug = aug_fn(X_train, y_train)
        if verbose:
            before = dict(zip(*np.unique(y_train, return_counts=True)))
            after = dict(zip(*np.unique(y_aug, return_counts=True)))
            print(f"  [SMOTE] Before: {before}  ->  After: {after}")
    else:
        X_aug, y_aug = X_train, y_train

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results: dict[str, dict[str, Any]] = {}

    for name, model in get_base_models().items():
        if verbose:
            print(f"\n  [{name}]", end=" ", flush=True)
        t0 = time.time()

        cv_res = cross_validate(
            model, X_aug, y_aug, cv=cv, scoring=scoring, n_jobs=-1
        )
        cv_summary = {
            f"cv_{k}": {
                "mean": float(v.mean()),
                "std": float(v.std()),
                "scores": v.tolist(),
            }
            for k, v in {
                "accuracy": cv_res["test_accuracy"],
                "precision": cv_res["test_precision"],
                "recall": cv_res["test_recall"],
                "f1": cv_res["test_f1"],
                "roc_auc": cv_res["test_roc_auc"],
            }.items()
        }

        model.fit(X_aug, y_aug)
        y_pred = model.predict(X_test)
        y_prob = _safe_proba(model, X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        elapsed = time.time() - t0

        results[name] = {
            "model": model,
            "train_time_s": round(elapsed, 2),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "sensitivity": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "specificity": round(specificity, 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "npv": round(npv, 4),
            "auc_roc": round(
                roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0, 4
            ),
            "avg_precision": round(
                average_precision_score(y_test, y_prob) if y_prob is not None else 0.0, 4
            ),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cm": cm,
            **cv_summary,
        }

        if verbose:
            r = results[name]
            print(
                f"Acc={r['accuracy']:.3f}  Sens={r['sensitivity']:.3f}  "
                f"Spec={r['specificity']:.3f}  F1={r['f1']:.3f}  "
                f"AUC={r['auc_roc']:.3f}  ({elapsed:.1f}s)"
            )

    if verbose:
        print("\n  [Stacking Ensemble]", end=" ", flush=True)
    t0 = time.time()
    stack = get_stacking_model()
    stack.fit(X_aug, y_aug)
    y_pred_s = stack.predict(X_test)
    y_prob_s = _safe_proba(stack, X_test)
    cm_s = confusion_matrix(y_test, y_pred_s)
    tn, fp, fn, tp = cm_s.ravel()
    spec_s = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv_s = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    elapsed = time.time() - t0

    results["Stacking Ensemble"] = {
        "model": stack,
        "train_time_s": round(elapsed, 2),
        "accuracy": round(accuracy_score(y_test, y_pred_s), 4),
        "precision": round(precision_score(y_test, y_pred_s, zero_division=0), 4),
        "sensitivity": round(recall_score(y_test, y_pred_s, zero_division=0), 4),
        "specificity": round(spec_s, 4),
        "f1": round(f1_score(y_test, y_pred_s, zero_division=0), 4),
        "npv": round(npv_s, 4),
        "auc_roc": round(
            roc_auc_score(y_test, y_prob_s) if y_prob_s is not None else 0.0, 4
        ),
        "avg_precision": round(
            average_precision_score(y_test, y_prob_s) if y_prob_s is not None else 0.0, 4
        ),
        "y_pred": y_pred_s, "y_prob": y_prob_s, "cm": cm_s,
        "cv_accuracy": {"mean": 0.0, "std": 0.0, "scores": []},
    }
    if verbose:
        r = results["Stacking Ensemble"]
        print(
            f"Acc={r['accuracy']:.3f}  Sens={r['sensitivity']:.3f}  "
            f"Spec={r['specificity']:.3f}  F1={r['f1']:.3f}  "
            f"AUC={r['auc_roc']:.3f}  ({elapsed:.1f}s)"
        )

    return results


def save_best_model(results: dict[str, dict[str, Any]], key: str = "auc_roc") -> str:
    """Persist the best model by a given metric and return its name."""
    best_name = max(results, key=lambda n: results[n].get(key, 0))
    joblib.dump(results[best_name]["model"], MODELS_DIR / "best_model.joblib")
    return best_name


def results_to_dataframe(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert results dict to a sorted summary DataFrame."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Sensitivity": r["sensitivity"],
            "Specificity": r["specificity"],
            "F1-Score": r["f1"],
            "NPV": r["npv"],
            "AUC-ROC": r["auc_roc"],
            "Avg Precision": r["avg_precision"],
            "CV Acc Mean": r.get("cv_accuracy", {}).get("mean", 0.0),
            "CV Acc Std": r.get("cv_accuracy", {}).get("std", 0.0),
            "Train Time (s)": r["train_time_s"],
        })
    df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    df.to_csv(RESULTS_CSV, index=False)
    return df


def compute_learning_curve(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_points: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sklearn learning curve."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, n_points),
    )
    return train_sizes, train_scores, val_scores
