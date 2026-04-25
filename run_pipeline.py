"""
run_pipeline.py — End-to-End Myopia Progression Prediction Pipeline
=====================================================================
Title  : Development and Validation of AI-Based Model for Myopia Progression
Author : Syed Ahmad Hassan (2024-MPhil-OP-037)

Usage
-----
    python run_pipeline.py [--skip-eda] [--skip-diagrams] [--smote smote_tomek]

Pipeline Phases
---------------
  Phase 1  Data Loading & Inspection
  Phase 2  Feature Engineering (25 clinical features)
  Phase 3  Preprocessing: cleaning, splitting, scaling
  Phase 4  Preprocessing Visualisations
  Phase 5  EDA: 15 visualisations + interactive plots
  Phase 6  Data Augmentation (SMOTE)
  Phase 7  Multi-Model Training (11 classifiers + stacking)
  Phase 8  Model Evaluation (ROC, PR, CM, calibration, learning curves)
  Phase 9  SHAP Explainability
  Phase 10 Architectural Diagrams
  Phase 11 Summary Report
"""
from __future__ import annotations
import argparse
import json
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    RAW_DATA_PATH, OUTPUTS_DIR, MODELS_DIR, REPORTS_DIR,
    ALL_FEATURE_COLS, RAW_NUMERIC_COLS, ENGINEERED_COLS,
    TARGET_COL, RANDOM_STATE, RESULTS_CSV,
    BEST_MODEL_FILE, SCALER_FILE, FEATURE_COLS_FILE,
)
from src.data.loader import load_raw, inspection_report
from src.data.feature_engineering import engineer_all_features
from src.data.preprocessor import full_preprocessing_pipeline
from src.data.augmentation import apply_smote, apply_smote_tomek, apply_smoteenn
from src.eda.visualizer import run_full_eda
from src.models.trainer import train_and_evaluate, save_best_model, results_to_dataframe
from src.models.evaluator import (
    plot_roc_curves, plot_precision_recall_curves,
    plot_confusion_matrices, plot_calibration_curves,
    plot_cv_boxplots, plot_model_metric_heatmap,
    plot_model_comparison_bars, plot_feature_importance,
    plot_learning_curves, bootstrap_confidence_intervals,
)
from src.models.explainer import generate_shap_report
from src.visualization.preprocessing_plots import run_all_preprocessing_plots

import numpy as np
import pandas as pd
import joblib


def parse_args():
    p = argparse.ArgumentParser(description="Myopia Progression ML Pipeline")
    p.add_argument("--skip-eda",      action="store_true", help="Skip EDA visualisations")
    p.add_argument("--skip-diagrams", action="store_true", help="Skip architectural diagrams")
    p.add_argument("--skip-shap",     action="store_true", help="Skip SHAP explanations")
    p.add_argument("--smote", default="smote",
                   choices=["smote", "smote_tomek", "smoteenn"],
                   help="SMOTE strategy (default: smote)")
    return p.parse_args()


def section(title: str):
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")


def main():
    args = parse_args()

    # ── Phase 1: Data Loading ─────────────────────────────────────────────────
    section("PHASE 1 . DATA LOADING & INSPECTION")
    df_raw = load_raw(RAW_DATA_PATH)
    report = inspection_report(df_raw)

    print(f"  Shape       : {df_raw.shape}")
    print(f"  Label dist  : {report['label_counts']}")
    print(f"  Balance     : {report['label_balance_pct']}")
    print(f"  Missing     : {sum(v for v in report['missing_values'].values())} total cells")
    print(f"  Duplicates  : {report['duplicate_rows']}")
    print(f"  Age range   : {report['age_range']}")
    print(f"  Gender      : {report['gender_counts']}")
    print(f"  Eye         : {report['eye_counts']}")

    # ── Phase 2: Feature Engineering ─────────────────────────────────────────
    section("PHASE 2 . FEATURE ENGINEERING")
    df = engineer_all_features(df_raw)
    new_cols = [c for c in df.columns if c not in df_raw.columns]
    print(f"  Raw features    : {len(df_raw.columns)}")
    print(f"  After engineering: {len(df.columns)} columns")
    print(f"  New features    : {len(new_cols)}")
    print(f"  New: {new_cols}")

    # ── Phase 3: Preprocessing ────────────────────────────────────────────────
    section("PHASE 3 . PREPROCESSING: CLEANING, SPLITTING, SCALING")
    prep = full_preprocessing_pipeline(
        df=df,
        numeric_cols=RAW_NUMERIC_COLS,
        feature_cols=ALL_FEATURE_COLS,
    )
    X_train     = prep["X_train"]
    X_val       = prep["X_val"]
    X_test      = prep["X_test"]
    y_train     = prep["y_train"].values
    y_val       = prep["y_val"].values
    y_test      = prep["y_test"].values
    X_train_sc  = prep["X_train_scaled"]
    X_val_sc    = prep["X_val_scaled"]
    X_test_sc   = prep["X_test_scaled"]
    scaler      = prep["scaler"]
    feature_cols = prep["feature_cols"]
    df_processed = prep["df_processed"]

    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"  Features used: {len(feature_cols)}")

    # ── Phase 4: SMOTE ────────────────────────────────────────────────────────
    section("PHASE 4 . DATA AUGMENTATION (SMOTE)")
    aug_fn = {"smote": apply_smote, "smote_tomek": apply_smote_tomek,
               "smoteenn": apply_smoteenn}[args.smote]
    X_aug, y_aug = aug_fn(X_train_sc, y_train)
    before = dict(zip(*np.unique(y_train, return_counts=True)))
    after  = dict(zip(*np.unique(y_aug,  return_counts=True)))
    print(f"  Strategy : {args.smote}")
    print(f"  Before   : {before}")
    print(f"  After    : {after}")

    # ── Phase 5: Preprocessing Plots ─────────────────────────────────────────
    section("PHASE 5 . PREPROCESSING VISUALISATIONS")
    run_all_preprocessing_plots(
        df_raw=df_raw,
        df_processed=df_processed,
        y_train=y_train, y_val=y_val, y_test=y_test,
        y_aug=y_aug,
        X_train=X_train,
        X_train_scaled=X_train_sc,
        feature_cols=feature_cols,
        numeric_cols=RAW_NUMERIC_COLS,
    )

    # ── Phase 6: EDA ──────────────────────────────────────────────────────────
    if not args.skip_eda:
        section("PHASE 6 . EXPLORATORY DATA ANALYSIS")
        run_full_eda(df, RAW_NUMERIC_COLS, ENGINEERED_COLS)
    else:
        print("\n  [SKIP] EDA (--skip-eda flag set)")

    # ── Phase 7: Model Training ───────────────────────────────────────────────
    section("PHASE 7 . MULTI-MODEL TRAINING")
    results = train_and_evaluate(
        X_train=X_train_sc, y_train=y_train,
        X_test=X_test_sc,   y_test=y_test,
        use_smote=True, smote_strategy=args.smote,
    )

    # ── Phase 8: Results Summary ──────────────────────────────────────────────
    section("PHASE 8 . RESULTS SUMMARY")
    df_results = results_to_dataframe(results)
    print(df_results[["Model", "AUC-ROC", "F1-Score", "Sensitivity", "Specificity",
                       "Accuracy", "Train Time (s)"]].to_string(index=False))

    best_name = df_results.iloc[0]["Model"]
    best_auc  = df_results.iloc[0]["AUC-ROC"]
    print(f"\n  Best Model  : {best_name}")
    print(f"  Best AUC    : {best_auc:.4f}")

    # Bootstrap CI for best model
    best_prob = results[best_name]["y_prob"]
    ci = None
    if best_prob is not None:
        ci = bootstrap_confidence_intervals(y_test, best_prob, n_boot=1000)
        print(f"  AUC 95% CI  : [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    # ── Phase 9: Evaluation Plots ─────────────────────────────────────────────
    section("PHASE 9 . MODEL EVALUATION VISUALISATIONS")
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)
    plot_confusion_matrices(results)
    plot_calibration_curves(results, y_test)
    plot_cv_boxplots(results)
    plot_model_metric_heatmap(results)
    plot_model_comparison_bars(results)
    plot_feature_importance(results, feature_cols)
    plot_learning_curves(results, X_aug, y_aug, top_n=3)

    # ── Phase 10: SHAP Explanations ───────────────────────────────────────────
    if not args.skip_shap:
        section("PHASE 10 . SHAP EXPLAINABILITY")
        best_model = results[best_name]["model"]
        generate_shap_report(
            model=best_model,
            X_train=X_train_sc,
            feature_names=feature_cols,
            model_name=best_name,
        )
    else:
        print("\n  [SKIP] SHAP (--skip-shap flag set)")

    # ── Phase 11: Save Artifacts ──────────────────────────────────────────────
    section("PHASE 11 . SAVING MODEL ARTIFACTS")
    best_name_saved = save_best_model(results, key="auc_roc")
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_cols, FEATURE_COLS_FILE)

    # Streamlit app artifacts (joblib format — no pickle)
    app_dir = ROOT / "app"
    app_dir.mkdir(exist_ok=True)
    joblib.dump(results[best_name_saved]["model"], app_dir / "best_model.joblib")
    joblib.dump(scaler, app_dir / "scaler.joblib")
    with open(app_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
    df_processed.to_csv(app_dir / "processed_data.csv", index=False)

    print(f"  [OK] best_model.joblib  -> {MODELS_DIR}")
    print(f"  [OK] scaler.joblib      -> {MODELS_DIR}")
    print(f"  [OK] feature_cols.joblib-> {MODELS_DIR}")
    print(f"  [OK] app/ artifacts     -> {app_dir}")

    # ── Phase 12: Architectural Diagrams ──────────────────────────────────────
    if not args.skip_diagrams:
        section("PHASE 12 . ARCHITECTURAL DIAGRAMS")
        try:
            from diagrams.generate_diagrams import generate_all_diagrams
            generate_all_diagrams()
        except Exception as e:
            print(f"  [WARN] Diagrams failed: {e}")
    else:
        print("\n  [SKIP] Diagrams (--skip-diagrams flag set)")

    # ── Final Summary ─────────────────────────────────────────────────────────
    section("PIPELINE COMPLETE")
    print(f"  Best Model  : {best_name}")
    print(f"  AUC-ROC     : {best_auc:.4f}")
    if ci:
        print(f"  95% CI      : [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    top3 = df_results.head(3)[["Model", "AUC-ROC", "F1-Score"]].to_string(index=False)
    print(f"\n  Top-3 Models:\n{top3}")
    print(f"\n  All outputs -> {OUTPUTS_DIR}")
    print(f"  Streamlit   : streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
