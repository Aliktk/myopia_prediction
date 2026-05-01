"""
run_pipeline.py — End-to-End Myopia Progression Prediction Pipeline
=====================================================================
Title  : Development and Validation of an AI-Based Model for Myopia Progression
Author : Syed Ahmad Hassan (2024-MPhil-OP-037)
Engineer: Ali Nawaz

Usage
-----
    python run_pipeline.py [--skip-eda] [--skip-shap] [--skip-diagrams]
                           [--smote {smote,smote_tomek,smoteenn}]

Pipeline Phases
---------------
  Phase 1   Data Loading & Inspection
  Phase 2   Feature Engineering
  Phase 3   Preprocessing: cleaning, splitting, scaling
  Phase 4   Data Augmentation (SMOTE)
  Phase 5   Preprocessing Visualisations
  Phase 6   Exploratory Data Analysis
  Phase 7   Multi-Model Training
  Phase 8   Results Summary
  Phase 9   Evaluation Visualisations
  Phase 10  SHAP Explainability
  Phase 11  Save Model Artifacts
  Phase 12  Architectural Diagrams
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    ALL_FEATURE_COLS,
    APP_DIR,
    ENGINEERED_COLS,
    FEATURE_COLS_FILE,
    MODELS_DIR,
    OUTPUTS_DIR,
    RAW_DATA_PATH,
    RAW_NUMERIC_COLS,
    SCALER_FILE,
)
from src.data.augmentation import (
    apply_smote,
    apply_smote_tomek,
    apply_smoteenn,
)
from src.data.feature_engineering import engineer_all_features
from src.data.loader import inspection_report, load_raw
from src.data.preprocessor import full_preprocessing_pipeline
from src.eda.visualizer import run_full_eda
from src.models.evaluator import (
    bootstrap_confidence_intervals,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_cv_boxplots,
    plot_feature_importance,
    plot_learning_curves,
    plot_model_comparison_bars,
    plot_model_metric_heatmap,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.models.explainer import generate_shap_report
from src.models.trainer import (
    results_to_dataframe,
    save_best_model,
    train_and_evaluate,
)
from src.visualization.preprocessing_plots import run_all_preprocessing_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Myopia Progression ML Pipeline")
    p.add_argument("--skip-eda", action="store_true", help="Skip EDA visualisations")
    p.add_argument("--skip-diagrams", action="store_true", help="Skip architectural diagrams")
    p.add_argument("--skip-shap", action="store_true", help="Skip SHAP explanations")
    p.add_argument("--smote", default="smote",
                   choices=["smote", "smote_tomek", "smoteenn"],
                   help="SMOTE strategy (default: smote)")
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main() -> None:
    args = parse_args()

    # Phase 1
    section("PHASE 1 - DATA LOADING & INSPECTION")
    df_raw = load_raw(RAW_DATA_PATH)
    report = inspection_report(df_raw)
    print(f"  Source      : {RAW_DATA_PATH.name}")
    print(f"  Shape       : {df_raw.shape}")
    print(f"  Label dist  : {report['label_counts']}")
    print(f"  Balance     : {report['label_balance_pct']}")
    print(f"  Missing     : {sum(report['missing_values'].values())}")
    print(f"  Duplicates  : {report['duplicate_rows']}")
    print(f"  Age range   : {report['age_range']}")
    print(f"  Gender      : {report['gender_counts']}")
    print(f"  Eye         : {report['eye_counts']}")

    # Phase 2
    section("PHASE 2 - FEATURE ENGINEERING")
    df = engineer_all_features(df_raw)
    new_cols = [c for c in df.columns if c not in df_raw.columns]
    print(f"  Raw features      : {len(df_raw.columns)}")
    print(f"  After engineering : {len(df.columns)}")
    print(f"  New features      : {len(new_cols)}")

    # Phase 3
    section("PHASE 3 - PREPROCESSING (clean, split, scale)")
    prep = full_preprocessing_pipeline(
        df=df,
        numeric_cols=RAW_NUMERIC_COLS,
        feature_cols=ALL_FEATURE_COLS,
    )
    X_train, X_val, X_test = prep["X_train"], prep["X_val"], prep["X_test"]
    y_train = prep["y_train"].values
    y_val = prep["y_val"].values
    y_test = prep["y_test"].values
    X_train_sc = prep["X_train_scaled"]
    X_val_sc = prep["X_val_scaled"]
    X_test_sc = prep["X_test_scaled"]
    scaler = prep["scaler"]
    feature_cols = prep["feature_cols"]
    df_processed = prep["df_processed"]

    print(f"  Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}")
    print(f"  Features used : {len(feature_cols)}")

    # Phase 4
    section("PHASE 4 - DATA AUGMENTATION (SMOTE)")
    aug_fn = {
        "smote": apply_smote,
        "smote_tomek": apply_smote_tomek,
        "smoteenn": apply_smoteenn,
    }[args.smote]
    X_aug, y_aug = aug_fn(X_train_sc, y_train)
    before = dict(zip(*np.unique(y_train, return_counts=True)))
    after = dict(zip(*np.unique(y_aug, return_counts=True)))
    print(f"  Strategy : {args.smote}")
    print(f"  Before   : {before}")
    print(f"  After    : {after}")

    # Phase 5
    section("PHASE 5 - PREPROCESSING VISUALISATIONS")
    run_all_preprocessing_plots(
        df_raw=df_raw,
        df_processed=df_processed,
        y_train=y_train, y_val=y_val, y_test=y_test, y_aug=y_aug,
        X_train=X_train,
        X_train_scaled=X_train_sc,
        feature_cols=feature_cols,
        numeric_cols=RAW_NUMERIC_COLS,
    )

    # Phase 6
    if not args.skip_eda:
        section("PHASE 6 - EXPLORATORY DATA ANALYSIS")
        run_full_eda(df, RAW_NUMERIC_COLS, ENGINEERED_COLS)
    else:
        print("\n  [SKIP] EDA")

    # Phase 7
    section("PHASE 7 - MULTI-MODEL TRAINING")
    results = train_and_evaluate(
        X_train=X_train_sc, y_train=y_train,
        X_test=X_test_sc, y_test=y_test,
        use_smote=True, smote_strategy=args.smote,
    )

    # Phase 8
    section("PHASE 8 - RESULTS SUMMARY")
    df_results = results_to_dataframe(results)
    cols_to_show = ["Model", "AUC-ROC", "F1-Score", "Sensitivity",
                    "Specificity", "Accuracy", "Train Time (s)"]
    print(df_results[cols_to_show].to_string(index=False))

    best_name = df_results.iloc[0]["Model"]
    best_auc = df_results.iloc[0]["AUC-ROC"]
    print(f"\n  Best Model : {best_name}")
    print(f"  Best AUC   : {best_auc:.4f}")

    best_prob = results[best_name]["y_prob"]
    ci = None
    if best_prob is not None:
        ci = bootstrap_confidence_intervals(y_test, best_prob, n_boot=1000)
        print(f"  AUC 95% CI : [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    # Phase 9
    section("PHASE 9 - MODEL EVALUATION VISUALISATIONS")
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)
    plot_confusion_matrices(results)
    plot_calibration_curves(results, y_test)
    plot_cv_boxplots(results)
    plot_model_metric_heatmap(results)
    plot_model_comparison_bars(results)
    plot_feature_importance(results, feature_cols)
    plot_learning_curves(results, X_aug, y_aug, top_n=3)

    # Phase 10
    if not args.skip_shap:
        section("PHASE 10 - SHAP EXPLAINABILITY")
        generate_shap_report(
            model=results[best_name]["model"],
            X_train=X_train_sc,
            feature_names=feature_cols,
            model_name=best_name,
        )
    else:
        print("\n  [SKIP] SHAP")

    # Phase 11
    section("PHASE 11 - SAVE MODEL ARTIFACTS")
    best_name_saved = save_best_model(results, key="auc_roc")
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_cols, FEATURE_COLS_FILE)

    APP_DIR.mkdir(exist_ok=True)
    joblib.dump(results[best_name_saved]["model"], APP_DIR / "best_model.joblib")
    joblib.dump(scaler, APP_DIR / "scaler.joblib")
    with open(APP_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
    df_processed.to_csv(APP_DIR / "processed_data.csv", index=False)

    print(f"  [OK] best_model.joblib   -> {MODELS_DIR}")
    print(f"  [OK] scaler.joblib       -> {MODELS_DIR}")
    print(f"  [OK] feature_cols.joblib -> {MODELS_DIR}")
    print(f"  [OK] app/ artifacts      -> {APP_DIR}")

    # Phase 12
    if not args.skip_diagrams:
        section("PHASE 12 - ARCHITECTURAL DIAGRAMS")
        try:
            from diagrams.generate_diagrams import generate_all_diagrams
            generate_all_diagrams()
        except Exception as e:
            print(f"  [WARN] Diagrams failed: {e}")
    else:
        print("\n  [SKIP] Diagrams")

    # Final summary
    section("PIPELINE COMPLETE")
    print(f"  Best Model : {best_name}")
    print(f"  AUC-ROC    : {best_auc:.4f}")
    if ci:
        print(f"  95% CI     : [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    top3 = df_results.head(3)[["Model", "AUC-ROC", "F1-Score"]].to_string(index=False)
    print(f"\n  Top-3 Models:\n{top3}")
    print(f"\n  All outputs -> {OUTPUTS_DIR}")
    print(f"  Streamlit   : streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
