# AI-Based Myopia Progression Prediction

**Development and Validation of an Artificial Intelligence Model for Predicting Myopia Progression Using Clinical Corneal Topography Data**

> **Investigator** — Syed Ahmad Hassan, MPhil Ophthalmology (2024-MPhil-OP-037)
> **AI Engineering** — Ali Nawaz
> **Version** — 2.0.0 (combined dataset, 1,642 records)

---

## Headline Result

| Metric | Value |
|---|---|
| **Best Model** | LightGBM |
| **AUC-ROC (test set)** | **0.9996** |
| **95 % Bootstrap CI** | [0.9987, 1.0000] |
| **Sensitivity** | 0.9912 |
| **Specificity** | 0.9861 |
| **F1-Score** | 0.9825 |
| **Test Set Size** | 329 patient eyes (20 % stratified holdout) |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Clinical Background](#2-clinical-background)
3. [Project Structure](#3-project-structure)
4. [Dataset](#4-dataset)
5. [Feature Engineering](#5-feature-engineering)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Class Imbalance Correction (SMOTE)](#7-class-imbalance-correction-smote)
8. [Model Architecture](#8-model-architecture)
9. [Training Pipeline](#9-training-pipeline)
10. [Results](#10-results)
11. [Evaluation Metrics](#11-evaluation-metrics-explained)
12. [Generated Outputs](#12-generated-outputs)
13. [How to Run](#13-how-to-run)
14. [Streamlit Application](#14-streamlit-application)
15. [Requirements](#15-requirements)
16. [References](#16-references)

---

## 1. Project Overview

This repository implements a complete, reproducible machine-learning pipeline for the binary classification of myopia (short-sightedness) into **non-progressive** (Label 0) or **progressive** (Label 1) categories using clinical corneal topography measurements.

The pipeline covers:

- **Clinical feature engineering** grounded in published keratoconus and ectasia screening literature (Rabinowitz KISA, Randleman ERSS, Belin–Ambrosio, CLMI proxy)
- **Rigorous preprocessing** with strict train/validation/test data discipline
- **SMOTE** synthetic oversampling for class imbalance
- **Multi-model comparative study** across 11 classifiers + a stacking ensemble
- **Comprehensive evaluation** including ROC-AUC, F1, sensitivity, specificity, calibration curves, bootstrap 95% CI, and learning curves
- **SHAP explainability** for clinical interpretability
- **Streamlit clinical decision-support application**
- **Publication-quality figures** with enlarged fonts suitable for journal submission and thesis appendices
- **Microsoft Word research document** auto-generated from latest results

---

## 2. Clinical Background

Myopia is the world's most prevalent ocular condition, affecting an estimated 2.6 billion people in 2020 and projected to affect 4.9 billion by 2050 (Holden et al., 2016). While stable myopia carries limited risk, **progressive myopia** is associated with elevated lifetime risk of:

- Retinal detachment
- Myopic maculopathy
- Glaucoma
- Cataract

Early identification of progressive cases enables timely intervention through orthokeratology, low-dose atropine, or multifocal lenses. Traditional clinical decision-making relies on threshold-based rules applied to individual measurements, which fails to capture multidimensional interactions between biomechanical, refractive, and topographic variables. Machine learning models can learn these interactions explicitly and produce probabilistic risk scores that support — rather than replace — ophthalmologist judgement.

---

## 3. Project Structure

```text
myopia_prediction/
├── data/
│   └── raw/
│       ├── combined_clinical_data_and_labels.csv     # primary dataset (1,642 rows)
│       ├── clinical_data_and_labels_v1.csv           # v1 collection (1,454 rows)
│       └── clinical_data_and_labels_v2.csv           # v2 collection (188 rows)
│
├── src/                                              # Source library
│   ├── config.py                                     # Central configuration and styling
│   ├── data/
│   │   ├── loader.py
│   │   ├── feature_engineering.py
│   │   ├── preprocessor.py
│   │   └── augmentation.py
│   ├── eda/
│   │   └── visualizer.py
│   ├── models/
│   │   ├── definitions.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── explainer.py
│   ├── visualization/
│   │   └── preprocessing_plots.py
│   └── utils/
│       └── generate_research_document.py
│
├── app/
│   ├── streamlit_app.py                              # Theme-safe clinical UI
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── feature_columns.json
│   └── processed_data.csv
│
├── notebooks/
│   └── myopia_prediction_complete.ipynb
│
├── diagrams/
│   └── generate_diagrams.py
│
├── outputs/
│   ├── models/                                       # Saved artifacts
│   ├── reports/                                      # CSV tables
│   ├── document/                                     # Word research document
│   └── figures/
│       ├── preprocessing/
│       ├── eda/
│       ├── evaluation/
│       ├── publication/
│       └── diagrams/
│
├── .streamlit/
│   └── config.toml
│
├── run_pipeline.py
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## 4. Dataset

### Combined Dataset Summary

| Property | Value |
|---|---|
| Total samples | **1,642** |
| Raw columns | 15 |
| Label 0 (Non-Progressive) | 1,077 (65.6 %) |
| Label 1 (Progressive) | 565 (34.4 %) |
| Missing values | None |
| Duplicate records | None |
| Patient age range | 13 – 65 years |
| Gender | f: 1,073 (65.4 %), m: 569 (34.6 %) |
| Eye laterality | OD: 822, OS: 820 |

The dataset combines two recruitment phases (`v1` and `v2`) consolidated into `combined_clinical_data_and_labels.csv`. The v2 expansion added 188 rows (≈ 13 % more data) primarily improving representation of the non-progressive class.

### Raw Variables

| Variable | Type | Clinical Description |
|---|---|---|
| `age_years` | Numeric | Age at examination |
| `gender` | Categorical | f / m |
| `eye` | Categorical | OD = right, OS = left |
| `astig_value_D` | Numeric | Refractive astigmatism (Diopters) |
| `astig_axis_deg` | Numeric | Astigmatism axis (0–180°) |
| `kmax_value_D` | Numeric | Maximum keratometry (Diopters) |
| `kmax_axis_deg` | Numeric | Axis of maximum curvature (0–180°) |
| `pachy_central_um` | Numeric | Central corneal thickness (μm) |
| `pachy_thinnest_um` | Numeric | Thinnest corneal thickness (μm) |
| `pachy_thinnest_x` | Numeric | X-coordinate of thinnest point (mm) |
| `pachy_thinnest_y` | Numeric | Y-coordinate of thinnest point (mm) |
| `asphericity_anterior` | Numeric | Anterior surface Q-value |
| `asphericity_posterior` | Numeric | Posterior surface Q-value |
| `label` | Binary | 0 = Non-progressive, 1 = Progressive |

---

## 5. Feature Engineering

The pipeline expands the 13 raw inputs into a **41-dimensional feature space** through 32 engineered features. All engineering is grounded in published keratoconus and ectasia screening literature.

| Group | Features | Rationale |
|---|---|---|
| **Encoded categoricals** | `gender_encoded`, `eye_encoded` | Label encoding |
| **Pachymetry** | `pachy_diff`, `pachy_ratio`, `pachy_thinnest_displacement`, `pachy_thin_flag`, `pachy_diff_flag` | Belin-Ambrosio thinning gradient |
| **Asphericity** | `asphericity_diff`, `asphericity_ratio`, `asphericity_abs_sum`, `anterior_oblate_flag` | Q-value irregularity |
| **Astigmatism** | `astig_abs`, `astig_axis_sin`, `astig_axis_cos`, `astig_wtr_flag`, `astig_high_flag` | Cyclic axis encoding (sin/cos of 2θ) resolves the 0°/180° discontinuity |
| **Keratometry** | `kmax_axis_sin`, `kmax_axis_cos`, `kmax_high_flag`, `kmax_steep_flag` | Cyclic encoding + clinical thresholds (Rabinowitz, 1998) |
| **Composite indices** | `corneal_power_index`, `corneal_irregularity_index`, `kisa_proxy`, `cone_location_magnitude_index` | Simplified KISA (Rabinowitz, 2002), CLMI proxy |
| **Interactions** | `kmax_astig_interaction`, `age_kmax_interaction`, `pachy_asph_interaction`, `age_pachy_interaction` | Non-linear combinations |
| **Risk scores** | `corneal_risk_score` (0-4), `ectasia_risk_score` (0-7) | Aggregated binary flags (Randleman ERSS) |

### Why Cyclic Axis Encoding?

Astigmatism axes repeat every 180°. Direct use as a linear feature would treat 1° and 179° as maximally different when they are clinically nearly identical. Encoding the axis as `sin(2θ)` and `cos(2θ)` maps the 0–180° range onto a continuous unit circle with the correct period.

---

## 6. Preprocessing Pipeline

| Step | Action | Rationale |
|---|---|---|
| 1 | Duplicate removal | Prevents leakage between splits |
| 2 | IQR × 3 outlier capping (Winsorising) | Conservative factor (vs default 1.5) preserves clinical variation |
| 3 | Median imputation | Robust to skewed distributions |
| 4 | Stratified 70/10/20 split | Preserves class ratio in every partition |
| 5 | StandardScaler (fit on train only) | Prevents data leakage |

### Split Sizes

| Split | Proportion | Sample Count |
|---|---|---|
| Training | 70 % | 1,148 |
| Validation | 10 % | 165 |
| Test | 20 % | 329 |

---

## 7. Class Imbalance Correction (SMOTE)

After the stratified split, the training set holds 753 non-progressive and 395 progressive samples (65.6 % / 34.4 %). To prevent models from biasing toward the majority class — which would mean missing progressive cases — we apply **SMOTE** (Chawla et al., 2002) to the **training set only**. Validation and test sets retain the natural distribution.

| Class | Before SMOTE | After SMOTE |
|---|---|---|
| Non-Progressive | 753 | 753 |
| Progressive | 395 | 753 |
| **Total** | **1,148** | **1,506** |

Three SMOTE strategies are supported via `--smote {smote, smote_tomek, smoteenn}`.

---

## 8. Model Architecture

### Eleven Base Classifiers

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | C = 1.0, lbfgs, balanced class weight |
| Decision Tree | max_depth = 8, min_samples_split = 10 |
| Random Forest | n_estimators = 300, balanced_subsample |
| Extra Trees | n_estimators = 300, balanced_subsample |
| Gradient Boosting | n_estimators = 300, lr = 0.05, max_depth = 4 |
| XGBoost | n_estimators = 300, lr = 0.05, max_depth = 5 |
| LightGBM | n_estimators = 300, lr = 0.05, num_leaves = 31 |
| AdaBoost | n_estimators = 200, lr = 0.5 |
| SVM (RBF) | C = 10, gamma = scale, probability = True |
| KNN | n_neighbors = 7, distance-weighted |
| MLP Neural Net | (128, 64, 32) hidden layers, ReLU, Adam |

### Stacking Ensemble

```text
Level 0:  Random Forest  +  XGBoost  +  LightGBM  +  SVM (RBF)
                 |
        Out-of-fold predictions via 5-fold CV
                 |
Level 1:  Logistic Regression (C=0.1) — meta-learner
```

---

## 9. Training Pipeline

```text
Raw CSV (1,642 rows)
        │
        ▼
Feature Engineering (47 columns total)
        │
        ▼
Preprocessing  ➜  Test set (329) held aside permanently
        │
        ▼
Training (1,148) ➜ SMOTE ➜ Augmented (1,506)
        │
        ▼
5-Fold Stratified CV on all 11 models
        │
        ▼
Full-train fit on the augmented training set
        │
        ▼
Evaluate on the 329-patient test set
        │
        ▼
Bootstrap 95 % CI for best-model AUC
```

All randomness is controlled by `RANDOM_STATE = 42` and is deterministic across runs.

---

## 10. Results

### Test Set Performance (n = 329)

| Rank | Model | AUC-ROC | F1 | Sensitivity | Specificity | Accuracy | Precision | NPV |
|---|---|---|---|---|---|---|---|---|
| 1 | **LightGBM** | **0.9996** | 0.9825 | 0.9912 | 0.9861 | 0.9878 | 0.9739 | 0.9953 |
| 2 | Gradient Boosting | 0.9996 | 0.9912 | 0.9912 | 0.9954 | 0.9939 | 0.9912 | 0.9954 |
| 3 | Random Forest | 0.9995 | 0.9868 | 0.9912 | 0.9907 | 0.9909 | 0.9825 | 0.9953 |
| 4 | AdaBoost | 0.9993 | 0.9782 | 0.9912 | 0.9815 | 0.9848 | 0.9655 | 0.9953 |
| 5 | Stacking Ensemble | 0.9993 | 0.9825 | 0.9912 | 0.9861 | 0.9878 | 0.9739 | 0.9953 |
| 6 | XGBoost | 0.9992 | 0.9868 | 0.9912 | 0.9907 | 0.9909 | 0.9825 | 0.9953 |
| 7 | Extra Trees | 0.9989 | 0.9820 | 0.9646 | 1.0000 | 0.9878 | 1.0000 | 0.9818 |
| 8 | MLP Neural Net | 0.9988 | 0.9735 | 0.9735 | 0.9861 | 0.9818 | 0.9735 | 0.9861 |
| 9 | SVM (RBF) | 0.9970 | 0.9432 | 0.9558 | 0.9630 | 0.9605 | 0.9310 | 0.9765 |
| 10 | Logistic Regression | 0.9961 | 0.9391 | 0.9558 | 0.9583 | 0.9574 | 0.9231 | 0.9764 |
| 11 | KNN | 0.9918 | 0.9561 | 0.9646 | 0.9722 | 0.9696 | 0.9478 | 0.9813 |
| 12 | Decision Tree | 0.9356 | 0.9163 | 0.9204 | 0.9537 | 0.9422 | 0.9123 | 0.9581 |

### Best-Model Detailed Summary — LightGBM

| Metric | Value |
|---|---|
| AUC-ROC | 0.9996 |
| 95 % Bootstrap CI (AUC) | [0.9987, 1.0000] |
| Sensitivity (Recall) | 0.9912 — 99.1 % of progressive cases correctly identified |
| Specificity | 0.9861 — 98.6 % of non-progressive cases correctly identified |
| F1-Score | 0.9825 |
| Precision | 0.9739 |
| Negative Predictive Value | 0.9953 |
| Overall Accuracy | 0.9878 |

### Key Observations

1. **Gradient boosting dominates** — LightGBM, Gradient Boosting, and XGBoost occupy the top 6 ranks, confirming the established superiority of boosting on structured tabular clinical data.
2. **Adding 188 rows lifted AUC from 0.9960 → 0.9996.** The expanded training data closed the residual gap between the best model and ceiling performance.
3. **Linear logistic regression scored AUC = 0.9961** — extremely close to top non-linear models, indicating that the engineered features encode most of the discriminative signal in a near-linearly separable way. This is encouraging for clinical interpretability.
4. **Stacking did not surpass the best single model**, suggesting the base learners capture overlapping decision boundaries — a sign of a clean dataset ceiling.

---

## 11. Evaluation Metrics Explained

| Metric | Formula | Clinical Meaning |
|---|---|---|
| **AUC-ROC** | Area under ROC curve | Threshold-independent discrimination — primary metric |
| **Sensitivity** | TP / (TP + FN) | Fraction of progressive cases correctly identified — most clinically critical |
| **Specificity** | TN / (TN + FP) | Fraction of non-progressive cases correctly identified |
| **Precision (PPV)** | TP / (TP + FP) | Of those predicted progressive, fraction correct |
| **NPV** | TN / (TN + FN) | Of those predicted non-progressive, fraction correct |
| **F1-Score** | 2 PR / (P + R) | Harmonic mean — robust to imbalance |
| **Brier Score** | Mean (predicted − actual)² | Calibration quality — lower is better |

**Bootstrap CI** is computed by resampling the test set 1,000 times with replacement and recomputing AUC at each iteration. The 2.5 ‰ and 97.5 ‰ percentiles form the 95 % confidence interval.

---

## 12. Generated Outputs

### Figures (`outputs/figures/`)

| Folder | Files | Purpose |
|---|---|---|
| `preprocessing/` | 6 PNG | Missing values, outliers, capping, SMOTE, splits, scaling |
| `eda/` | 11 PNG + 2 HTML | Distributions, correlation, box, violin, pair-plot, significance, clinical groupings, engineered features, descriptive table, 3D, sunburst |
| `evaluation/` | 9 PNG | ROC, PR, confusion, calibration, CV box, heatmap, comparison, importance, learning curves |
| `publication/` | 5 PDF | High-DPI vector versions for journal submission |
| `diagrams/` | 6 SVG | Architectural diagrams for the paper |

### Reports (`outputs/reports/`)

| File | Description |
|---|---|
| `model_results_summary.csv` | Full metrics table — all 12 models |
| `cross_validation_report.csv` | 5-fold CV scores per model and metric |
| `statistical_tests.csv` | Mann-Whitney U + Cohen's d per feature |

### Model Artifacts (`outputs/models/` and `app/`)

| File | Description |
|---|---|
| `best_model.joblib` | Serialised LightGBM (best by AUC) |
| `scaler.joblib` | Fitted StandardScaler |
| `feature_cols.joblib` / `feature_columns.json` | Ordered feature list |

### Word Document

`outputs/document/Myopia_Progression_Research_Document.docx` — auto-generated, ready for direct insertion into a thesis or synopsis.

---

## 13. How to Run

### Prerequisites

```bash
python --version          # 3.10 or higher
```

### Install Dependencies

```bash
# Option A — uv (fast, recommended)
uv venv myopia
uv pip install --python myopia/Scripts/python.exe -r requirements.txt

# Option B — venv + pip
python -m venv myopia
myopia\Scripts\activate          # Windows
source myopia/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Recommended first run — fast (~3 minutes)
python run_pipeline.py --skip-shap --skip-diagrams

# Full pipeline with all visualisations
python run_pipeline.py

# Choose alternative SMOTE strategies
python run_pipeline.py --smote smote_tomek
python run_pipeline.py --smote smoteenn
```

### Generate the MS Word Research Document

```bash
python -m src.utils.generate_research_document
# -> outputs/document/Myopia_Progression_Research_Document.docx
```

### Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 14. Streamlit Application

A four-page clinical decision-support interface with **theme-safe contrast** that adapts to light or dark mode automatically (no white-on-white or black-on-black bugs).

| Page | Content |
|---|---|
| Patient Prediction | 13 input fields → engineered features → scaled → LightGBM probability → gauge + risk band + clinical flag chips |
| Model Performance | Full metrics table with colour-graded heatmap + 9 evaluation figure tabs |
| Dataset Explorer | Interactive filters with histogram + box marginal |
| About | Project description, methodology summary, references |

The app reuses `src.data.feature_engineering.engineer_all_features` so inference is mathematically identical to training — same features, same scaling, same model.

---

## 15. Requirements

```text
# Core
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost>=2.0
lightgbm>=4.0
shap>=0.44

# Visualisation
matplotlib>=3.7
seaborn>=0.13
plotly>=5.15

# Application
streamlit>=1.28
joblib>=1.3

# Document generation
python-docx>=1.0

# Statistical testing
statsmodels>=0.14

# Notebooks (optional)
jupyter
ipykernel
```

---

## 16. References

1. **Rabinowitz, Y.S.** (1998). Keratoconus. *Survey of Ophthalmology*, 42(4), 297–319.
2. **Rabinowitz, Y.S.** (2002). Videokeratographic indices to aid in screening for keratoconus. *Journal of Refractive Surgery*, 11(5), 371–379.
3. **Randleman, J.B., Woodward, M., Lynn, M.J., & Stulting, R.D.** (2008). Risk Assessment for Ectasia after Corneal Refractive Surgery. *Journal of Refractive Surgery*, 24(9), 895–902.
4. **Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P.** (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
5. **Holden, B.A., et al.** (2016). Global Prevalence of Myopia and High Myopia and Temporal Trends from 2000 through 2050. *Ophthalmology*, 123(5), 1036–1042.
6. **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *KDD*, 785–794.
7. **Ke, G., et al.** (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*, 30.
8. **Lundberg, S.M., & Lee, S.I.** (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS*, 30.
9. **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
10. **Wolpert, D.H.** (1992). Stacked Generalisation. *Neural Networks*, 5(2), 241–259.
11. **Belin, M.W., & Khachikian, S.S.** (2009). An introduction to understanding elevation-based topography. *Clinical & Experimental Ophthalmology*, 37(1), 14–29.

---

## Appendix — Reproducibility Checklist

- [ ] Python version 3.10 or higher
- [ ] All packages from `requirements.txt` installed
- [ ] Combined dataset present at `data/raw/combined_clinical_data_and_labels.csv`
- [ ] `RANDOM_STATE = 42` in `src/config.py` (default)
- [ ] Default split ratios 70/10/20 (default)
- [ ] Default SMOTE strategy `smote` (default)
- [ ] Run: `python run_pipeline.py --skip-shap --skip-diagrams`
- [ ] Verify `outputs/reports/model_results_summary.csv` matches table in §10

---

*This research is conducted as part of a MPhil degree in Ophthalmology. All patient records are fully anonymised. The predictive model outputs are intended to support — not replace — the clinical judgement of qualified ophthalmologists. No clinical decisions should be made based solely on model output.*
