# AI-Based Myopia Progression Prediction

**Development and Validation of an Artificial Intelligence Model for Predicting Myopia Progression Using Clinical Corneal Topography Data**

> **Research Project** — **Syed Ahmad Hasan** MPhil Ophthalmology (2024-MPhil-OP-037)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Clinical Background](#2-clinical-background)
3. [Project Structure](#3-project-structure)
4. [Dataset Description](#4-dataset-description)
5. [Feature Engineering](#5-feature-engineering)
6. [Data Preprocessing Pipeline](#6-data-preprocessing-pipeline)
7. [Data Augmentation](#7-data-augmentation-smote)
8. [Model Architecture](#8-model-architecture)
9. [Training Pipeline](#9-training-pipeline)
10. [Model Evaluation Results](#10-model-evaluation-results)
11. [Evaluation Metrics](#11-evaluation-metrics-explained)
12. [Multi-Agent Architecture](#12-multi-agent-architecture)
13. [Generated Outputs](#13-generated-outputs)
14. [How to Run](#14-how-to-run)
15. [Streamlit Clinical App](#15-streamlit-clinical-app)
16. [Requirements](#16-requirements)
17. [References](#17-references)

---

## 1. Project Overview

This project presents a complete, end-to-end machine learning pipeline for the binary classification of myopia (short-sightedness) as either **non-progressive** (Label 0) or **progressive** (Label 1) using clinical corneal topography measurements.

The pipeline covers:

- Clinical **feature engineering** grounded in published keratoconus and myopia screening literature
- Rigorous **data preprocessing** including outlier capping, imputation, and stratified splitting
- **Synthetic oversampling** (SMOTE) to address class imbalance
- **Multi-model comparative study** across 11 classifiers plus a stacking ensemble
- Comprehensive **evaluation**: ROC-AUC, F1, sensitivity, specificity, calibration curves, bootstrap confidence intervals, and learning curves
- **SHAP explainability** for clinical interpretability
- A **Streamlit web application** for real-time clinical inference
- Publication-quality visualisations suitable for journal submission and thesis appendices

The best-performing model (LightGBM) achieved an **AUC-ROC of 0.9960** with a **95% bootstrap confidence interval of [0.9900, 0.9998]**, demonstrating that machine learning can reliably predict myopia progression from routine clinical measurements.

---

## 2. Clinical Background

### What is Myopia?

Myopia (nearsightedness) is a refractive error in which light focuses in front of the retina rather than on it, causing distant objects to appear blurred. It is one of the most prevalent ocular conditions worldwide, affecting an estimated 2.6 billion people in 2020 and projected to affect 4.9 billion by 2050.

### Why Predict Progression?

Not all myopic eyes are at equal risk. **Progressive myopia** — where the eye continues to elongate and the refractive error worsens year over year — is associated with:

- Retinal detachment
- Myopic maculopathy
- Glaucoma
- Cataracts

Early identification of patients likely to progress allows clinicians to initiate **myopia management interventions** (orthokeratology, atropine therapy, multifocal lenses) that slow or halt progression, preventing vision-threatening complications.

### Why Machine Learning?

Traditional clinical decision-making relies on discrete thresholds applied to individual measurements (e.g., Kmax > 47.2 D). Machine learning can:

1. Learn non-linear relationships between multiple measurements simultaneously
2. Quantify interaction effects that are invisible to threshold-based rules
3. Generate probabilistic risk scores rather than binary yes/no decisions
4. Improve generalisability across heterogeneous patient populations

---

## 3. Project Structure

```
myopia_prediction/
│
├── data/
│   └── raw/
│       └── clinical_data_and_labels.csv     # Original dataset
│
├── src/                                      # Core source library
│   ├── config.py                             # Central configuration (all paths, constants)
│   ├── data/
│   │   ├── loader.py                         # Data loading and inspection
│   │   ├── feature_engineering.py            # 8 engineering modules -> 34 features
│   │   ├── preprocessor.py                   # Cleaning, splitting, scaling
│   │   └── augmentation.py                   # SMOTE, SMOTETomek, SMOTEENN
│   ├── models/
│   │   ├── definitions.py                    # All 11 model definitions + ensemble
│   │   ├── trainer.py                        # Training loop + cross-validation
│   │   ├── evaluator.py                      # All evaluation plots
│   │   └── explainer.py                      # SHAP explainability
│   ├── eda/
│   │   └── visualizer.py                     # 15+ EDA plots
│   └── visualization/
│       └── preprocessing_plots.py            # 6 preprocessing visualisations
│
├── agents/                                   # Multi-agent modular pipeline
│   ├── run_all_agents.py                     # Sequential agent orchestrator
│   ├── agent_01_data_analyzer.py             # Data inspection agent
│   ├── agent_02_preprocessor.py              # Preprocessing agent
│   ├── agent_03_eda.py                       # EDA agent
│   ├── agent_04_model_trainer.py             # Training agent
│   ├── agent_05_evaluator.py                 # Evaluation agent
│   └── agent_06_inference.py                 # Inference agent
│
├── app/
│   ├── streamlit_app.py                      # Clinical inference web app
│   ├── best_model.joblib                     # Serialised best model
│   ├── scaler.joblib                         # Fitted StandardScaler
│   ├── feature_columns.json                  # Feature column list
│   └── processed_data.csv                    # Processed data for explorer
│
├── notebooks/
│   └── myopia_prediction_complete.ipynb      # End-to-end annotated notebook
│
├── diagrams/
│   └── generate_diagrams.py                  # Architectural diagram generator
│
├── outputs/
│   ├── models/                               # Saved model artifacts
│   ├── reports/                              # model_results_summary.csv
│   └── figures/
│       ├── preprocessing/                    # 6 preprocessing plots
│       ├── eda/                              # 15 EDA plots + 3 interactive HTML
│       ├── evaluation/                       # 9 evaluation plots
│       ├── publication/                      # High-DPI figures for paper
│       └── diagrams/                         # Architectural diagrams (PNG + PDF)
│
├── run_pipeline.py                           # 12-phase end-to-end orchestrator
└── requirements.txt
```

---

## 4. Dataset Description

### Source and Size

| Property                   | Value                                  |
| -------------------------- | -------------------------------------- |
| Total samples              | 1,454                                  |
| Total raw columns          | 15                                     |
| Target variable            | `label` (binary)                     |
| Label 0 — Non-Progressive | 889 (61.1%)                            |
| Label 1 — Progressive     | 565 (38.9%)                            |
| Missing values             | None                                   |
| Duplicate records          | None                                   |
| Patient age range          | 13 – 65 years                         |
| Gender distribution        | Female: 947 (65.1%), Male: 507 (34.9%) |
| Eye laterality             | OD (Right): 730, OS (Left): 724        |

### Class Imbalance

The dataset exhibits a **61.1% / 38.9%** split in favour of non-progressive cases. While not extreme, this imbalance is clinically significant — models trained without correction tend to become biased toward predicting the majority class, which would translate to missed progressive cases in clinical practice. SMOTE is applied to correct this before training (see Section 7).

### Raw Feature Columns

The 15 raw columns comprise:

| Column                    | Type        | Clinical Description                              |
| ------------------------- | ----------- | ------------------------------------------------- |
| `age_years`             | Numeric     | Patient age at time of examination                |
| `gender`                | Categorical | Patient sex (f / m)                               |
| `eye`                   | Categorical | Laterality (OD = right, OS = left)                |
| `astig_value_D`         | Numeric     | Refractive astigmatism magnitude (Diopters)       |
| `astig_axis_deg`        | Numeric     | Astigmatism axis angle (0–180 degrees)           |
| `kmax_value_D`          | Numeric     | Maximum keratometry — peak corneal curvature (D) |
| `kmax_axis_deg`         | Numeric     | Axis of maximum curvature (0–180 degrees)        |
| `pachy_central_um`      | Numeric     | Central corneal thickness (microns)               |
| `pachy_thinnest_um`     | Numeric     | Thinnest point corneal thickness (microns)        |
| `pachy_thinnest_x`      | Numeric     | X-coordinate of thinnest point (mm from apex)     |
| `pachy_thinnest_y`      | Numeric     | Y-coordinate of thinnest point (mm from apex)     |
| `asphericity_anterior`  | Numeric     | Anterior surface Q-value (shape factor)           |
| `asphericity_posterior` | Numeric     | Posterior surface Q-value                         |
| `label`                 | Binary      | 0 = Non-progressive, 1 = Progressive              |

---

## 5. Feature Engineering

A total of **34 features** are used for model training: the 11 raw numeric columns, 2 encoded categorical columns, and **21 engineered features** derived from domain knowledge in corneal topography and keratoconus screening literature.

### 5.1 Categorical Encoding

| Engineered Feature | Description                          |
| ------------------ | ------------------------------------ |
| `gender_encoded` | Label encoding: Female = 0, Male = 1 |
| `eye_encoded`    | Label encoding: OD = 0, OS = 1       |

### 5.2 Pachymetry Features

Corneal thickness is a primary biomarker for ectasia and myopia progression. Asymmetry between the central and thinnest zones is clinically significant.

| Engineered Feature              | Formula                 | Clinical Significance                                   |
| ------------------------------- | ----------------------- | ------------------------------------------------------- |
| `pachy_diff`                  | central - thinnest (um) | > 30 um signals abnormal thinning gradient              |
| `pachy_ratio`                 | thinnest / central      | < 0.94 is a recognised keratoconus risk threshold       |
| `pachy_thinnest_displacement` | sqrt(x^2 + y^2) in mm   | > 1 mm indicates decentred thinning (irregular ectasia) |

### 5.3 Asphericity Features

The corneal Q-value (asphericity) describes how the cornea departs from a perfect sphere. Normal corneas are prolate (Q < 0). Positive or abnormal Q-values are associated with irregular corneal shapes.

| Engineered Feature       | Formula                            | Clinical Significance                                          |
| ------------------------ | ---------------------------------- | -------------------------------------------------------------- |
| `asphericity_diff`     | anterior Q - posterior Q           | Anterior–posterior imbalance signals ectatic distortion       |
| `asphericity_ratio`    | anterior Q / posterior Q           | Relative surface irregularity between the two corneal surfaces |
| `asphericity_abs_sum`  | abs(anterior Q) + abs(posterior Q) | Total magnitude of shape deviation on both surfaces            |
| `anterior_oblate_flag` | 1 if anterior Q > 0                | Binary flag for abnormal oblate (positive) asphericity         |

### 5.4 Astigmatism Features

Raw astigmatism axis (0–180 degrees) has a circular discontinuity: 0 degrees and 180 degrees represent the same axis. Direct use in linear models introduces a spurious mathematical boundary. Cyclic (double-angle) encoding resolves this.

| Engineered Feature  | Formula               | Clinical Significance                                       |
| ------------------- | --------------------- | ----------------------------------------------------------- |
| `astig_abs`       | abs(astig_value_D)    | Magnitude of astigmatism regardless of sign convention      |
| `astig_axis_sin`  | sin(2 x axis_radians) | Cyclic y-component: resolves the 0/180 degree discontinuity |
| `astig_axis_cos`  | cos(2 x axis_radians) | Cyclic x-component                                          |
| `astig_axis_type` | WTR / ATR / Oblique   | Clinical classification of astigmatism axis type            |

> **Why double-angle encoding?** Astigmatism axes repeat every 180 degrees, so multiplying by 2 maps the 0–180 degree range onto a full 360 degree circle, making sin and cos continuous and periodic with the correct period. This prevents the model from treating 0 degrees and 179 degrees as maximally different when they are actually almost identical.

### 5.5 Keratometry Features

| Engineered Feature  | Formula                    | Clinical Significance                                     |
| ------------------- | -------------------------- | --------------------------------------------------------- |
| `kmax_axis_sin`   | sin(2 x kmax_axis_radians) | Cyclic encoding of the steepest corneal meridian          |
| `kmax_axis_cos`   | cos(2 x kmax_axis_radians) | Cyclic encoding of the steepest corneal meridian          |
| `kmax_high_flag`  | 1 if Kmax > 47.2 D         | Standard keratoconus suspect threshold (Rabinowitz, 1998) |
| `kmax_steep_flag` | 1 if Kmax > 46.0 D         | Early-warning borderline threshold                        |

### 5.6 Composite Clinical Indices

These composite features are inspired by published keratoconus screening indices:

| Index                             | Formula                                           | Literature Basis                                                         |
| --------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------ |
| `corneal_power_index`           | Kmax x (1 + Q_anterior)                           | Combines peak curvature with shape deviation into a single power measure |
| `corneal_irregularity_index`    | abs(astig) x pachy_diff / 100                     | Combines astigmatic irregularity with the thinning gradient              |
| `kisa_proxy`                    | (Kmax-45)+ x abs(astig) x displacement x 10 / 100 | Simplified KISA index (Rabinowitz, 2002, Cornea 21:S60)                  |
| `cone_location_magnitude_index` | displacement x (1 - pachy_ratio) x Kmax           | Approximates the CLMI used by commercial corneal topographers            |

### 5.7 Interaction Features

Non-linear interactions between known clinical risk factors:

| Feature                    | Formula                             | Rationale                                                                        |
| -------------------------- | ----------------------------------- | -------------------------------------------------------------------------------- |
| `kmax_astig_interaction` | Kmax x abs(astig)                   | High curvature combined with high astigmatism represents a strong ectasia signal |
| `age_kmax_interaction`   | Age x Kmax                          | Young patients with steep corneas carry higher progression risk                  |
| `pachy_asph_interaction` | Central_thickness x abs(Q_anterior) | Thin cornea with abnormal shape elevates surgical and ectasia risk               |
| `age_pachy_interaction`  | Age x Central_thickness             | Older patients with thin corneas may reflect chronic remodelling                 |

### 5.8 Composite Risk Scores

Two ordinal risk scores aggregate binary clinical thresholds into a single severity measure for patient stratification:

**Corneal Risk Score (0–4):** Count of 4 primary flags:

- Kmax > 46.0 D
- Astigmatism magnitude > 2.5 D
- Central thickness < 510 um
- Anterior asphericity > 0

**Ectasia Risk Score (0–7):** Weighted sum inspired by the Randleman ERSS (2008):

- Kmax > 47.2 D: +2 points
- Central thickness < 500 um: +2 points
- pachy_diff > 30 um: +1 point
- Astigmatism > 3.0 D: +1 point
- Anterior Q > 0.5: +1 point
- Thinnest point displacement > 1 mm: +1 point
- Age < 25 years: +1 point

---

## 6. Data Preprocessing Pipeline

All preprocessing steps follow strict train/validation/test data discipline. No information from the validation or test sets is used to fit any transformation, which is a fundamental requirement for an unbiased evaluation.

### Step 1 — Duplicate Removal

All exact duplicate rows are identified and removed before the dataset is split. This prevents identical records from appearing in both the training set and the test set, which would artificially inflate performance metrics.

### Step 2 — Outlier Capping (Winsorising)

Extreme outliers are capped using the **IQR method** with a conservative factor of 3.0:

```
lower_bound = Q1 - 3.0 x IQR
upper_bound = Q3 + 3.0 x IQR
Values outside these bounds are clipped to the bound value.
```

A factor of 3.0 (rather than the standard 1.5) is used deliberately to preserve genuine clinical variation in rare but real measurements while removing only clear measurement artefacts and data entry errors. Outlier capping is applied **only to the 11 raw numeric columns**, not to binary flags or engineered indicators.

### Step 3 — Missing Value Imputation

**Median imputation** is applied to any missing numeric values. The median is preferred over the mean in clinical datasets because measurements such as pachymetry and keratometry often follow skewed distributions where the mean is pulled by extreme values.

> Note: This dataset contains zero missing values. The imputation step is retained in the pipeline for robustness when applied to future patient data.

### Step 4 — Stratified 3-Way Split

The dataset is partitioned into three non-overlapping sets using **stratified random sampling**, which preserves the original class ratio (61%/39%) in every partition:

| Split          | Proportion | Sample Count |
| -------------- | ---------- | ------------ |
| Training set   | 70%        | 1,017        |
| Validation set | 10%        | 146          |
| Test set       | 20%        | 291          |

- **Training set**: Used exclusively to fit all model parameters and cross-validation folds
- **Validation set**: Used to monitor training progress and inform early stopping decisions
- **Test set**: Held out entirely until final evaluation; represents genuinely unseen clinical data

The split uses `random_state=42` to ensure full reproducibility. The splitting procedure is:

```
Step 1: Separate test set (20%) from the full dataset using stratified sampling
Step 2: From the remaining 80%, separate validation (10% of original = 12.5% of remainder)
Step 3: The rest becomes the training set (70% of original)
```

### Step 5 — Feature Scaling (StandardScaler)

All 34 numerical features are standardised to zero mean and unit variance:

```
z = (x - mean_train) / std_train
```

**Critical rule:** The scaler is **fit exclusively on the training set** (`scaler.fit_transform(X_train)`) and then applied to the validation and test sets using training statistics only (`scaler.transform(X_val)`, `scaler.transform(X_test)`). Fitting on the full dataset before splitting is a form of data leakage that artificially inflates evaluation metrics.

The fitted scaler is serialised to `outputs/models/scaler.joblib` and loaded identically during inference in the Streamlit application, ensuring that new patient data undergoes exactly the same transformation.

---

## 7. Data Augmentation (SMOTE)

### The Class Imbalance Problem

After stratified splitting, the training set contains:

- Label 0 (Non-Progressive): 622 samples
- Label 1 (Progressive): 395 samples

This 61%/39% imbalance can bias models toward predicting the majority class, resulting in **low sensitivity** — the most clinically dangerous failure mode, since it corresponds to missed progressive myopia cases.

### Why SMOTE?

Simple random oversampling (duplicating minority class records) introduces no new information and causes models to memorise specific records, leading to overfitting. **SMOTE (Synthetic Minority Over-sampling Technique)** (Chawla et al., 2002) generates synthetic minority-class samples by interpolating between real training samples:

1. For each minority class sample, identify its k nearest neighbours (k=5 by default)
2. Randomly select one of those neighbours
3. Generate a new synthetic sample at a random position along the line segment between the original and selected neighbour in feature space

This creates plausible synthetic patients that improve the model's ability to learn the decision boundary for progressive myopia without memorising specific training records.

### Augmentation Result

| Class               | Before SMOTE    | After SMOTE     |
| ------------------- | --------------- | --------------- |
| Non-Progressive (0) | 622             | 622             |
| Progressive (1)     | 395             | 622             |
| **Total**     | **1,017** | **1,244** |

### Three SMOTE Strategies Provided

| Strategy            | Description                                       | When to Use                                                                                      |
| ------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `smote` (default) | Standard SMOTE — interpolates between k-NN pairs | General use; balanced datasets with reasonably clean boundaries                                  |
| `smote_tomek`     | SMOTE + Tomek link removal                        | Removes noisy borderline samples after oversampling; improves boundary clarity                   |
| `smoteenn`        | SMOTE + Edited Nearest Neighbours                 | More aggressive cleaning; removes majority samples whose label disagrees with k-NN majority vote |

**Important:** SMOTE is applied **only to the training set**, after splitting. Validation and test sets contain only real patient data and are never augmented. This ensures that evaluation metrics reflect performance on the natural data distribution.

---

## 8. Model Architecture

### 8.1 Base Models — 11 Classifiers

Eleven classifiers spanning the full complexity spectrum are trained and evaluated, enabling a comprehensive comparative study.

#### Linear Models

| Model                         | Key Hyperparameters                                       | Notes                                                                                                                       |
| ----------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | C=1.0, max_iter=2000, solver=lbfgs, class_weight=balanced | Interpretable linear baseline; coefficients directly indicate feature importance; strong regularisation via class weighting |

#### Tree-Based Models

| Model                   | Key Hyperparameters                                                   | Notes                                                                                                      |
| ----------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Decision Tree** | max_depth=8, min_samples_split=10, class_weight=balanced              | Single tree; fully interpretable; depth-limited to reduce overfitting; included as a weak learner baseline |
| **Random Forest** | n_estimators=300, min_samples_leaf=2, class_weight=balanced_subsample | Bootstrap aggregation of 300 decision trees; robust to noise; naturally handles feature interactions       |
| **Extra Trees**   | n_estimators=300, min_samples_leaf=2, class_weight=balanced_subsample | Randomises both features and split thresholds; faster training, often more regularised than Random Forest  |

#### Gradient Boosting Models

| Model                       | Key Hyperparameters                                                                   | Notes                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Gradient Boosting** | n_estimators=300, lr=0.05, max_depth=4, subsample=0.8                                 | Sequential tree building where each tree corrects the residuals of the previous; stochastic subsampling reduces overfitting    |
| **XGBoost**           | n_estimators=300, lr=0.05, max_depth=5, colsample_bytree=0.8                          | Regularised boosting with second-order gradient information; built-in L1/L2 regularisation; industry standard for tabular data |
| **LightGBM**          | n_estimators=300, lr=0.05, num_leaves=31, colsample_bytree=0.8, class_weight=balanced | Leaf-wise tree growth instead of depth-wise; faster training; achieves best AUC-ROC in this study                              |
| **AdaBoost**          | n_estimators=200, learning_rate=0.5                                                   | Adaptive boosting of weak decision stumps; assigns higher weight to misclassified samples at each round                        |

#### Other Classifiers

| Model                        | Key Hyperparameters                                                                         | Notes                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **SVM (RBF)**          | C=10, gamma=scale, probability=True, class_weight=balanced                                  | Maximum-margin classifier in RBF kernel-transformed feature space; achieves highest raw accuracy in this study              |
| **KNN**                | n_neighbors=7, weights=distance, metric=minkowski                                           | Non-parametric instance-based learner; classification by majority vote of 7 nearest neighbours weighted by inverse distance |
| **MLP Neural Network** | hidden_layers=(128, 64, 32), activation=relu, solver=adam, alpha=0.001, early_stopping=True | Three-layer fully connected network; early stopping on validation loss prevents overfitting                                 |

### 8.2 Stacking Ensemble

A two-level stacking ensemble is constructed using the four strongest base learners as Level 0 estimators, with Logistic Regression as the Level 1 meta-learner.

```
Level 0 — Base Learners
┌───────────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐
│ Random Forest │  │  XGBoost  │  │ LightGBM │  │ SVM (RBF) │
└───────┬───────┘  └─────┬─────┘  └────┬─────┘  └─────┬─────┘
        │                │              │               │
        └────────────────┴──────────────┴───────────────┘
                                 │
              [Out-of-fold predictions via 5-fold CV]
                                 │
Level 1 — Meta-Learner
                  ┌──────────────────────────┐
                  │   Logistic Regression    │
                  │   C=0.1, max_iter=2000   │
                  └──────────────────────────┘
                                 │
                        Final prediction
```

**Why stacking works:** Each base learner captures different patterns in the data. The meta-learner learns which base learners to trust more for which types of inputs, producing predictions more accurate than any individual model alone.

**Out-of-fold cross-validation:** Base learner predictions for the meta-learner training are generated using 5-fold cross-validation. Each fold's held-out predictions are used, ensuring the meta-learner never sees predictions made on data the base learner was trained on — preventing leakage between the two levels.

---

## 9. Training Pipeline

### Cross-Validation Protocol

All 11 base models are evaluated using **5-Fold Stratified Cross-Validation** on the SMOTE-augmented training set, before final model fitting. Stratified folding ensures equal class proportions in each fold.

```
SMOTE-Augmented Training Set (1,244 samples)
|
+-- Fold 1: Train on 995, validate on 249  -->  Acc, Precision, Recall, F1, AUC
+-- Fold 2: Train on 995, validate on 249  -->  ...
+-- Fold 3: Train on 996, validate on 248  -->  ...
+-- Fold 4: Train on 996, validate on 248  -->  ...
+-- Fold 5: Train on 996, validate on 248  -->  Mean +/- SD across all 5 folds
```

Cross-validation provides an unbiased estimate of model generalisation performance and reveals overfitting tendencies that would not be visible from a single train/test split.

### Final Model Fitting

After cross-validation metrics are recorded, each model is **re-fitted on the entire SMOTE-augmented training set** (1,244 samples) to maximise the amount of training data before evaluating on the held-out test set.

### Complete End-to-End Flow

```
Raw CSV (1,454 rows, 15 columns)
         |
         v
Feature Engineering (47 columns, 32 new features)
         |
         v
Preprocessing: dedup -> cap outliers -> impute -> split -> scale
         |
         +----> Test set (291 samples) -- set aside, never touched again
         |
         +----> Training set (1,017) -> SMOTE -> Augmented (1,244)
                         |
                         v
               5-Fold CV on all 11 models
               (metrics: Acc, Prec, Recall, F1, AUC per fold)
                         |
                         v
               Full fit on all 1,244 augmented training samples
                         |
                         v
               Predict on test set (291 samples, never seen before)
                         |
                         v
               Compute all test metrics + bootstrap CI
                         |
                         v
               Save best model artifact + Streamlit app files
```

---

## 10. Model Evaluation Results

### Test Set Performance (n = 291, 20% holdout)

All metrics below are computed on the **unseen test set** that was held out from the beginning. Models are ranked by AUC-ROC.

| Rank | Model               | AUC-ROC          | F1-Score | Sensitivity | Specificity | Accuracy         | Precision | NPV    |
| ---- | ------------------- | ---------------- | -------- | ----------- | ----------- | ---------------- | --------- | ------ |
| 1    | **LightGBM**  | **0.9960** | 0.9732   | 0.9646      | 0.9888      | 0.9794           | 0.9820    | 0.9778 |
| 2    | XGBoost             | 0.9956           | 0.9646   | 0.9646      | 0.9775      | 0.9725           | 0.9646    | 0.9775 |
| 3    | Random Forest       | 0.9954           | 0.9686   | 0.9558      | 0.9888      | 0.9759           | 0.9818    | 0.9724 |
| 4    | AdaBoost            | 0.9954           | 0.9600   | 0.9558      | 0.9775      | 0.9691           | 0.9643    | 0.9721 |
| 5    | Stacking Ensemble   | 0.9951           | 0.9732   | 0.9646      | 0.9888      | 0.9794           | 0.9820    | 0.9778 |
| 6    | Gradient Boosting   | 0.9950           | 0.9689   | 0.9646      | 0.9831      | 0.9759           | 0.9732    | 0.9777 |
| 7    | Logistic Regression | 0.9946           | 0.9686   | 0.9558      | 0.9888      | 0.9759           | 0.9818    | 0.9724 |
| 8    | Extra Trees         | 0.9943           | 0.9727   | 0.9469      | 1.0000      | 0.9794           | 1.0000    | 0.9674 |
| 9    | SVM (RBF)           | 0.9913           | 0.9821   | 0.9735      | 0.9944      | **0.9863** | 0.9910    | 0.9833 |
| 10   | MLP Neural Net      | 0.9908           | 0.9633   | 0.9292      | 1.0000      | 0.9725           | 1.0000    | 0.9570 |
| 11   | KNN                 | 0.9839           | 0.9643   | 0.9558      | 0.9831      | 0.9725           | 0.9730    | 0.9722 |
| 12   | Decision Tree       | 0.9529           | 0.9013   | 0.9292      | 0.9157      | 0.9210           | 0.8750    | 0.9532 |

### Best Model: LightGBM — Complete Summary

| Metric                    | Value             | Interpretation                                                      |
| ------------------------- | ----------------- | ------------------------------------------------------------------- |
| AUC-ROC                   | 0.9960            | Near-perfect discrimination between progressive and non-progressive |
| 95% Bootstrap CI (AUC)    | [0.9900, 0.9998]  | Narrow CI confirms reliable performance, not a statistical artefact |
| F1-Score                  | 0.9732            | Excellent balance between precision and sensitivity                 |
| Sensitivity (Recall)      | 0.9646            | 96.5% of progressive cases correctly identified                     |
| Specificity               | 0.9888            | 98.9% of non-progressive cases correctly identified                 |
| Precision (PPV)           | 0.9820            | When predicted progressive, correct 98.2% of the time               |
| Negative Predictive Value | 0.9778            | When predicted non-progressive, correct 97.8% of the time           |
| Overall Accuracy          | 0.9794            | 97.9% of all predictions are correct                                |
| CV Accuracy (Mean +/- SD) | 0.9839 +/- 0.0092 | Consistent performance across 5 cross-validation folds              |

### Key Observations

1. **Gradient boosting models dominate**: LightGBM, XGBoost, and Gradient Boosting occupy the top 3 AUC positions, confirming their superior suitability for structured tabular clinical data compared to neural networks and kernel methods.
2. **SVM achieves highest accuracy**: Despite a lower AUC than gradient boosters, SVM (RBF) achieves the highest raw accuracy (0.9863) and precision (0.9910), with a tighter decision boundary that minimises false positives at the expense of a slightly lower AUC.
3. **Logistic Regression competitive**: The linear model achieves AUC 0.9946 — extremely close to the top gradient boosting models. This indicates that the engineered features are highly discriminative and largely linearly separable, which is an encouraging finding for clinical transparency.
4. **Decision Tree is weakest**: AUC 0.9529 vs 0.9960 for LightGBM confirms the well-established finding that single trees overfit without ensemble aggregation, even with depth regularisation.
5. **Stacking ensemble validates individual models**: The stacking ensemble (AUC 0.9951) performs similarly to the best individual model rather than surpassing it, suggesting that the base learners are already highly correlated in their predictions — a sign of dataset ceiling effects.

---

## 11. Evaluation Metrics Explained

### Primary Metric — AUC-ROC

The **Area Under the Receiver Operating Characteristic Curve** is the primary ranking metric in this study. It represents the probability that the model assigns a higher probability score to a randomly selected progressive case than to a randomly selected non-progressive case.

- AUC = 1.0: perfect discrimination
- AUC = 0.5: random chance (no predictive ability)

The ROC curve plots True Positive Rate (Sensitivity) against False Positive Rate (1 - Specificity) at every possible classification threshold. AUC is threshold-independent and robust to class imbalance.

### Clinical Metrics Defined

| Metric                                    | Formula                                         | Clinical Interpretation                                                                                                                                                |
| ----------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sensitivity** (Recall, TPR)       | TP / (TP + FN)                                  | Fraction of true progressive cases correctly identified. Missing progressive cases (FN) is the most dangerous clinical error — the model prioritises maximising this. |
| **Specificity** (TNR)               | TN / (TN + FP)                                  | Fraction of true non-progressive cases correctly identified. Low specificity means unnecessary treatment for stable patients.                                          |
| **Precision** (PPV)                 | TP / (TP + FP)                                  | Among patients predicted to be progressive, the fraction who truly are. High precision reduces unnecessary intervention burden.                                        |
| **NPV** (Negative Predictive Value) | TN / (TN + FN)                                  | Among patients predicted to be non-progressive, the fraction who truly are. High NPV provides clinician confidence in negative predictions.                            |
| **F1-Score**                        | 2 x (Precision x Recall) / (Precision + Recall) | Harmonic mean of precision and sensitivity. Balances the two and is robust to class imbalance. Used as secondary ranking metric.                                       |
| **Accuracy**                        | (TP + TN) / N                                   | Overall fraction of correct predictions. Can be misleading under class imbalance; included for completeness.                                                           |

### Confusion Matrix Terms

```
                    Predicted
                    Non-Prog  |  Progressive
Actual  Non-Prog  |    TN     |      FP      (False Alarm)
        Progressive|    FN     |      TP      (Correct Detection)
                  (Miss)
```

- **TP (True Positive):** Progressive correctly predicted as progressive
- **TN (True Negative):** Non-progressive correctly predicted as non-progressive
- **FP (False Positive):** Non-progressive incorrectly predicted as progressive (over-treatment risk)
- **FN (False Negative):** Progressive incorrectly predicted as non-progressive (missed diagnosis risk)

### Calibration

**Calibration curves** (reliability diagrams) assess whether predicted probabilities match observed event rates. A model with predicted probability of 0.8 should be correct approximately 80% of the time. Well-calibrated models are essential for clinical risk communication — a clinician quoting a "78% progression probability" needs that number to be meaningful.

**Brier Score** quantifies overall calibration quality (lower is better; 0 = perfect, 0.25 = no-skill model).

### Bootstrap Confidence Intervals

A 95% bootstrap confidence interval (n=1,000 resamples) is computed for the best model's AUC. This provides a statistically valid uncertainty estimate that accounts for the finite test set size. A narrow CI confirms that the reported AUC reflects genuine predictive ability rather than sampling luck.

---

## 12. Multi-Agent Architecture

The project implements a **modular multi-agent architecture** where each agent is a self-contained, independently executable Python module responsible for a specific pipeline phase. This design enables:

- Independent testing and debugging of each phase
- Parallel development of different pipeline components
- Easy extension with new agents without modifying existing code
- Clear separation of concerns for academic reproducibility

```
run_all_agents.py  (Sequential Orchestrator)
        |
        |-- Agent 01: Data Analyzer
        |        Load raw data, generate inspection report,
        |        outlier analysis, normality tests (Shapiro-Wilk)
        |
        |-- Agent 02: Preprocessor & Feature Engineering
        |        Encode categoricals, compute 21 engineered features,
        |        clean, split, scale
        |
        |-- Agent 03: EDA Visualizer
        |        15 static plots + 3 interactive HTML visualisations
        |
        |-- Agent 04: Model Trainer
        |        Train all 11 classifiers + stacking ensemble,
        |        5-fold CV, record all metrics
        |
        |-- Agent 05: Evaluator
        |        ROC, PR curves, confusion matrices, calibration,
        |        SHAP, bootstrap CI, learning curves
        |
        +-- Agent 06: Inference & Diagrams
                 Save model artifacts, generate architectural diagrams
```

### Running the Agent Pipeline

```bash
# Run all agents sequentially
python agents/run_all_agents.py

# Run only specific agents (e.g., retrain and re-evaluate)
python agents/run_all_agents.py --agents 4 5

# Run a single agent standalone for debugging
python agents/agent_01_data_analyzer.py
```

---

## 13. Generated Outputs

### Preprocessing Figures (`outputs/figures/preprocessing/`)

| Filename                              | Description                                                           |
| ------------------------------------- | --------------------------------------------------------------------- |
| `01_class_distribution_smote.png`   | Side-by-side bar charts showing class balance before and after SMOTE  |
| `02_missing_values.png`             | Missing value heatmap and per-column count bar chart                  |
| `03_outlier_detection.png`          | IQR scatter plots for all raw features — outliers highlighted in red |
| `04_outlier_capping_comparison.png` | Box plots showing the effect of Winsorising on all raw features       |
| `05_split_distribution.png`         | Pie charts of label proportion in train, validation, and test splits  |
| `06_scaling_comparison.png`         | Histogram comparison of 6 features before and after StandardScaler    |

### EDA Figures (`outputs/figures/eda/`)

| Filename                                | Description                                                        |
| --------------------------------------- | ------------------------------------------------------------------ |
| `01_label_distribution.png`           | Pie, count bars, gender x label, age group x label                 |
| `02_feature_distributions.png`        | Overlaid histograms + KDE for each raw feature, split by label     |
| `03_correlation_heatmap.png`          | Full Pearson correlation matrix (lower triangle)                   |
| `04_correlation_with_target.png`      | Ranked horizontal bar chart of feature-to-label correlation        |
| `05_boxplots_by_class.png`            | Box plots with Mann-Whitney U p-values and significance stars      |
| `06_violin_plots.png`                 | Violin plots showing full distribution shape by label              |
| `07_pairplot_top_features.png`        | Pairwise scatter matrix for the 4 most discriminative features     |
| `08_statistical_significance.png`     | -log10(p-value) and Cohen's d effect size for all features         |
| `09_clinical_grouping_analysis.png`   | Risk score distributions, astigmatism axis type breakdown          |
| `10_engineered_features_by_class.png` | Box plots for all 21 engineered features                           |
| `11_descriptive_stats_table.png`      | Mean, SD, min, max table comparing progressive vs. non-progressive |
| `12_3d_scatter.html`                  | Interactive 3D scatter: Kmax vs. Astigmatism vs. Pachymetry        |
| `13_sunburst.html`                    | Interactive sunburst: Gender → Age Group → Label hierarchy       |

### Evaluation Figures (`outputs/figures/evaluation/`)

| Filename                        | Description                                                 |
| ------------------------------- | ----------------------------------------------------------- |
| `roc_curves.png`              | ROC curves for all 12 models with AUC annotations           |
| `precision_recall_curves.png` | PR curves for all models                                    |
| `confusion_matrices.png`      | Normalised confusion matrices for all models (%)            |
| `calibration_curves.png`      | Reliability diagrams with Brier scores                      |
| `cv_boxplots.png`             | Box plots of 5-fold CV score distributions per model        |
| `model_metric_heatmap.png`    | All metrics x all models as a colour-coded heatmap          |
| `model_comparison_bars.png`   | Grouped bar chart comparing key metrics across models       |
| `feature_importance.png`      | Feature importance from tree-based models                   |
| `learning_curves.png`         | Train vs. validation AUC as a function of training set size |

### Saved Artifacts

| File                          | Location             | Description                          |
| ----------------------------- | -------------------- | ------------------------------------ |
| `best_model.joblib`         | `outputs/models/`  | Serialised LightGBM model            |
| `scaler.joblib`             | `outputs/models/`  | Fitted StandardScaler                |
| `feature_cols.joblib`       | `outputs/models/`  | Ordered feature name list            |
| `model_results_summary.csv` | `outputs/reports/` | Full metrics table for all 12 models |

---

## 14. How to Run

### Prerequisites

```bash
# Python 3.10 or higher is required
python --version

# Create virtual environment
python -m venv myopia
myopia\Scripts\activate        # Windows
source myopia/bin/activate     # Linux / macOS
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Complete pipeline — all 12 phases
python run_pipeline.py

# Recommended first run (skips slow optional phases, takes ~3 minutes)
python run_pipeline.py --skip-shap --skip-diagrams

# Individual phase control
python run_pipeline.py --skip-eda            # Skip EDA (saves ~1 minute)
python run_pipeline.py --skip-shap           # Skip SHAP explanations
python run_pipeline.py --skip-diagrams       # Skip architectural diagrams

# Choose SMOTE strategy
python run_pipeline.py --smote smote_tomek
python run_pipeline.py --smote smoteenn
```

### Pipeline Phases Reference

| Phase | Description                            | Approximate Duration |
| ----- | -------------------------------------- | -------------------- |
| 1     | Data loading and inspection report     | < 5 seconds          |
| 2     | Feature engineering (32 new features)  | < 5 seconds          |
| 3     | Preprocessing (clean, split, scale)    | < 5 seconds          |
| 4     | SMOTE augmentation                     | < 5 seconds          |
| 5     | Preprocessing visualisations (6 plots) | ~30 seconds          |
| 6     | EDA visualisations (15 plots + 3 HTML) | ~60 seconds          |
| 7     | Multi-model training (11 + stacking)   | ~60 seconds          |
| 8     | Results summary table                  | < 5 seconds          |
| 9     | Evaluation plots (9 figures)           | ~60 seconds          |
| 10    | SHAP explanations (best model)         | ~120 seconds         |
| 11    | Save model artifacts                   | < 5 seconds          |
| 12    | Architectural diagrams                 | ~30 seconds          |

---

## 15. Streamlit Clinical App

A four-page Streamlit application provides an interactive clinical interface:

```bash
streamlit run app/streamlit_app.py
```

### Pages

| Page                         | Content                                                                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Patient Prediction** | Enter raw clinical measurements; receive probability of progression as a gauge chart, risk level label, and breakdown of triggered clinical flags |
| **Model Performance**  | View the full evaluation metrics table, ROC curve, and confusion matrix of the deployed model                                                     |
| **Dataset Explorer**   | Interactive exploration of the processed dataset with filters and visualisations                                                                  |
| **About**              | Project description, methodology summary, and clinical context                                                                                    |

### Inference Workflow

```
Clinician enters 13 raw measurements
           |
           v
App applies identical feature engineering (21 new features computed)
           |
           v
StandardScaler normalises using training-set statistics (loaded from scaler.joblib)
           |
           v
LightGBM model predicts P(progressive) as a probability (0.0 - 1.0)
           |
           v
Result displayed as gauge chart + risk level + clinical flag summary
```

This ensures that inference is mathematically identical to training — the same features, the same scaling, the same model — which is a fundamental requirement for valid clinical deployment.

---

## 16. Requirements

```
# Core ML
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

# Analysis
statsmodels>=0.14

# Development
jupyter
ipykernel
```

---

## 17. References

1. **Rabinowitz, Y.S.** (1998). Keratoconus. *Survey of Ophthalmology*, 42(4), 297–319. https://doi.org/10.1016/S0039-6257(97)00089-0
2. **Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P.** (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
3. **Randleman, J.B., Woodward, M., Lynn, M.J., & Stulting, R.D.** (2008). Risk Assessment for Ectasia after Corneal Refractive Surgery. *Journal of Refractive Surgery*, 24(9), 895–902.
4. **Rabinowitz, Y.S.** (2002). Videokeratographic indices to aid in screening for keratoconus. *Journal of Refractive Surgery*, 11(5), 371–379.
5. **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
6. **Ke, G., Meng, Q., Finley, T., et al.** (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30.
7. **Lundberg, S.M., & Lee, S.I.** (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *Advances in Neural Information Processing Systems*, 30.
8. **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
9. **Vapnik, V.** (1995). *The Nature of Statistical Learning Theory*. Springer, New York.
10. **Wolpert, D.H.** (1992). Stacked Generalisation. *Neural Networks*, 5(2), 241–259.
11. **Holden, B.A., et al.** (2016). Global Prevalence of Myopia and High Myopia and Temporal Trends from 2000 through 2050. *Ophthalmology*, 123(5), 1036–1042.

---

## Appendix A — Clinical Threshold Reference

| Measurement                         | Normal Range | Risk Threshold                           | Source              |
| ----------------------------------- | ------------ | ---------------------------------------- | ------------------- |
| Kmax (D)                            | < 45.0       | > 47.2 (keratoconus suspect)             | Rabinowitz (1998)   |
| Central pachymetry (um)             | > 520        | < 500 (high risk), < 510 (moderate risk) | Randleman (2008)    |
| pachy_diff: central - thinnest (um) | < 20         | > 30 (abnormal gradient)                 | Clinical consensus  |
| pachy_ratio: thinnest/central       | > 0.96       | < 0.94 (Belin-Ambrosio threshold)        | Belin-Ambrosio      |
| Astigmatism magnitude (D)           | < 1.5        | > 2.5 (irregular), > 3.0 (high)          | Standard refraction |
| Anterior Q-value                    | -0.2 to -0.3 | > 0 (oblate — abnormal)                 | Topographer norms   |
| Thinnest point displacement (mm)    | < 0.5        | > 1.0 (decentred thinning)               | CLMI literature     |
| Ectasia Risk Score                  | 0            | >= 3 (elevated), >= 5 (high)             | Randleman ERSS      |

---

## Appendix B — Reproducibility Checklist

To reproduce the exact results reported in this document:

- [ ] Python version: 3.10 or higher
- [ ] All packages installed from `requirements.txt`
- [ ] Raw data file at `data/raw/clinical_data_and_labels.csv`
- [ ] Run: `python run_pipeline.py --skip-shap --skip-diagrams`
- [ ] `RANDOM_STATE = 42` in `src/config.py` (default — do not change)
- [ ] SMOTE strategy: `smote` (default — do not change)
- [ ] Split ratios: 70/10/20 (default — do not change)

All random seeds are controlled via `RANDOM_STATE = 42` which is passed to every stochastic operation including train/test split, SMOTE, and all model initialisations.

---

*This research was conducted as part of a MPhil degree in Ophthalmology. All patient records are fully anonymised. The predictive model outputs are intended to support — not replace — the clinical judgement of qualified ophthalmologists. No clinical decisions should be made based solely on model output.*
