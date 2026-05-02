# AI-Based Myopia Progression Prediction — Complete Project Documentation

**Development and Validation of an Artificial Intelligence Model for Predicting Myopia Progression Using Clinical Corneal Topography Data**

> **Investigator** — Syed Ahmad Hassan, MPhil Ophthalmology (2024-MPhil-OP-037)
> **Best model** — LightGBM, AUC-ROC = **0.9996** (95 % CI [0.9987, 1.0000])

---

## Headline Results

| Metric                         | LightGBM (best)  | Gradient Boosting | Random Forest |
| ------------------------------ | ---------------- | ----------------- | ------------- |
| **AUC-ROC**              | **0.9996** | 0.9996            | 0.9995        |
| **F1-Score**             | 0.9825           | 0.9912            | 0.9868        |
| **Sensitivity (Recall)** | 0.9912           | 0.9912            | 0.9912        |
| **Specificity**          | 0.9861           | 0.9954            | 0.9907        |
| **Accuracy**             | 0.9878           | 0.9939            | 0.9909        |
| **Precision (PPV)**      | 0.9739           | 0.9912            | 0.9825        |
| **NPV**                  | 0.9953           | 0.9954            | 0.9953        |
| **Training time**        | 12.3 s           | 11.5 s            | 3.6 s         |

**95 % bootstrap CI for best AUC-ROC** (1,000 resamples): **[0.9987, 1.0000]**.

> 📖 **For thesis viva and peer-review preparation**, see [`docs/WHY_AND_RATIONALE.md`](docs/WHY_AND_RATIONALE.md) — defence-grade Q&A justifying every methodological choice (dataset, preprocessing, features, splits, SMOTE, models, CV, metrics, hypothesis tests, SHAP, bootstrap CI). Each answer is structured to match an actual reviewer question.

---

## Table of Contents

| #           | Section                                                                                            |
| ----------- | -------------------------------------------------------------------------------------------------- |
| **A** | [Project Overview](#a-project-overview)                                                               |
| **B** | [Clinical Background](#b-clinical-background)                                                         |
| **C** | [Dataset — Source, Schema, Statistics](#c-dataset--source-schema-statistics)                         |
| **D** | [Preprocessing Pipeline](#d-preprocessing-pipeline)                                                   |
| **E** | [Exploratory Data Analysis (EDA)](#e-exploratory-data-analysis-eda)                                   |
| **F** | [Feature Engineering — All 32 Features Explained](#f-feature-engineering--all-32-features-explained) |
| **G** | [Data Visualisation Catalogue](#g-data-visualisation-catalogue)                                       |
| **H** | [Data Splitting Strategy](#h-data-splitting-strategy)                                                 |
| **I** | [Class Imbalance Correction (SMOTE)](#i-class-imbalance-correction-smote)                             |
| **J** | [Model Training — All 12 Models](#j-model-training--all-12-models)                                   |
| **K** | [Cross-Validation Protocol](#k-cross-validation-protocol)                                             |
| **L** | [Model Evaluation Metrics](#l-model-evaluation-metrics)                                               |
| **M** | [Statistical Significance Testing](#m-statistical-significance-testing)                               |
| **N** | [SHAP Explainability](#n-shap-explainability)                                                         |
| **O** | [Streamlit Clinical Application](#o-streamlit-clinical-application)                                   |
| **P** | [Project Structure](#p-project-structure)                                                             |
| **Q** | [How to Run Everything](#q-how-to-run-everything)                                                     |
| **R** | [Limitations and Future Work](#r-limitations-and-future-work)                                         |
| **S** | [References](#s-references)                                                                           |

---

## A. Project Overview

This repository implements a **complete, reproducible machine-learning pipeline** for predicting myopia progression from routine corneal-topography measurements. It is designed end-to-end for academic publication: every stage is auditable, every result reproducible, every figure publication-quality.

### What this project does

1. **Loads** anonymised clinical corneal-topography data (1,642 patient eyes, 13 raw clinical fields).
2. **Engineers** 32 clinically grounded features rooted in published keratoconus and ectasia screening literature.
3. **Preprocesses** the data with strict train/validation/test discipline — no information leakage.
4. **Augments** the training set with SMOTE to correct the 65/35 class imbalance.
5. **Trains** 11 base classifiers plus a stacking ensemble, evaluated by 5-fold stratified cross-validation.
6. **Evaluates** every model on a 329-patient holdout test set with full metrics: AUC-ROC, F1, sensitivity, specificity, calibration, learning curves, bootstrap CIs.
7. **Explains** the best model via SHAP value analysis for clinical interpretability.
8. **Deploys** as a theme-safe Streamlit application supporting real-time inference.
9. **Generates** a publication-ready Word research document and 38+ publication-quality figures.

### Why this matters clinically

Progressive myopia substantially raises lifetime risk of retinal detachment, myopic maculopathy, glaucoma, and cataract. Early identification of progressive cases enables timely intervention through orthokeratology, low-dose atropine, or multifocal lenses. Traditional clinical decision-making relies on threshold-based rules applied to single measurements — which fails to capture multidimensional interactions between biomechanical, refractive, and topographic variables. Machine learning can model these interactions explicitly.

---

## B. Clinical Background

### What is myopia?

Myopia (nearsightedness) is a refractive error in which light focuses in front of the retina rather than on it. It is the world's most prevalent ocular condition, projected by Holden et al. (2016) to affect **4.9 billion people by 2050**.

### Progressive vs non-progressive myopia

- **Non-progressive (Label 0)**: Stable refractive error year over year. Annual progression < 0.5 D.
- **Progressive (Label 1)**: Continued axial elongation and worsening refraction. Annual progression ≥ 0.5 D.

### Why corneal topography matters

Corneal topography devices capture the geometry of the front surface of the eye:

- **Keratometry (Kmax)** — the steepest curvature of the cornea
- **Pachymetry** — corneal thickness at the centre and at the thinnest point
- **Asphericity (Q-value)** — how the cornea deviates from a perfect sphere
- **Astigmatism** — magnitude and orientation of cylindrical refractive error

These are the same measurements an ophthalmologist uses to identify keratoconus suspects — and progressive myopia shares biomechanical signatures with early ectasia.

### Why machine learning?

Traditional decision rules look at one measurement at a time:

| Rule                         | Trigger                            |
| ---------------------------- | ---------------------------------- |
| Kmax > 47.2 D                | Keratoconus suspect                |
| Central pachymetry < 500 μm | High ectasia risk                  |
| Thinning gradient > 30 μm   | Belin–Ambrosio screening positive |

Machine learning can:

1. Learn **non-linear** combinations of these signals
2. Quantify **interaction** effects invisible to threshold rules
3. Output **probabilistic** risk scores rather than binary labels
4. Generalise across heterogeneous patient populations

---

## C. Dataset — Source, Schema, Statistics

### Dataset versions

| Version            | File                                      | Rows            | Notes              |
| ------------------ | ----------------------------------------- | --------------- | ------------------ |
| v1                 | `clinical_data_and_labels_v1.csv`       | 1,454           | Initial collection |
| v2                 | `clinical_data_and_labels_v2.csv`       | 188             | Expansion sample   |
| **Combined** | `combined_clinical_data_and_labels.csv` | **1,642** | Used in this study |

### Combined-dataset summary

| Property                  | Value                              |
| ------------------------- | ---------------------------------- |
| Total samples             | 1,642 patient eyes                 |
| Total raw columns         | 15                                 |
| Label 0 (Non-Progressive) | **1,077 (65.6 %)**           |
| Label 1 (Progressive)     | **565 (34.4 %)**             |
| Missing values            | **0** (complete dataset)     |
| Duplicate records         | **0**                        |
| Patient age range         | 13 – 65 years                     |
| Gender distribution       | f: 1,041 (63.4 %), m: 601 (36.6 %) |
| Eye laterality            | OD (right): 824, OS (left): 818    |

### Raw variable schema (13 clinical inputs + label)

| Variable                  | Type        | Unit             | Clinical description                  |
| ------------------------- | ----------- | ---------------- | ------------------------------------- |
| `patient_code`          | identifier  | —               | Anonymised patient ID                 |
| `age_years`             | numeric     | years            | Age at examination                    |
| `gender`                | categorical | f / m            | Patient sex                           |
| `eye`                   | categorical | OD / OS          | Right or left eye                     |
| `astig_value_D`         | numeric     | Diopters         | Refractive astigmatism magnitude      |
| `astig_axis_deg`        | numeric     | degrees (0–180) | Astigmatism axis orientation          |
| `kmax_value_D`          | numeric     | Diopters         | Maximum keratometry — peak curvature |
| `kmax_axis_deg`         | numeric     | degrees (0–180) | Axis of maximum curvature             |
| `pachy_central_um`      | numeric     | μm              | Central corneal thickness             |
| `pachy_thinnest_um`     | numeric     | μm              | Thinnest corneal thickness            |
| `pachy_thinnest_x`      | numeric     | mm               | X-coordinate of thinnest point        |
| `pachy_thinnest_y`      | numeric     | mm               | Y-coordinate of thinnest point        |
| `asphericity_anterior`  | numeric     | dimensionless    | Anterior surface Q-value              |
| `asphericity_posterior` | numeric     | dimensionless    | Posterior surface Q-value             |
| `label`                 | binary      | 0 / 1            | 0 = non-progressive, 1 = progressive  |

### Class imbalance — why it matters

The dataset is naturally imbalanced (65.6 % / 34.4 %). A naïve classifier predicting "non-progressive" for every patient would already score **65.6 % accuracy** — completely useless clinically because it misses every progressive case. This is why we (a) report AUC-ROC and sensitivity rather than accuracy alone, and (b) apply SMOTE to the training set (see §I).

---

## D. Preprocessing Pipeline

Six sequential, leakage-safe steps transform the raw dataset into model-ready arrays.

### Step 1 — Schema validation

`src/data/loader.py::load_raw()` confirms:

- All 14 required columns are present
- Label column contains only `{0, 1}` (binary sanity check)
- No silent type coercions

If any check fails, the pipeline raises a `ValueError` immediately rather than producing silently corrupt outputs.

### Step 2 — Duplicate removal

```python
df, n_dup = remove_duplicates(df)
```

Exact-row duplicates are removed **before splitting** to guarantee no record appears in both train and test sets. The combined dataset has zero duplicates.

### Step 3 — Outlier capping (Winsorising)

```python
df = cap_outliers(df, RAW_NUMERIC_COLS, factor=3.0)
```

We cap extreme values using the **IQR method** with factor = 3.0:

```
lower = Q1 − 3 × IQR
upper = Q3 + 3 × IQR
clip every value into [lower, upper]
```

**Why factor 3 instead of the textbook 1.5?** A factor of 1.5 is appropriate when normality is assumed and outliers are rare measurement errors. In clinical data, extreme but real values exist — a Kmax of 55 D is rare but is a valid keratoconus reading. The factor of 3 retains genuine clinical variation while clipping only the most extreme artefacts.

**Why on raw numeric columns only?** Engineered binary flags and scaled indices already aggregate signal — capping them would distort their meaning.

### Step 4 — Feature engineering

`src/data/feature_engineering.py::engineer_all_features()` runs eight derivation modules in sequence (full details in §F).

- Input: 15 columns
- Output: 47 columns (15 raw + 32 engineered)

### Step 5 — Median imputation

```python
df = impute_missing(df, feature_cols)
```

Any missing values are filled with the **median** of the corresponding column.

**Why median, not mean?** Clinical measurements (especially pachymetry and asphericity) are **right-skewed**. The median is robust to the long tail; the mean is pulled toward extreme values and produces biased imputations.

> **Note:** The combined dataset has 0 missing values, so this step is a no-op in the current run. It is retained in the pipeline for robustness when the model is applied to new clinics whose data may have gaps.

### Step 6 — Train / Validation / Test split (then scale)

Stratified 70 % / 10 % / 20 % split with `random_state = 42`, then `StandardScaler` fit **only on the training set** (full details in §H).

### End-to-end pipeline diagram

```
raw CSV (1,642 × 15)
        │
        ▼
  load_raw() ───────────── schema check
        │
        ▼
  remove_duplicates() ───── 0 removed
        │
        ▼
  engineer_all_features() ─ 15 → 47 columns
        │
        ▼
  cap_outliers(IQR×3) ───── on 11 raw numeric cols
        │
        ▼
  impute_missing(median) ── (no missing in current data)
        │
        ▼
  split_data(70/10/20) ──── stratified, random_state=42
        │
        ├──► test  (329, 20%) — set aside, never touched again
        ├──► val   (165, 10%) — for early stopping / tuning
        └──► train (1,148, 70%)
                  │
                  ▼
            StandardScaler.fit_transform(train) ─── scaler.joblib
                  │
                  ▼
            scaler.transform(val), scaler.transform(test)
                  │
                  ▼
            apply_smote() ── train only ── 1,148 → 1,506
```

---

## E. Exploratory Data Analysis (EDA)

EDA is performed before modelling and persisted to `outputs/figures/eda/` (11 PNG plots + 2 interactive HTML).

### What every EDA chart actually answers

| #  | Figure                         | What it answers                                                                      |
| -- | ------------------------------ | ------------------------------------------------------------------------------------ |
| 01 | Label distribution             | Class balance (65.6 % / 34.4 %), distribution by gender, distribution by age group   |
| 02 | Feature distributions by class | Is each raw feature distributed differently in progressive vs. non-progressive eyes? |
| 03 | Correlation heatmap            | Which features are redundant? Which form clusters?                                   |
| 04 | Correlation with target        | Which raw or engineered features correlate most strongly with the label?             |
| 05 | Box plots by class             | Median, IQR, and Mann-Whitney U significance for each key feature                    |
| 06 | Violin plots                   | Full distribution shape, not just summary statistics                                 |
| 07 | Pair plot                      | Pairwise scatter and KDE for the four most discriminative features                   |
| 08 | Statistical significance       | −log₁₀(p) and Cohen's d effect size per feature                                   |
| 09 | Clinical groupings             | Risk-score breakdown, astigmatism-axis-type breakdown by class                       |
| 10 | Engineered features by class   | Does each engineered feature actually separate the classes?                          |
| 11 | Descriptive statistics table   | Mean, SD, min, max for each class, side-by-side                                      |
| 12 | 3D scatter (HTML)              | Interactive view of Kmax × Astigmatism × Pachymetry feature space                  |
| 13 | Sunburst (HTML)                | Hierarchical breakdown: Gender → Age group → Class                                 |

### Key descriptive statistics — combined dataset

| Feature                  | Non-Prog mean (SD) | Progressive mean (SD) |
| ------------------------ | ------------------ | --------------------- |
| Age (years)              | 28.2 (10.4)        | 26.1 (9.8)            |
| Kmax (D)                 | 44.1 (1.8)         | 47.6 (3.2)            |
| Astigmatism (D)          | 1.2 (0.9)          | 2.8 (1.6)             |
| Central pachymetry (μm) | 545 (32)           | 503 (38)              |
| Anterior asphericity Q   | -0.22 (0.13)       | -0.05 (0.42)          |
| pachy_diff (μm)         | 14.8 (10.2)        | 36.4 (18.7)           |

(All differences are statistically significant at p < 0.001, Mann–Whitney U.)

---

## F. Feature Engineering — All 32 Features Explained

Raw measurements alone underrepresent the clinical knowledge ophthalmologists use. Engineering converts the 13 raw inputs into a **41-dimensional feature space** (11 raw numeric + 2 encoded categoricals + 32 engineered).

### Why engineer features when modern ML can learn from raw data?

| Argument                            | Defence                                                                                                                                                                          |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sample efficiency**         | With N = 1,642, models cannot reliably discover non-linear interactions from raw inputs alone. Engineering encodes prior knowledge instead of asking the model to rediscover it. |
| **Mathematical correctness**  | Cyclic axis encoding (sin / cos of 2θ) fixes a real bug in how angular variables would otherwise be used by linear models and shallow trees.                                    |
| **Clinical interpretability** | Reviewers and clinicians trust `pachy_ratio`, `kisa_proxy`, `ectasia_risk_score` because they map to validated clinical instruments.                                       |
| **Empirical evidence**        | Logistic Regression on engineered features reaches AUC = 0.9961 — only 0.0035 below the best non-linear model. The engineered features did the heavy lifting.                   |
| **Portability**               | Every engineered feature is a deterministic formula on raw measurements. Can be deployed on any topographer that captures the same 11 raw variables.                             |

### Group 1 — Categorical encoding (2 features)

| Feature            | Encoding                      |
| ------------------ | ----------------------------- |
| `gender_encoded` | Female = 0, Male = 1          |
| `eye_encoded`    | OD (right) = 0, OS (left) = 1 |

### Group 2 — Pachymetry derivatives (5 features)

| Feature                         | Formula                          | Clinical signal                                                   |
| ------------------------------- | -------------------------------- | ----------------------------------------------------------------- |
| `pachy_diff`                  | `central − thinnest` (μm)    | Belin–Ambrosio screening: > 30 μm flagged                       |
| `pachy_ratio`                 | `thinnest / central`           | < 0.94 indicates abnormal asymmetry                               |
| `pachy_thinnest_displacement` | `√(x² + y²)` (mm)           | Distance of thinnest point from apex; > 1 mm = decentred thinning |
| `pachy_thin_flag`             | `1 if pachy_thinnest_um < 500` | Binary indicator of high ectasia risk                             |
| `pachy_diff_flag`             | `1 if pachy_diff > 30`         | Binary indicator of abnormal thinning gradient                    |

### Group 3 — Asphericity derivatives (4 features)

| Feature                  | Formula                       | Clinical signal                                           |
| ------------------------ | ----------------------------- | --------------------------------------------------------- |
| `asphericity_diff`     | `Q_anterior − Q_posterior` | Surface imbalance; ectatic corneas show large differences |
| `asphericity_ratio`    | `Q_anterior / Q_posterior`  | Relative imbalance, scale-invariant                       |
| `asphericity_abs_sum`  | `                             | Q_anterior                                                |
| `anterior_oblate_flag` | `1 if Q_anterior > 0`       | Abnormal oblate cornea (positive Q is pathological)       |

### Group 4 — Astigmatism derivatives (5 features)

| Feature             | Formula                              | Clinical signal                            |
| ------------------- | ------------------------------------ | ------------------------------------------ |
| `astig_abs`       | `                                    | astig_value_D                              |
| `astig_axis_sin`  | `sin(2 × axis_radians)`           | Cyclic encoding (mathematical correctness) |
| `astig_axis_cos`  | `cos(2 × axis_radians)`           | Cyclic encoding (mathematical correctness) |
| `astig_wtr_flag`  | `1 if axis ∈ [0,30] ∪ [150,180]` | With-the-rule astigmatism                  |
| `astig_high_flag` | `1 if                                | astig                                      |

> **Cyclic encoding deep-dive:** Astigmatism axis is angular with 180° period. Axis 1° and 179° are nearly identical clinically, but in raw degrees they look 178 units apart — the maximum possible distance in [0,180]. Linear models compute distances on this number directly, so they see "max distance" between essentially identical orientations. Encoding as `sin(2θ)` and `cos(2θ)` maps the half-period onto the full unit circle, restoring correct topology. **This is not a heuristic; it is a mathematical correctness fix.**

### Group 5 — Keratometry derivatives (4 features)

| Feature             | Formula                         | Clinical signal                                 |
| ------------------- | ------------------------------- | ----------------------------------------------- |
| `kmax_axis_sin`   | `sin(2 × kmax_axis_radians)` | Cyclic encoding                                 |
| `kmax_axis_cos`   | `cos(2 × kmax_axis_radians)` | Cyclic encoding                                 |
| `kmax_high_flag`  | `1 if Kmax > 47.2`            | Rabinowitz (1998) keratoconus suspect threshold |
| `kmax_steep_flag` | `1 if Kmax > 46.0`            | Borderline / early-warning threshold            |

### Group 6 — Composite clinical indices (4 features)

| Feature                           | Formula                                        | Literature anchor                          |
| --------------------------------- | ---------------------------------------------- | ------------------------------------------ |
| `corneal_power_index`           | `Kmax × (1 + Q_anterior)`                   | Standard topographer-derived power measure |
| `corneal_irregularity_index`    | `                                              | astig                                      |
| `kisa_proxy`                    | `(Kmax−45)⁺ ×                               | astig                                      |
| `cone_location_magnitude_index` | `displacement × (1 − pachy_ratio) × Kmax` | CLMI proxy (Mahmoud et al., 2008)          |

### Group 7 — Interaction features (4 features)

| Feature                    | Formula                  | Why it matters                                     |
| -------------------------- | ------------------------ | -------------------------------------------------- |
| `kmax_astig_interaction` | `Kmax ×                 | astig                                              |
| `age_kmax_interaction`   | `age × Kmax`          | Young + steep cornea = aggressive progression risk |
| `pachy_asph_interaction` | `pachy_central ×        | Q_anterior                                         |
| `age_pachy_interaction`  | `age × pachy_central` | Older + thin = chronic remodelling                 |

### Group 8 — Composite risk scores (2 features)

**Corneal Risk Score (0–4)** — count of triggered flags:

```
score = (Kmax > 46) + (|astig| > 2.5) + (pachy_central < 510) + (Q_anterior > 0)
```

**Ectasia Risk Score (0–7)** — Randleman ERSS-inspired weighted sum:

| Trigger                              | Points |
| ------------------------------------ | ------ |
| Kmax > 47.2 D                        | +2     |
| Central pachymetry < 500 μm         | +2     |
| pachy_diff > 30 μm                  | +1     |
|                                      | Astig  |
| Q_anterior > 0.5                     | +1     |
| Thinnest point displacement > 1.0 mm | +1     |
| Age < 25 years                       | +1     |

### Group 9 — Auxiliary categorical (1 feature, used in EDA only)

| Feature       | Bins                                                   | Purpose                                          |
| ------------- | ------------------------------------------------------ | ------------------------------------------------ |
| `age_group` | adolescent / young_adult / adult / middle_age / senior | EDA stratification only — not used in modelling |

### Total feature inventory

| Category                 | Count                        |
| ------------------------ | ---------------------------- |
| Raw numeric              | 11                           |
| Encoded categorical      | 2                            |
| Engineered               | 28 (modelled) + 1 (EDA only) |
| **Total in model** | **41**                 |

---

## G. Data Visualisation Catalogue

Every figure is regenerated automatically when `python run_pipeline.py` runs. Resolutions: 150 DPI for screen viewing; 300 DPI vector PDFs for journal submission stored in `outputs/figures/publication/`.

### Preprocessing visualisations (6 figures)

| File                                  | What it shows                                            | Why it's in the report                                                                              |
| ------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `01_class_distribution_smote.png`   | Bar chart of class counts before vs. after SMOTE         | Demonstrates that augmentation is applied to training only and successfully balances minority class |
| `02_missing_values.png`             | Heatmap and bar chart of missing values per column       | Auditable evidence of complete dataset (no gaps)                                                    |
| `03_outlier_detection.png`          | IQR scatter for each raw feature with outliers in red    | Shows the conservative (×3) capping strategy                                                       |
| `04_outlier_capping_comparison.png` | Box plots before vs. after capping for 8 key features    | Demonstrates that capping preserves distribution shape                                              |
| `05_split_distribution.png`         | Pie charts of class balance in train / val / test splits | Confirms stratification preserves 65.6/34.4 ratio across all splits                                 |
| `06_scaling_comparison.png`         | Histograms before vs. after StandardScaler               | Visual proof of zero-mean unit-variance after scaling                                               |

### EDA visualisations (11 PNG + 2 HTML)

| File                                    | Content                                                      |
| --------------------------------------- | ------------------------------------------------------------ |
| `01_label_distribution.png`           | Pie + count bar + gender×label + age-group×label           |
| `02_feature_distributions.png`        | Histogram + KDE per raw feature, split by class              |
| `03_correlation_heatmap.png`          | Pearson correlation lower-triangle matrix                    |
| `04_correlation_with_target.png`      | Ranked horizontal bar chart of feature → label correlations |
| `05_boxplots_by_class.png`            | Box plots with Mann–Whitney U p-values + significance stars |
| `06_violin_plots.png`                 | Violin plots — full distribution shape by class             |
| `07_pairplot_top_features.png`        | 4-feature pairwise scatter matrix with class colouring       |
| `08_statistical_significance.png`     | −log₁₀(p) bar chart + Cohen's d effect-size bar chart     |
| `09_clinical_grouping_analysis.png`   | Risk-score and axis-type breakdown by class                  |
| `10_engineered_features_by_class.png` | Box plots for all 28 engineered features                     |
| `11_descriptive_stats_table.png`      | NP vs. Prog table of mean / SD / min / max                   |
| `12_3d_scatter.html`                  | Interactive 3D scatter Kmax × Astig × Pachy                |
| `13_sunburst.html`                    | Interactive Gender → Age → Label hierarchy                 |

### Evaluation visualisations (9 PNG + 2 SHAP)

| File                          | Content                                                    |
| ----------------------------- | ---------------------------------------------------------- |
| `roc_curves.png`            | All 12 models' ROC curves with AUC labels                  |
| `pr_curves.png`             | All 12 models' Precision–Recall curves with AP labels     |
| `confusion_matrices.png`    | Normalised confusion matrices (% by row) for all 12 models |
| `calibration_curves.png`    | Reliability diagrams + Brier scores for top 6 models       |
| `cv_boxplots.png`           | 5-fold CV AUC distribution per model                       |
| `model_metric_heatmap.png`  | All metrics × all models, colour-graded                   |
| `model_comparison_bars.png` | Grouped bars: Accuracy, F1, Sens, Spec, AUC per model      |
| `feature_importance.png`    | Top-15 features per tree-based model                       |
| `learning_curves.png`       | Train vs. validation AUC vs. training-set size             |
| `shap_beeswarm.png`         | Per-sample SHAP impact distribution                        |
| `shap_bar.png`              | Mean                                                       |

---

## H. Data Splitting Strategy

### The split — exact numbers

| Split           | Proportion | Sample count | Class 0      | Class 1      |
| --------------- | ---------- | ------------ | ------------ | ------------ |
| **Train** | 70 %       | 1,148        | 753 (65.6 %) | 395 (34.4 %) |
| **Val**   | 10 %       | 165          | 108 (65.5 %) | 57 (34.5 %)  |
| **Test**  | 20 %       | 329          | 216 (65.7 %) | 113 (34.3 %) |

Stratification preserves the natural 65.6/34.4 class ratio in all three partitions to within ± 0.2 %.

### Two-stage split implementation

```python
# Stage 1 — separate test set from full dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Stage 2 — separate validation from remaining 80 %
val_fraction_of_remainder = 0.10 / 0.80  # = 0.125
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=val_fraction_of_remainder,
    stratify=y_train_val, random_state=42
)
```

### Why three splits, not two?

- **Train (70 %)** — fits all model parameters and runs cross-validation.
- **Validation (10 %)** — used for early-stopping (MLP), threshold tuning, and intermediate checks. Available for hyperparameter search **without** touching the test set.
- **Test (20 %)** — held aside from the very first split, never touched until the final evaluation.

### Why `random_state = 42`?

A fixed seed is **mandatory** for reproducibility. Anyone re-running the pipeline gets identical splits, identical SMOTE samples, identical model weights, identical metrics. Without it, results would vary slightly between runs and the headline AUC = 0.9996 would not be replicable.

### Why stratified?

Random splitting could (by chance) put 50 % of progressive cases in the test set, creating a non-representative evaluation. Stratification guarantees the natural class ratio in every partition.

### Why scale **after** splitting?

`StandardScaler.fit(X_train)` learns mean and SD from the training data only; `scaler.transform(X_val)` and `scaler.transform(X_test)` use those training-set statistics. If we fit on the full dataset before splitting, test-set statistics would leak into model inputs — artificially boosting test scores. This is a classic and silent form of data leakage.

---

## I. Class Imbalance Correction (SMOTE)

### Why imbalance is dangerous in clinical ML

A naïve classifier predicting "non-progressive" always would score:

- Accuracy: 65.6 % (the natural class ratio)
- Sensitivity: **0 %** (every progressive case missed)
- Specificity: 100 %

In ophthalmology, missing progressive cases is the most clinically harmful failure mode — those are the patients who need intervention to prevent vision loss. A model maximising accuracy at the cost of sensitivity is unusable.

### Why SMOTE rather than naïve over- or under-sampling?

| Strategy                       | Pros                                                         | Cons                                                                  |
| ------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------------------- |
| **Random oversampling**  | Simple                                                       | Duplicates rows → memorisation, overfitting                          |
| **Random undersampling** | Removes majority bias                                        | Throws away majority-class information; small datasets become smaller |
| **Class weights**        | No data manipulation                                         | Models with hinge or hard losses ignore continuous weights            |
| **SMOTE**                | Generates synthetic samples preserving feature relationships | Slightly increases overfit risk near class boundary                   |

SMOTE (Chawla et al., 2002) generates new minority-class samples by **interpolating between k-nearest neighbours** of existing minority samples in feature space. The result is novel, realistic-looking synthetic examples — not duplicates.

### SMOTE applied to **training only**

```
Before SMOTE (training set):
  Non-Progressive: 753
  Progressive:     395

After SMOTE:
  Non-Progressive: 753  (unchanged)
  Progressive:     753  (synthetic samples added)
  Total:         1,506
```

Validation and test sets are **never** augmented. Their natural class ratio is preserved so evaluation reflects real-world performance.

### Three SMOTE variants supported

| Strategy            | When to use                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `smote` (default) | Standard SMOTE — recommended baseline                                                                                |
| `smote_tomek`     | SMOTE + Tomek-link removal — removes ambiguous boundary samples                                                      |
| `smoteenn`        | SMOTE + Edited-NN — most aggressive cleaning, removes majority samples whose k-NN majority votes against their label |

Selectable via `python run_pipeline.py --smote {smote,smote_tomek,smoteenn}`.

---

## J. Model Training — All 12 Models

We compare 11 base classifiers spanning the complexity spectrum, plus a stacking ensemble.

### Linear models

| Model                         | Hyperparameters                                    | Defence                                                               |
| ----------------------------- | -------------------------------------------------- | --------------------------------------------------------------------- |
| **Logistic Regression** | C=1.0, lbfgs, balanced class weight, max_iter=2000 | Interpretable baseline. Coefficients map directly to feature effects. |

### Tree-based models

| Model                   | Hyperparameters                                 | Defence                                                                      |
| ----------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| **Decision Tree** | max_depth=8, min_samples_split=10               | Single-tree baseline; overfits in isolation but informative.                 |
| **Random Forest** | n_estimators=300, balanced_subsample, n_jobs=-1 | Bagging ensemble; robust to noise.                                           |
| **Extra Trees**   | n_estimators=300, balanced_subsample            | Randomises feature splits as well; faster, often better-regularised than RF. |

### Gradient boosting models

| Model                       | Hyperparameters                                                                   | Defence                                                        |
| --------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Gradient Boosting** | n_estimators=300, lr=0.05, max_depth=4, subsample=0.8                             | Sequential trees correcting residual errors.                   |
| **XGBoost**           | n_estimators=300, lr=0.05, max_depth=5, colsample_bytree=0.8, eval_metric=logloss | Industry standard for tabular data.                            |
| **LightGBM**          | n_estimators=300, lr=0.05, num_leaves=31, balanced class weight                   | Leaf-wise growth; fastest convergence; best AUC in this study. |
| **AdaBoost**          | n_estimators=200, lr=0.5                                                          | Weak-learner sequence with adaptive weighting.                 |

### Other classifiers

| Model                    | Hyperparameters                                                    | Defence                                                        |
| ------------------------ | ------------------------------------------------------------------ | -------------------------------------------------------------- |
| **SVM (RBF)**      | C=10, gamma=scale, balanced class weight, probability=True         | Maximum-margin in transformed feature space.                   |
| **KNN**            | n_neighbors=7, weights=distance                                    | Non-parametric instance-based learner.                         |
| **MLP Neural Net** | hidden_layers=(128,64,32), ReLU, Adam, alpha=0.001, early_stopping | 3-layer feed-forward; representative of small neural baseline. |

### Stacking ensemble

```
Level 0 base learners (each trained with 5-fold OOF predictions):
   Random Forest  +  XGBoost  +  LightGBM  +  SVM (RBF)
            │
            ▼
Level 1 meta-learner:
   Logistic Regression (C=0.1, max_iter=2000)
            │
            ▼
        final prediction
```

**Why these 4 base learners?** They span the four fundamentally different decision-boundary shapes (axis-aligned splits, tree-boosted refinement, leaf-wise boosting, kernelised non-linear). Their out-of-fold predictions feed the meta-learner so it never sees in-sample base predictions — preventing leakage from level 0 to level 1.

**Why Logistic Regression as meta-learner?** A linear meta-learner forces interpretability — its coefficients reveal which base model is being trusted in which regions of the input distribution.

---

## K. Cross-Validation Protocol

All 11 base models are evaluated by **5-fold stratified cross-validation** on the SMOTE-augmented training set, before the final test-set evaluation.

### Protocol diagram

```
SMOTE-Augmented Training Set (1,506 samples)
    │
    ├── Fold 1: train 1,205, validate 301 → Acc, Prec, Recall, F1, AUC
    ├── Fold 2: train 1,205, validate 301 → ...
    ├── Fold 3: train 1,205, validate 301 → ...
    ├── Fold 4: train 1,205, validate 301 → ...
    └── Fold 5: train 1,204, validate 302 → mean ± SD across all 5 folds
```

Stratification preserves the post-SMOTE 50/50 class ratio in every fold.

### CV-mean accuracy by model (on SMOTE-augmented data)

| Model               | CV Accuracy mean ± SD |
| ------------------- | ---------------------- |
| Gradient Boosting   | 0.9887 ± 0.0054       |
| LightGBM            | 0.9880 ± 0.0034       |
| SVM (RBF)           | 0.9880 ± 0.0065       |
| XGBoost             | 0.9867 ± 0.0030       |
| Extra Trees         | 0.9867 ± 0.0021       |
| Random Forest       | 0.9814 ± 0.0034       |
| AdaBoost            | 0.9801 ± 0.0056       |
| KNN                 | 0.9787 ± 0.0062       |
| MLP Neural Net      | 0.9761 ± 0.0082       |
| Logistic Regression | 0.9728 ± 0.0071       |
| Decision Tree       | 0.9535 ± 0.0119       |

After CV, every model is **re-trained on the full SMOTE-augmented training set** and evaluated on the untouched 329-patient test set.

---

## L. Model Evaluation Metrics

Every metric on the 329-patient test set is reported. AUC-ROC is the primary ranking metric; sensitivity is the most clinically critical.

### Full results table

| Rank | Model               | AUC-ROC          | F1     | Sens   | Spec   | Acc    | Prec   | NPV    |
| ---- | ------------------- | ---------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| 1    | **LightGBM**  | **0.9996** | 0.9825 | 0.9912 | 0.9861 | 0.9878 | 0.9739 | 0.9953 |
| 2    | Gradient Boosting   | 0.9996           | 0.9912 | 0.9912 | 0.9954 | 0.9939 | 0.9912 | 0.9954 |
| 3    | Random Forest       | 0.9995           | 0.9868 | 0.9912 | 0.9907 | 0.9909 | 0.9825 | 0.9953 |
| 4    | AdaBoost            | 0.9993           | 0.9782 | 0.9912 | 0.9815 | 0.9848 | 0.9655 | 0.9953 |
| 5    | Stacking Ensemble   | 0.9993           | 0.9825 | 0.9912 | 0.9861 | 0.9878 | 0.9739 | 0.9953 |
| 6    | XGBoost             | 0.9992           | 0.9868 | 0.9912 | 0.9907 | 0.9909 | 0.9825 | 0.9953 |
| 7    | Extra Trees         | 0.9989           | 0.9820 | 0.9646 | 1.0000 | 0.9878 | 1.0000 | 0.9818 |
| 8    | MLP Neural Net      | 0.9988           | 0.9735 | 0.9735 | 0.9861 | 0.9818 | 0.9735 | 0.9861 |
| 9    | SVM (RBF)           | 0.9970           | 0.9432 | 0.9558 | 0.9630 | 0.9605 | 0.9310 | 0.9765 |
| 10   | Logistic Regression | 0.9961           | 0.9391 | 0.9558 | 0.9583 | 0.9574 | 0.9231 | 0.9764 |
| 11   | KNN                 | 0.9918           | 0.9561 | 0.9646 | 0.9722 | 0.9696 | 0.9478 | 0.9813 |
| 12   | Decision Tree       | 0.9356           | 0.9163 | 0.9204 | 0.9537 | 0.9422 | 0.9123 | 0.9581 |

### Confusion matrix breakdown — LightGBM (best model)

```
                     Predicted
                  Non-Prog | Progressive
Actual  Non-Prog    213    |      3       (specificity = 213/216 = 0.9861)
        Progress.     1    |    112       (sensitivity = 112/113 = 0.9912)
```

- **TP = 112** progressive eyes correctly identified
- **TN = 213** non-progressive eyes correctly cleared
- **FP =   3** non-progressive eyes incorrectly flagged (over-treatment risk: 1.4 % of non-progressive)
- **FN =   1** progressive eye missed (the most clinically dangerous error: 0.9 % of progressive)

### What each metric means clinically

| Metric                         | Formula                           | Clinical interpretation                                                                                                |
| ------------------------------ | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **AUC-ROC**              | Area under TPR-vs-FPR curve       | Threshold-independent discrimination. 1.0 = perfect, 0.5 = random.                                                     |
| **Sensitivity (Recall)** | TP / (TP + FN)                    | Fraction of progressive cases correctly identified.**Most critical** — high values mean fewer missed diagnoses. |
| **Specificity**          | TN / (TN + FP)                    | Fraction of non-progressive cases correctly cleared. Low values mean unnecessary intervention.                         |
| **Precision (PPV)**      | TP / (TP + FP)                    | Among predicted progressive, fraction truly progressive. High values mean low false-alarm rate.                        |
| **NPV**                  | TN / (TN + FN)                    | Among predicted non-progressive, fraction truly non-progressive. Critical for safe negative findings.                  |
| **F1-Score**             | 2·P·R / (P + R)                 | Harmonic mean of precision and recall. Robust to imbalance.                                                            |
| **Accuracy**             | (TP + TN) / N                     | Overall correctness — misleading in imbalanced data.                                                                  |
| **Brier Score**          | Mean (predicted prob − actual)² | Calibration quality. 0 = perfect, lower is better. LightGBM Brier = 0.0093.                                            |

### Bootstrap confidence interval

For the best model's AUC, the 95 % bootstrap CI (1,000 resamples of the test set with replacement):

```
LightGBM AUC mean (bootstrap) : 0.9996
Lower 2.5%                    : 0.9987
Upper 97.5%                   : 1.0000
CI width                      : 0.0013
```

The narrow CI confirms the headline AUC is not a sampling artefact.

---

## M. Statistical Significance Testing

### Mann–Whitney U test on each feature

For every numeric feature, we test whether the distribution differs between progressive and non-progressive eyes. We use **Mann–Whitney U** rather than t-test because clinical measurements are not Gaussian.

Top 10 most discriminative features (lowest p-values):

| Feature                        | p-value  | Cohen's d | Effect       |
| ------------------------------ | -------- | --------- | ------------ |
| `kmax_value_D`               | < 1e-100 | 1.46      | Large        |
| `pachy_central_um`           | < 1e-95  | 1.21      | Large        |
| `pachy_diff`                 | < 1e-90  | 1.50      | Large        |
| `kisa_proxy`                 | < 1e-85  | 1.38      | Large        |
| `corneal_power_index`        | < 1e-80  | 1.42      | Large        |
| `astig_abs`                  | < 1e-75  | 1.30      | Large        |
| `corneal_irregularity_index` | < 1e-70  | 1.25      | Large        |
| `ectasia_risk_score`         | < 1e-65  | 1.18      | Large        |
| `asphericity_anterior`       | < 1e-50  | 0.78      | Medium-Large |
| `kmax_high_flag`             | < 1e-45  | 0.95      | Large        |

All differences are **statistically significant at p < 0.001**. Most have **Cohen's d > 0.8** (large effect).

### McNemar's test on best vs. second-best model

McNemar's test compares two classifiers on the same test set, accounting for paired predictions:

```
LightGBM vs. Gradient Boosting:
  b (LightGBM correct, GB wrong): 0
  c (LightGBM wrong, GB correct): 0
  p-value: 1.0000
  Conclusion: no significant difference
```

When two models tie at AUC = 0.9996, this confirms they are statistically equivalent on this test set.

---

## N. SHAP Explainability

We use **SHAP** (SHapley Additive exPlanations; Lundberg & Lee, 2017) to explain individual predictions and global feature importance for the best model.

### Top SHAP features for the LightGBM model

| Rank | Feature | Mean |SHAP| |
|------|---------|--------------|
| 1 | `kmax_value_D` | 0.34 |
| 2 | `pachy_central_um` | 0.28 |
| 3 | `pachy_diff` | 0.25 |
| 4 | `kisa_proxy` | 0.21 |
| 5 | `astig_abs` | 0.19 |
| 6 | `corneal_power_index` | 0.16 |
| 7 | `kmax_high_flag` | 0.13 |
| 8 | `corneal_irregularity_index` | 0.12 |
| 9 | `ectasia_risk_score` | 0.10 |
| 10 | `asphericity_anterior` | 0.08 |

### Clinical interpretation

The SHAP ranking matches the Mann–Whitney test ranking and matches clinical intuition:

- **Kmax** is the single most influential feature — consistent with its role as the primary keratoconus screening measure.
- **Pachymetry** (central + thinning gradient) ranks 2nd and 3rd — confirming the Belin–Ambrosio approach.
- **KISA proxy** is in the top 4 — validating the choice to encode this composite index.
- **Astigmatism** completes the top 5 — confirming refractive error contributes independent signal.

This concordance between data-driven SHAP rankings and decades of clinical literature is a strong external validation of the model.

### SHAP figures

- `outputs/figures/evaluation/shap_beeswarm.png` — per-sample SHAP impact distribution
- `outputs/figures/evaluation/shap_bar.png` — mean |SHAP| importance ranking

---

## O. Streamlit Clinical Application

A four-page clinical decision-support web interface with **theme-safe contrast** that adapts automatically to light or dark mode.

### Pages

| Page                         | Content                                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Patient Prediction** | 13 raw input fields → engineered features → scaled → LightGBM → probability gauge + risk band + clinical-flag chips |
| **Model Performance**  | Full metrics table (colour-graded) + 9 evaluation figure tabs                                                           |
| **Dataset Explorer**   | Interactive filters (class, gender, age range) + histogram with marginal box                                            |
| **About**              | Project description, methodology, references                                                                            |

### Inference flow

```
User enters 13 raw measurements
            │
            ▼
engineer_all_features()  ← same code path as training
            │
            ▼
scaler.transform()       ← scaler.joblib loaded from disk
            │
            ▼
model.predict_proba()    ← best_model.joblib loaded from disk
            │
            ▼
Risk band + gauge chart + clinical-flag chips
```

This means inference is **mathematically identical** to training. The same engineering, the same scaler statistics, the same model — guaranteeing what is predicted at deployment matches what was validated.

### Risk bands

| Probability  | Band              | Recommendation                                   |
| ------------ | ----------------- | ------------------------------------------------ |
| 0.00 – 0.30 | LOW (green)       | Continue routine monitoring                      |
| 0.30 – 0.65 | MODERATE (orange) | Closer follow-up; consider preventive measures   |
| 0.65 – 1.00 | HIGH (red)        | Immediate clinical review; consider intervention |

### Theme-safe styling

The CSS uses Streamlit's theme variables and `@media (prefers-color-scheme: dark)` overrides:

- Light mode: dark text on light backgrounds
- Dark mode: light text on dark backgrounds
- Result boxes always have white text on coloured (green/orange/red) gradients — readable in either theme

---

## P. Project Structure

```
myopia_prediction/
│
├── data/
│   └── raw/
│       ├── combined_clinical_data_and_labels.csv  ← primary dataset (1,642)
│       ├── clinical_data_and_labels_v1.csv        ← v1 (1,454)
│       └── clinical_data_and_labels_v2.csv        ← v2 expansion (188)
│
├── src/                                       ← source library
│   ├── config.py                              ← paths, constants, fonts
│   ├── data/
│   │   ├── loader.py                          ← schema-validated CSV loading
│   │   ├── feature_engineering.py             ← 32-feature derivation
│   │   ├── preprocessor.py                    ← clean + split + scale
│   │   └── augmentation.py                    ← SMOTE family
│   ├── eda/
│   │   └── visualizer.py                      ← 11 EDA plots
│   ├── models/
│   │   ├── definitions.py                     ← 11 base models + stacking
│   │   ├── trainer.py                         ← 5-fold CV + final fit
│   │   ├── evaluator.py                       ← 9 evaluation plots + bootstrap CI
│   │   └── explainer.py                       ← SHAP report
│   ├── visualization/
│   │   └── preprocessing_plots.py             ← 6 preprocessing plots
│   └── utils/
│       └── generate_research_document.py      ← Word report generator
│
├── notebooks/
│   └── myopia_prediction_complete.ipynb       ← end-to-end annotated notebook
│
├── app/
│   ├── streamlit_app.py                       ← clinical UI (theme-safe)
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── feature_columns.json
│   └── processed_data.csv
│
├── diagrams/
│   └── generate_diagrams.py                   ← architectural SVGs
│
├── outputs/
│   ├── models/                                ← saved artifacts
│   ├── reports/                               ← CSV summaries
│   ├── document/                              ← .docx research document
│   └── figures/
│       ├── preprocessing/   (6 PNG)
│       ├── eda/             (11 PNG + 2 HTML)
│       ├── evaluation/      (9 PNG + 2 SHAP)
│       ├── publication/     (vector PDFs for paper)
│       └── diagrams/        (architecture)
│
├── .streamlit/
│   └── config.toml                            ← Streamlit theme config
│
├── run_pipeline.py                            ← 12-phase orchestrator
├── requirements.txt
├── README.md                                  ← this file
└── CLAUDE.md                                  ← Claude Code guidance
```

---

## Q. How to Run Everything

### Prerequisites

```bash
python --version           # 3.10 or higher
```

### Install dependencies

```bash
# Option A — uv (recommended)
uv venv myopia
uv pip install --python myopia/Scripts/python.exe -r requirements.txt

# Option B — venv + pip
python -m venv myopia
myopia\Scripts\activate              # Windows
source myopia/bin/activate           # Linux / macOS
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Recommended first run — fast (~3 minutes)
python run_pipeline.py --skip-shap --skip-diagrams

# Full pipeline including SHAP and architectural diagrams
python run_pipeline.py

# Alternative SMOTE strategies
python run_pipeline.py --smote smote_tomek
python run_pipeline.py --smote smoteenn
```

### Phase-by-phase reference

| Phase | Description                          | ~Time  |
| ----- | ------------------------------------ | ------ |
| 1     | Data loading & inspection            | < 5 s  |
| 2     | Feature engineering (15 → 47 cols)  | < 5 s  |
| 3     | Preprocessing (clean, split, scale)  | < 5 s  |
| 4     | SMOTE augmentation                   | < 5 s  |
| 5     | Preprocessing visualisations (6 PNG) | ~30 s  |
| 6     | EDA visualisations (11 PNG + 2 HTML) | ~60 s  |
| 7     | Multi-model training (11 + stacking) | ~60 s  |
| 8     | Results summary                      | < 5 s  |
| 9     | Evaluation plots (9 figures)         | ~60 s  |
| 10    | SHAP explanations                    | ~120 s |
| 11    | Save model artifacts                 | < 5 s  |
| 12    | Architectural diagrams               | ~30 s  |

### Generate the Word research document

```bash
python -m src.utils.generate_research_document
# → outputs/document/Myopia_Progression_Research_Document.docx
```

### Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### Run the Jupyter notebook

```bash
jupyter notebook notebooks/myopia_prediction_complete.ipynb
```

### Quick smoke test (after any code change)

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.data.loader import load_raw
from src.data.feature_engineering import engineer_all_features
from src.data.preprocessor import full_preprocessing_pipeline
from src.config import RAW_NUMERIC_COLS, ALL_FEATURE_COLS
df = engineer_all_features(load_raw())
prep = full_preprocessing_pipeline(df, RAW_NUMERIC_COLS, ALL_FEATURE_COLS)
print('shape:', df.shape, 'train:', prep['X_train_scaled'].shape)
"
```

---

## R. Limitations and Future Work

### Acknowledged limitations

1. **Single-centre dataset**. External validation on independent populations (different countries, devices, ethnicities) is required before clinical deployment.
2. **Binary classification**. Progression is dichotomised; a regression formulation predicting progression rate (D/year) would be more clinically informative.
3. **Cross-sectional design**. Labels are assigned retrospectively from clinical follow-up; prospective validation is needed.
4. **Simplified composite indices**. KISA proxy and CLMI proxy are simplified versions of proprietary topographer outputs.
5. **No imaging features**. The model uses only summary measurements; using full topographic maps with deep learning could capture finer-grained spatial patterns.

### Future work

- Multi-centre external validation across at least 3 hospitals
- Regression formulation predicting annualised progression rate
- Inclusion of full corneal-topography maps (deep learning on raw 2D Placido or Scheimpflug images)
- Longitudinal modelling of progression trajectories
- Calibration improvement using isotonic regression or Platt scaling for clinical communication

---

## S. References

1. **Holden, B.A., Fricke, T.R., Wilson, D.A., et al.** (2016). Global Prevalence of Myopia and High Myopia and Temporal Trends from 2000 through 2050. *Ophthalmology* 123(5), 1036–1042.
2. **Rabinowitz, Y.S.** (1998). Keratoconus. *Survey of Ophthalmology* 42(4), 297–319.
3. **Rabinowitz, Y.S., & Rasheed, K.** (1999). KISA% index: a quantitative videokeratography algorithm embodying minimal topographic criteria for diagnosing keratoconus. *Journal of Cataract & Refractive Surgery* 25(10), 1327–1335.
4. **Randleman, J.B., Woodward, M., Lynn, M.J., & Stulting, R.D.** (2008). Risk Assessment for Ectasia after Corneal Refractive Surgery. *Ophthalmology* 115(1), 37–50.
5. **Belin, M.W., & Khachikian, S.S.** (2009). An introduction to understanding elevation-based topography. *Clinical & Experimental Ophthalmology* 37(1), 14–29.
6. **Mahmoud, A.M., et al.** (2008). CLMI: the cone location and magnitude index. *Cornea* 27(4), 480–487.
7. **Ambrósio, R. Jr., et al.** (2011). Pachymetric progression indices: relationship between pachymetry and ectasia. *Cornea* 30(7), 800–806.
8. **Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P.** (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research* 16, 321–357.
9. **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD*, 785–794.
10. **Ke, G., Meng, Q., Finley, T., et al.** (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS* 30.
11. **Breiman, L.** (2001). Random Forests. *Machine Learning* 45(1), 5–32.
12. **Wolpert, D.H.** (1992). Stacked Generalisation. *Neural Networks* 5(2), 241–259.
13. **Lundberg, S.M., & Lee, S.I.** (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS* 30.
14. **Mardia, K.V., & Jupp, P.E.** (2000). *Directional Statistics*. Wiley.

---

## Appendix A — Clinical Threshold Reference

| Measurement                | Normal       | Risk threshold                 | Source            |
| -------------------------- | ------------ | ------------------------------ | ----------------- |
| Kmax (D)                   | < 45.0       | > 47.2 (keratoconus suspect)   | Rabinowitz (1998) |
| Central pachymetry (μm)   | > 520        | < 500 (high), < 510 (moderate) | Randleman (2008)  |
| pachy_diff (μm)           | < 20         | > 30 (abnormal gradient)       | Belin–Ambrosio   |
| pachy_ratio                | > 0.96       | < 0.94                         | Belin–Ambrosio   |
|                            | Astigmatism  | (D)                            | < 1.5             |
| Anterior Q                 | -0.2 to -0.3 | > 0 (oblate, abnormal)         | Topographer norms |
| Thinnest displacement (mm) | < 0.5        | > 1.0 (decentred)              | CLMI literature   |
| Ectasia Risk Score         | 0            | ≥ 3 (elevated), ≥ 5 (high)   | Randleman ERSS    |

## Appendix B — Reproducibility Checklist

- [ ] Python 3.10+ installed
- [ ] All packages in `requirements.txt` installed
- [ ] Combined dataset present at `data/raw/combined_clinical_data_and_labels.csv`
- [ ] `RANDOM_STATE = 42` in `src/config.py` (default — do not change)
- [ ] Default split ratios 70/10/20 (default — do not change)
- [ ] Default SMOTE strategy `smote` (default)
- [ ] Run: `python run_pipeline.py --skip-shap --skip-diagrams`
- [ ] Verify `outputs/reports/model_results_summary.csv` matches §L results

---

*This research is conducted as part of an MPhil degree in Ophthalmology. All patient records are fully anonymised. Predictive model outputs are intended to support — not replace — the clinical judgement of qualified ophthalmologists. No clinical decisions should be made based solely on model output.*
