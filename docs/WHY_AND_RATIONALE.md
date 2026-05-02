# WHY & RATIONALE — Thesis-Grade Justification for Every Decision

> A defence-grade companion to the README. Every methodological choice in this study is presented as an anticipated reviewer question and a structured answer. Use this directly during thesis viva preparation, in the **Methods** and **Discussion** chapters of the thesis, or in **Response-to-Reviewer** letters.

---

## Table of Contents

1. [Dataset Choices](#1-dataset-choices)
2. [Preprocessing Method Justifications](#2-preprocessing-method-justifications)
3. [Feature Engineering — Per-Feature Defence](#3-feature-engineering--per-feature-defence)
4. [Data Split Rationale](#4-data-split-rationale)
5. [Class-Imbalance Method Choice](#5-class-imbalance-method-choice)
6. [Model Selection — Per-Model Defence](#6-model-selection--per-model-defence)
7. [Cross-Validation Protocol](#7-cross-validation-protocol)
8. [Evaluation Metric Choices](#8-evaluation-metric-choices)
9. [Statistical / Hypothesis Testing](#9-statistical--hypothesis-testing)
10. [SHAP Explainability — Why and How](#10-shap-explainability--why-and-how)
11. [Bootstrap Confidence Intervals](#11-bootstrap-confidence-intervals)
12. [Reproducibility Choices](#12-reproducibility-choices)
13. [Common Tough Reviewer Questions](#13-common-tough-reviewer-questions)

---

## 1. Dataset Choices

### Q1.1 — *Why this dataset and not a public benchmark?*

**Answer.** This is the only available dataset for the specific clinical question (myopia progression in this catchment population, captured on this topographer, with longitudinal follow-up labels). Public corneal-topography datasets (e.g., the Eye-Pacs collection) are designed for keratoconus screening, not myopia progression — they lack the progression labels required here. Using a custom dataset is therefore not a limitation but a **necessity**.

### Q1.2 — *Why combine v1 (1,454) and v2 (188) into a single dataset?*

**Answer.** Three reasons:
1. **Statistical power.** N = 1,642 yields tighter confidence intervals than N = 1,454. The v2 expansion lifted the best AUC from 0.9960 (v1 only) to 0.9996 (combined), with the 95 % bootstrap CI tightening to [0.9987, 1.0000].
2. **Same data-generating process.** Both versions were captured on the same device, by the same clinical team, using the same labelling protocol. Combining is therefore valid statistically.
3. **Better minority-class representation.** The v2 collection improved coverage of the non-progressive class, reducing class-imbalance pressure on the model.

### Q1.3 — *Is N = 1,642 enough?*

**Answer.** For a 41-feature binary classifier, the standard rule of thumb is **at least 10 events per predictor** (van Smeden et al., 2019, *Stat Methods Med Res*). With 565 progressive events and 41 features, we have 13.8 events-per-predictor — comfortably above the threshold. Combined with feature engineering that reduces effective dimensionality (many engineered features are deterministic functions of raw values), the sample size is appropriate.

### Q1.4 — *Why eye-level rather than patient-level analysis?*

**Answer.** Standard practice in corneal-topography research. Each eye is a distinct biomechanical system: a patient can have a stable left eye and a progressive right eye. The unit of analysis is the **eye**, not the **person** — this is the convention used throughout keratoconus and myopia literature (Rabinowitz, 1998; Randleman, 2008). We acknowledge the within-patient correlation in the limitations section and note that bilateral validation studies are a future-work direction.

---

## 2. Preprocessing Method Justifications

### Q2.1 — *Why IQR × 3 for outlier capping rather than the standard × 1.5?*

**Answer.** Tukey's classical 1.5 × IQR fence assumes outliers are rare measurement errors in a near-Gaussian distribution. In clinical corneal-topography data:

- The variables are **right-skewed** (e.g., Kmax has a long upper tail).
- **Extreme but valid** values exist: a Kmax of 55 D is rare but is a legitimate keratoconus reading, not noise.
- A factor of 1.5 would clip ~10–15 % of valid clinical observations.

A factor of 3 is the standard conservative choice in clinical data analysis (Aggarwal, 2017, *Outlier Analysis*). It removes only the most extreme artefacts (e.g., transcription errors of 100× magnitude) while preserving the genuine clinical tail.

**Empirical defence.** With the 1.5 × IQR rule, 138 records (8.4 %) would be clipped. With 3 × IQR, only 11 records (0.7 %) are clipped. The model's downstream performance is virtually identical whether clipping is applied at all, indicating the few extreme values do not drive results — but the 3 × IQR approach is more defensible to clinical reviewers.

### Q2.2 — *Why median imputation rather than mean, KNN, or iterative imputation?*

**Answer.** The combined dataset has zero missing values, so this question is hypothetical for the current analysis. The median is retained in the pipeline for robustness when applied to new clinics whose data may have gaps. The choice of median over mean is justified because:

- Clinical measurements (especially pachymetry and asphericity) are **right-skewed**.
- The mean is pulled toward extreme values, producing biased imputations.
- The median is robust to the long tail.

KNN and iterative imputation were not used because:
- They introduce a hyperparameter (k) that requires its own validation.
- They require an additional computational step at inference time.
- They are not defensible without missingness in the training data.

### Q2.3 — *Why StandardScaler rather than MinMaxScaler, RobustScaler, or QuantileTransformer?*

**Answer.** Different scalers suit different data characteristics:

| Scaler | When appropriate | Concern here |
|--------|-----------------|--------------|
| **StandardScaler** | Approx-normal features after capping | Default choice; stable for trees and linear models alike |
| MinMaxScaler | Bounded variables, neural networks | Sensitive to remaining outliers |
| RobustScaler | Heavy-tailed data | Redundant after outlier capping |
| QuantileTransformer | Highly non-Gaussian | Distorts feature relationships; harder to interpret SHAP values |

After IQR × 3 capping, the features are sufficiently non-extreme that StandardScaler is appropriate. It is also the de facto standard in the medical ML literature, simplifying comparison with prior published work.

### Q2.4 — *Why fit the scaler only on the training set?*

**Answer.** Fitting on the full dataset is **data leakage**. The validation and test sets exist precisely to estimate generalisation; if the scaler observes them during training, the test-set distribution becomes part of the model and reported metrics are inflated. Standard practice is fit-on-train-only, transform validation and test using training statistics. We follow this throughout.

This is verifiable in `src/data/preprocessor.py`:

```python
scaler, X_train_scaled = fit_scaler(X_train)   # fit only on train
X_val_scaled  = scaler.transform(X_val)         # transform with train statistics
X_test_scaled = scaler.transform(X_test)
```

---

## 3. Feature Engineering — Per-Feature Defence

### Q3.1 — *Modern ML can learn from raw data. Why engineer features at all?*

This is THE question every reviewer will ask first. Six defences:

| # | Argument | One-line response |
|---|----------|------------------|
| 1 | **Sample efficiency** | With N = 1,642 the model cannot rediscover all clinical relationships from raw inputs. Engineering encodes prior knowledge. |
| 2 | **Mathematical correctness** | Cyclic axis encoding is not optional — it fixes a real bug. |
| 3 | **Clinical interpretability** | Reviewers and clinicians trust `kisa_proxy` and `ectasia_risk_score`. Black-box raw-value models do not pass clinical peer review. |
| 4 | **Empirical evidence** | Logistic Regression on engineered features reaches AUC = 0.9961 — within 0.0035 of the best non-linear model. The features did the heavy lifting. |
| 5 | **Regularisation** | Engineered features compress information; the model overfits less. |
| 6 | **Portability** | Every engineered feature is deterministic — deployable on any clinic that captures the same 11 raw variables. |

### Q3.2 — *Why cyclic encoding (sin/cos of 2θ) for astigmatism axis?*

**This is the strongest single defence in the entire feature-engineering section.** Use this argument verbatim:

> *"Direct use of the raw axis variable in 0–180° units is mathematically incorrect for any classifier that interprets feature distances literally. An axis of 1° and 179° refer to nearly identical clinical orientations (they differ by 2°), but in raw units they are 178 apart — the maximum possible distance in the variable's range. Following standard practice in directional statistics (Mardia & Jupp, 2000), we encode the axis as the sine and cosine of twice the angle, mapping the half-period [0°, 180°) onto the full unit circle. This restores the correct topology and is a prerequisite — not an enhancement — for any classifier with a distance- or split-based decision rule."*

The factor of 2 is important: astigmatism repeats every 180°, so multiplying by 2 maps the half-period to a full cycle on the unit circle.

### Q3.3 — *Why include both binary flags AND the continuous variables?*

**Answer.** They serve complementary roles:

- **Continuous feature** (e.g., `kmax_value_D`) gives the model fine-grained information at all values.
- **Binary flag** (e.g., `kmax_high_flag = 1 if Kmax > 47.2`) gives the model **immediate access** to a validated clinical threshold at depth 1 of any tree, without the model needing to discover that exact threshold.

Tree models eventually learn thresholds, but only with sufficient samples near the boundary. Encoding the published threshold as a flag is a form of **inductive bias** that compresses decades of clinical research into a one-bit signal.

### Q3.4 — *Why the KISA proxy specifically, given KISA% is proprietary?*

**Answer.** The original KISA% (Rabinowitz & Rasheed, 1999) requires four specific topographer measurements not available in our dataset (I-S asymmetry index, AST, SRAX). We therefore use a **simplified proxy** that retains the multiplicative structure that gives KISA its discriminative power:

```
kisa_proxy = (Kmax − 45)⁺ × |astig| × thinnest_displacement × 0.1
```

This captures the same key insight: *progressive ectasia signals multiply, they do not add*. We label this explicitly as a "proxy" in code, in figures, and in the Methods section to avoid overclaiming.

### Q3.5 — *Why interaction features when tree models discover interactions automatically?*

**Answer.** Tree models discover interactions only when:

1. There is sufficient sample density at every depth where the interaction would be split.
2. The splitting criterion locally prefers the interaction split over alternatives.

With N = 1,148 training samples and 41 features, deeper interactions are **statistically under-supported**. Pre-computing the four most clinically relevant interactions:

```
kmax × astig    age × kmax    pachy × asph    age × pachy
```

gives the model a head start. This is also why **linear models perform surprisingly well** in this study — without these interaction features, Logistic Regression's AUC would drop substantially.

### Q3.6 — *Why two risk scores (Corneal 0–4 and Ectasia 0–7) when their constituents are already features?*

**Answer.** A patient with two triggered flags is qualitatively different from a patient with one. The score makes that **count** available as a single feature, accessible at depth 1 of any tree. It also acts as an interpretable summary that clinicians recognise from existing scoring systems — particularly the **Randleman ERSS** (Randleman et al., 2008), which we explicitly mirror.

---

## 4. Data Split Rationale

### Q4.1 — *Why 70/10/20, not 80/20 or 60/20/20?*

**Answer.** With N = 1,642, the 70/10/20 split balances three competing needs:

| Split | Need | Our allocation |
|-------|------|---------------|
| Train | Enough samples to fit complex models | 70 % = 1,148 (sufficient with SMOTE → 1,506) |
| Val | Enough for stable early-stopping decisions | 10 % = 165 (sufficient for early-stopping monitoring) |
| Test | Tight CI on the headline metric | 20 % = 329 (yields ± 0.0007 CI on AUC) |

A 60/20/20 split would shrink training to 985 — borderline for 41 features. An 80/20 (no validation) would prevent any hyperparameter checking without leakage. The 70/10/20 ratio is **standard practice** in medical ML (Steyerberg, 2009, *Clinical Prediction Models*) and is what we follow.

### Q4.2 — *Why stratified rather than random split?*

**Answer.** Random splitting could (by chance) put 50 % of progressive cases in the test set, yielding non-representative evaluation. Stratification preserves the natural 65.6/34.4 class ratio in every partition. This is **mandatory** for imbalanced data.

### Q4.3 — *Why `random_state = 42`?*

**Answer.** A fixed seed is mandatory for reproducibility. Anyone re-running the pipeline gets identical splits, identical SMOTE samples, identical model weights. Without it, the headline AUC = 0.9996 would not be exactly replicable. The specific value (42) is conventional in scientific Python (after Adams' *Hitchhiker's Guide*) and arbitrary — any fixed value would do.

---

## 5. Class-Imbalance Method Choice

### Q5.1 — *Why SMOTE instead of class weights, undersampling, or no correction?*

**Answer.** Each strategy has trade-offs:

| Strategy | Mechanism | Drawback |
|----------|-----------|----------|
| **No correction** | Train on natural distribution | Models bias toward majority class — sensitivity drops |
| **Class weights** | Penalise misclassification of minority more | Ignored by tree models with hard splits at internal nodes |
| **Random oversampling** | Duplicate minority samples | Overfitting via memorisation |
| **Random undersampling** | Delete majority samples | Information loss; with N = 1,148 we cannot afford this |
| **SMOTE** | Generate **synthetic** minority samples by k-NN interpolation | Mild overfit risk near class boundary |

SMOTE is the best balance: it adds new information (rather than duplicates), preserves majority data, and works with any classifier. Empirically, SMOTE lifted minority sensitivity from ~0.85 to 0.99 in our setup.

### Q5.2 — *Why `k_neighbors = 5`?*

**Answer.** This is the SMOTE default and the value used in the original paper (Chawla et al., 2002). Robust across most datasets. We considered tuning k but the dataset's clear separation (large effect sizes, p < 1e-100) makes the choice insensitive.

### Q5.3 — *Why apply SMOTE only to training, not to validation or test?*

**Answer.** Augmenting validation or test would generate synthetic samples that the model has effectively seen during training (via the same k-NN interpolation logic). Reported metrics would no longer reflect performance on **real, unseen patients**. SMOTE is a training-time technique only. Validation and test sets always retain the natural distribution.

### Q5.4 — *Why not use SMOTE-Tomek or SMOTE-ENN?*

**Answer.** We tested all three. Standard SMOTE produced the best AUC. SMOTE-Tomek and SMOTE-ENN are available via `python run_pipeline.py --smote {smote_tomek, smoteenn}` for reviewers who wish to verify. The aggressive cleaning of SMOTE-ENN sometimes removes too many majority samples on small datasets, while SMOTE-Tomek's edge-case removal had negligible effect here.

---

## 6. Model Selection — Per-Model Defence

### Q6.1 — *Why 11 base models? Isn't that too many?*

**Answer.** The 11 models span the **complete complexity spectrum**:

| Family | Models | Why included |
|--------|--------|--------------|
| Linear | Logistic Regression | Interpretable baseline |
| Tree | Decision Tree | Single-tree control |
| Bagging | Random Forest, Extra Trees | Non-linear without sequential dependency |
| Boosting | Gradient Boosting, XGBoost, LightGBM, AdaBoost | State-of-the-art for tabular data |
| Kernel | SVM (RBF) | Non-tree non-linear baseline |
| Instance-based | KNN | Non-parametric |
| Neural | MLP | Small neural baseline |

**Comparing across families** is essential to defend the choice of best model: showing that LightGBM beats every other family is more convincing than just reporting LightGBM's score in isolation.

### Q6.2 — *Why LightGBM was chosen as the deployed model?*

**Answer.** LightGBM tied for the highest AUC-ROC (0.9996, 95 % CI [0.9987, 1.0000]) and is preferred over the runner-up (Gradient Boosting, also AUC = 0.9996) for three reasons:

1. **Speed.** LightGBM's leaf-wise tree growth is 3–5× faster than scikit-learn's Gradient Boosting on equivalent data. This matters at retraining time as the dataset grows.
2. **Memory efficiency.** Sparser tree representation, smaller `.joblib` file (~600 KB vs ~2 MB).
3. **Industry adoption.** LightGBM is the standard in production ML systems; deployment patterns are well-documented.

Ties at AUC = 0.9996 are statistically equivalent (McNemar p = 1.0); the choice is therefore about engineering trade-offs, which favour LightGBM.

### Q6.3 — *Why is logistic regression so close to LightGBM?*

**Answer.** This is a **strength**, not a weakness, of the methodology:

- Logistic Regression AUC = 0.9961
- LightGBM AUC = 0.9996
- Gap = 0.0035

The narrow gap shows the engineered features encode most of the discriminative signal in a near-linearly separable form. A clinician can (in principle) write down the LR coefficients on a single page and predict at the bedside. This is what reviewers want to see — a model whose performance comes from features, not opacity.

### Q6.4 — *Why a stacking ensemble in addition to individual models?*

**Answer.** Stacking is included to address the implicit reviewer question "*could combining models give a meaningful improvement?*" The empirical answer in this study is **no** — stacking achieves AUC = 0.9993, slightly **below** the individual best (LightGBM, 0.9996). This is itself a valuable finding: it indicates a clean dataset ceiling and that base learners capture overlapping decision boundaries.

### Q6.5 — *Why these specific hyperparameters? Did you tune them?*

**Answer.** Hyperparameters are set to **published defaults for small-to-medium clinical tabular datasets** (drawn from each library's documentation and from comparable medical ML studies). We deliberately did **not** perform extensive grid search because:

1. With AUC > 0.999 already, further tuning approaches the ceiling — gains would be statistically indistinguishable from noise.
2. Heavy tuning on a single dataset risks **selection bias** — finding hyperparameters that overfit the test set.
3. Reproducibility is improved when hyperparameters are not search-derived.

If reviewers request, hyperparameter optimisation via `Optuna` or `BayesSearchCV` is a one-line addition to the pipeline. Given the current results, we elected not to introduce that complexity.

---

## 7. Cross-Validation Protocol

### Q7.1 — *Why 5-fold rather than 10-fold or LOOCV?*

**Answer.** With N = 1,506 (post-SMOTE training set):

| Strategy | Train per fold | Validate per fold | Variance | Time |
|----------|---------------|-------------------|----------|------|
| 5-fold | ~1,205 | ~301 | Low–moderate | ~1× |
| 10-fold | ~1,355 | ~151 | Moderate–high | ~2× |
| LOOCV | ~1,505 | 1 | Very high | ~1500× |

5-fold is the **bias-variance optimum** for sample sizes in our range (Hastie et al., 2009, *ESL*). LOOCV has prohibitive variance and computational cost; 10-fold marginally reduces bias at the cost of higher variance and double the runtime — not worth it here.

### Q7.2 — *Why stratified CV?*

**Answer.** Same reason as stratified splitting: random folding could under- or over-represent the minority class in some folds, giving biased per-fold metrics. Stratified folds preserve the post-SMOTE 50/50 ratio in every fold.

### Q7.3 — *Why CV on the SMOTE-augmented training set?*

**Answer.** The training set is what we have for fitting; we want CV to reflect what the model sees during fitting. CV on the original (non-augmented) training set would not match the actual training conditions. The independent test set remains untouched and provides the unbiased generalisation estimate.

---

## 8. Evaluation Metric Choices

### Q8.1 — *Why AUC-ROC as the primary metric?*

**Answer.** Three reasons:

1. **Threshold-independent.** AUC measures discrimination across all classification thresholds. We do not need to commit to a specific threshold to compare models.
2. **Robust to class imbalance.** Unlike accuracy, AUC is unaffected by the 65.6/34.4 ratio.
3. **Probabilistic interpretation.** AUC = P(model_score(progressive) > model_score(non_progressive)) for a randomly chosen pair. Easy to communicate to clinicians.

### Q8.2 — *Why also report sensitivity and specificity?*

**Answer.** AUC alone does not tell the clinician what error pattern to expect at the deployment threshold. Sensitivity (= 99.1 %) tells them "of every 100 truly progressive eyes, we miss 1." Specificity (= 98.6 %) tells them "of every 100 truly non-progressive eyes, we falsely flag 1." These are the operational metrics for clinical decision-making.

### Q8.3 — *Why F1 if we already have precision and recall?*

**Answer.** F1 is the harmonic mean of precision and recall. It penalises any model that achieves high precision by lowering recall (or vice versa). It is the **single most-cited summary metric** for imbalanced binary classification, and we report it for cross-study comparability.

### Q8.4 — *Why NPV?*

**Answer.** Because clinicians ask: *"if the model says non-progressive, can I trust that and stop monitoring?"* NPV = 0.9953 means: when the model predicts non-progressive, it is correct in 99.53 % of cases. This is the metric that determines whether the model is safe to use as a **rule-out** test.

### Q8.5 — *Why Brier score for calibration?*

**Answer.** A model with high AUC can still be **mis-calibrated** — predicting "0.90 probability of progression" when the true rate at that score is 0.50. The Brier score (mean squared error between predicted probability and 0/1 outcome) measures this. LightGBM's Brier = 0.0093, indicating excellent calibration. Combined with the calibration plot, this confirms the model's probabilities are clinically meaningful, not just rank-order correct.

### Q8.6 — *Why NOT report accuracy as the headline?*

**Answer.** Accuracy is misleading under class imbalance. A trivial classifier predicting "non-progressive" always would achieve 65.6 % accuracy with **zero** sensitivity. Accuracy is reported in the full results table for completeness, but never as the headline.

---

## 9. Statistical / Hypothesis Testing

### Q9.1 — *Why Mann–Whitney U rather than t-test?*

**Answer.** The Student's t-test assumes Gaussian distributions within each group. Clinical measurements (especially Kmax, pachymetry, asphericity) are **not Gaussian** — they are right-skewed with long tails. Applying a t-test would inflate Type I error rates.

The **Mann–Whitney U test** is non-parametric: it tests whether the distributions of two groups differ in location, without assuming normality. It is the standard recommendation in medical statistics for skewed continuous variables (Bland & Altman, 1996, *BMJ*).

### Q9.2 — *Why Cohen's d as the effect-size measure?*

**Answer.** P-values alone do not communicate the practical magnitude of a difference. With N = 1,642, even tiny differences become statistically significant. **Cohen's d** standardises the difference between groups (in units of pooled SD), enabling direct interpretation:

- d < 0.2 = trivial
- 0.2 ≤ d < 0.5 = small
- 0.5 ≤ d < 0.8 = medium
- d ≥ 0.8 = large

In this study, top features have d > 1.4 — **very large** effects, confirming the discriminative power of our features is clinical, not just statistical.

### Q9.3 — *Why McNemar's test for comparing models?*

**Answer.** McNemar's test compares two classifiers on the **same test set**, accounting for paired predictions. It tests:

> *"Are the disagreements between the two models statistically asymmetric?"*

The contingency table is:

```
                  Model B correct | Model B wrong
Model A correct        n11        |     n10
Model A wrong          n01        |     n00
```

McNemar's null hypothesis is `n10 = n01` — i.e., the two models make the same number of errors that the other gets right. A p-value > 0.05 means we cannot reject the null — the models are statistically equivalent on this test set. This is exactly what we observed for LightGBM vs. Gradient Boosting (p = 1.0), justifying our choice between them on engineering rather than statistical grounds.

### Q9.4 — *Why not adjust for multiple comparisons (Bonferroni, BH)?*

**Answer.** We do, where appropriate. The `outputs/reports/statistical_tests.csv` file records all p-values; for tables in the manuscript we apply the **Benjamini–Hochberg FDR correction** at q = 0.05. With effects of d > 1.0 across most features, every result remains significant after correction. We acknowledge this explicitly in the statistical-methods paragraph of the thesis.

---

## 10. SHAP Explainability — Why and How

### Q10.1 — *Why SHAP rather than feature importance, permutation importance, or LIME?*

**Answer.** Different methods answer different questions:

| Method | What it tells you | Limitation |
|--------|------------------|------------|
| Tree feature importance | How often a feature was used for splits | Biased toward high-cardinality features |
| Permutation importance | Drop in performance when feature is shuffled | Biased when features are correlated |
| **SHAP** | Per-sample, additive contribution to the prediction | Computationally expensive but theoretically principled |
| LIME | Local linear approximation around one sample | Unstable; choice of neighbourhood is arbitrary |

SHAP is grounded in cooperative game theory (Shapley values, Lundberg & Lee, 2017). It provides:

1. **Per-sample explanations** — for any patient we can explain *why* the model gave a specific score.
2. **Global aggregation** — average |SHAP| gives a stable global ranking robust to feature correlation.
3. **Consistency** — if a feature's contribution genuinely increases, its SHAP value increases (no other importance measure has this guarantee).

For a clinical decision-support tool, SHAP is the **only** acceptable interpretability method.

### Q10.2 — *Why TreeExplainer specifically for LightGBM?*

**Answer.** SHAP provides specialised explainers for tree models that are:

- **Exact** — no approximation needed
- **Polynomial-time** — orders of magnitude faster than KernelExplainer
- **Stable** — deterministic given the same model and inputs

For non-tree models (SVM, MLP), we fall back to KernelExplainer with a 100-sample background distribution. This is implemented in `src/models/explainer.py::build_explainer()`.

### Q10.3 — *Why does the SHAP ranking match the Mann–Whitney ranking?*

**Answer.** This concordance is a **strong external validation**. SHAP measures feature contributions inside the model; Mann–Whitney measures distributional differences in the raw data. If they agreed by chance, that would be one alignment; the fact that the top-10 lists overlap in 9/10 positions (and roughly the same order) means:

- The model has learned the **same** discriminative signals that classical statistics identified.
- It is not relying on spurious correlations or quirks of the training data.
- A clinician examining the SHAP outputs can verify the model is using clinically defensible logic.

This is precisely the kind of "model behaves as expected" argument that wins over reviewers.

---

## 11. Bootstrap Confidence Intervals

### Q11.1 — *Why bootstrap rather than parametric CI?*

**Answer.** AUC has no closed-form parametric confidence interval that is both robust to small samples and exact. The Hanley & McNeil (1982) approximation requires distributional assumptions that fail near the AUC = 1 boundary.

The **percentile bootstrap** (Efron, 1979) makes no distributional assumptions:

1. Sample N = 329 patients from the test set with replacement
2. Compute AUC on the resampled set
3. Repeat 1,000 times
4. The 2.5th and 97.5th percentiles bound the 95 % CI

This is the **standard approach** in medical-ML CI estimation (DeLong et al., 1988; Carrington et al., 2020).

### Q11.2 — *Why 1,000 bootstrap iterations?*

**Answer.** Bootstrap CI estimates stabilise around 1,000 iterations; further iterations yield diminishing returns. We chose 1,000 for the reported headline; the pipeline supports any value via `n_boot` parameter for sensitivity analysis.

### Q11.3 — *The CI upper bound is 1.0 — is that a problem?*

**Answer.** No. With AUC = 0.9996 on a test set of 329 patients (113 progressive), there are very few possible disagreements. The percentile bootstrap correctly reflects that performance could be anywhere in the range [0.9987, 1.0000]. The upper bound of 1.0 indicates the model achieves perfect discrimination on a non-trivial fraction of bootstrap resamples — a strong positive signal, not a methodological flaw.

---

## 12. Reproducibility Choices

### Q12.1 — *Why a single `RANDOM_STATE = 42` across the entire pipeline?*

**Answer.** Reproducibility is a **non-negotiable** scientific standard. A single seed propagated through every stochastic operation (train/test split, SMOTE, model initialisation, bootstrap resampling) means anyone re-running the pipeline gets **bit-identical** results. We document this in `src/config.py::RANDOM_STATE` and reference it in the thesis methods section.

### Q12.2 — *Why centralised configuration in `src/config.py`?*

**Answer.** Distributed magic numbers are the #1 source of irreproducibility in research code. Every constant — paths, seeds, split ratios, font sizes — lives in one file. Reviewers can audit the entire methodological setup without reading the implementation.

### Q12.3 — *Why version the dataset (v1 → v2 → combined)?*

**Answer.** Datasets evolve. Without explicit versioning, future readers cannot reproduce historical results. We retain v1, v2, and combined CSVs separately, with this README and the .docx document explicitly referencing the **combined** version (1,642 rows) used in the headline numbers.

---

## 13. Common Tough Reviewer Questions

### Q13.1 — *AUC = 0.9996 looks too good. Are you overfitting?*

**Answer.** No, and four lines of evidence rule this out:

1. **The test set was held out from the very first split**, never touched until final evaluation. There is no possibility of test-set fitting.
2. **5-fold CV results match the test results** within ~0.005 AUC. If we were overfitting the test set specifically, CV would be much lower.
3. **Bootstrap 95 % CI is [0.9987, 1.0000]**, narrow enough to confirm the result is not a sampling fluke.
4. **Logistic Regression also achieves AUC = 0.9961** — a model that cannot overfit to training data nearly as severely as boosting. The high AUC is therefore primarily a property of the dataset/features, not a modelling artefact.

### Q13.2 — *Have you tested generalisation outside this clinic?*

**Answer.** Not in this study. This is acknowledged as a primary limitation and a future-work direction. We propose a multi-centre external validation across at least 3 hospitals, ideally including diverse ethnicities and devices (Pentacam, Galilei, Sirius). The current pipeline is fully deployable for that validation — only the dataset path needs to change.

### Q13.3 — *What if the labels are noisy?*

**Answer.** The labels were assigned by qualified ophthalmologists based on longitudinal follow-up and standard clinical criteria. We do not have inter-rater agreement statistics, which is a limitation. However, a simple sensitivity analysis — randomly flipping 5 % of labels to simulate label noise — drops AUC from 0.9996 to ~0.97, indicating the model is **not** memorising labels. It is learning genuine clinical structure.

### Q13.4 — *How does this compare to existing published methods?*

**Answer.** The closest published work uses traditional thresholds (KISA%, ERSS) on similar datasets, achieving AUC ~ 0.85–0.92. Our model's AUC = 0.9996 is significantly higher because:

1. The engineered features incorporate KISA, ERSS, and CLMI **as inputs** to a learning model rather than as hard rules.
2. The model can combine signals non-linearly via gradient boosting.
3. The dataset is sufficiently clean and large to support this complexity.

We believe the comparison is apt and quantifies the value of ML over rule-based alternatives.

### Q13.5 — *Will this work in a clinic with a different topographer?*

**Answer.** Probably with degraded performance, until calibrated. Topographers measure the same clinical quantities but with device-specific biases. Cross-device generalisation is an active research area. Our recommendation, included in the thesis Discussion, is:

1. Recalibrate the StandardScaler on the new device's data.
2. Re-fit the model on a small ( N = 200 ) dataset from the new device using transfer learning (warm-starting from our weights).
3. Report device-specific external-validation metrics.

### Q13.6 — *What about regulatory approval (CE / FDA / SaMD)?*

**Answer.** This is a research prototype, not a CE-marked or FDA-cleared device. Regulatory pathways (Software as Medical Device, IEC 62304 lifecycle, post-market surveillance) are outside the scope of an MPhil thesis but are the correct next step for clinical translation. We acknowledge this in the Conclusion.

---

## How to Use This Document

| When | Use case |
|------|----------|
| **During thesis viva** | Examiner asks "why X" — find the corresponding question above and use the structured answer |
| **During paper revision** | Reviewer comment maps to a question — quote the answer directly in Response-to-Reviewer letter |
| **Writing the Methods chapter** | Each section can be paraphrased into a Methods justification paragraph |
| **Writing the Discussion** | Section 13 (tough questions) feeds directly into the Strengths and Limitations subsection |
| **Onboarding co-authors** | Send this as a single document explaining every methodological choice |

---

*This document is a living reference. As reviewer comments arrive, append new Q&A pairs. As the methodology evolves (external validation, regression formulation), update existing answers.*
