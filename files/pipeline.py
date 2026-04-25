#!/usr/bin/env python3
"""
=============================================================================
AI-Based Myopia Prediction — Complete Research Pipeline
=============================================================================
Title  : Development and Validation of AI Based Model for Myopia Prediction
Author : Syed Ahmad Hassan (2024-MPhil-OP-037)
=============================================================================
"""

import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from scipy import stats
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

OUT = "/home/claude/outputs"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 : DATA LOADING & INITIAL INSPECTION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 1: DATA LOADING & INITIAL INSPECTION")
print("=" * 70)

df_raw = pd.read_csv("/mnt/user-data/uploads/clinical_data_and_labels.csv")
print(f"Raw dataset shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
print(f"\nLabel distribution:\n{df_raw['label'].value_counts()}")
print(f"\nMissing values:\n{df_raw.isnull().sum()}")
print(f"\nBasic statistics:\n{df_raw.describe().round(3)}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 : DATA PREPROCESSING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 2: DATA PREPROCESSING & CLEANING")
print("=" * 70)

df = df_raw.copy()

# Encode categorical
df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])  # f=0, m=1
df['eye_encoded'] = LabelEncoder().fit_transform(df['eye'])        # OD=0, OS=1

# Check for outliers using IQR
numeric_cols = ['age_years', 'astig_value_D', 'astig_axis_deg', 'kmax_value_D',
                'kmax_axis_deg', 'pachy_central_um', 'pachy_thinnest_um',
                'pachy_thinnest_x', 'pachy_thinnest_y', 'asphericity_anterior',
                'asphericity_posterior']

outlier_report = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = n_outliers
    
print("Outlier counts (IQR method):")
for k, v in outlier_report.items():
    print(f"  {k}: {v}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 : FEATURE ENGINEERING — Creating Research-Grade Features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 3: FEATURE ENGINEERING")
print("=" * 70)

# 3.1 Pachymetry features
df['pachy_diff'] = df['pachy_central_um'] - df['pachy_thinnest_um']
df['pachy_ratio'] = df['pachy_thinnest_um'] / df['pachy_central_um']
df['pachy_thinnest_displacement'] = np.sqrt(df['pachy_thinnest_x']**2 + df['pachy_thinnest_y']**2)

# 3.2 Asphericity features
df['asphericity_diff'] = df['asphericity_anterior'] - df['asphericity_posterior']
df['asphericity_ratio'] = df['asphericity_anterior'] / (df['asphericity_posterior'].replace(0, np.nan))
df['asphericity_ratio'] = df['asphericity_ratio'].fillna(0)
df['asphericity_abs_sum'] = abs(df['asphericity_anterior']) + abs(df['asphericity_posterior'])

# 3.3 Astigmatism features  
df['astig_abs'] = abs(df['astig_value_D'])
df['astig_axis_category'] = pd.cut(df['astig_axis_deg'], bins=[0, 30, 60, 120, 150, 180],
                                     labels=['WTR_steep', 'Oblique1', 'ATR', 'Oblique2', 'WTR_flat'],
                                     include_lowest=True)
df['astig_axis_sin'] = np.sin(2 * np.radians(df['astig_axis_deg']))
df['astig_axis_cos'] = np.cos(2 * np.radians(df['astig_axis_deg']))

# 3.4 Keratometry features
df['kmax_axis_sin'] = np.sin(2 * np.radians(df['kmax_axis_deg']))
df['kmax_axis_cos'] = np.cos(2 * np.radians(df['kmax_axis_deg']))

# 3.5 Corneal power index (derived)
df['corneal_power_index'] = df['kmax_value_D'] * (1 + df['asphericity_anterior'])

# 3.6 Age-based grouping
df['age_group'] = pd.cut(df['age_years'], bins=[0, 18, 25, 35, 50, 100],
                          labels=['adolescent', 'young_adult', 'adult', 'middle_age', 'senior'])

# 3.7 Interaction features
df['kmax_astig_interaction'] = df['kmax_value_D'] * df['astig_abs']
df['age_kmax_interaction'] = df['age_years'] * df['kmax_value_D']
df['pachy_asph_interaction'] = df['pachy_central_um'] * abs(df['asphericity_anterior'])

# 3.8 Risk scoring (composite)
df['corneal_risk_score'] = (
    (df['kmax_value_D'] > 46).astype(int) +
    (df['astig_abs'] > 3).astype(int) +
    (df['pachy_central_um'] < 520).astype(int) +
    (df['asphericity_anterior'] > 0).astype(int)
)

print(f"Features after engineering: {df.shape[1]} columns")
print(f"New engineered features: {df.shape[1] - df_raw.shape[1]}")
print(f"\nNew columns: {[c for c in df.columns if c not in df_raw.columns]}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 : EXPLORATORY DATA ANALYSIS (EDA) — Comprehensive Visualizations
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 4: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = ['#2196F3', '#FF5722']
PALETTE = {0: '#2196F3', 1: '#FF5722'}
PALETTE_STR = {'0': '#2196F3', '1': '#FF5722'}

# --- 4.1 Label Distribution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels_counts = df['label'].value_counts()
axes[0].pie(labels_counts, labels=['Non-Progressive (0)', 'Progressive (1)'],
            colors=COLORS, autopct='%1.1f%%', startangle=90,
            explode=(0.05, 0.05), shadow=True, textprops={'fontsize': 12})
axes[0].set_title('Label Distribution', fontsize=14, fontweight='bold')

sns.countplot(data=df, x='gender', hue='label', ax=axes[1], palette=PALETTE)
axes[1].set_title('Gender vs Label', fontsize=14, fontweight='bold')
axes[1].legend(title='Label', labels=['Non-Progressive', 'Progressive'])

sns.countplot(data=df, x='age_group', hue='label', ax=axes[2], palette=PALETTE,
              order=['adolescent', 'young_adult', 'adult', 'middle_age', 'senior'])
axes[2].set_title('Age Group vs Label', fontsize=14, fontweight='bold')
axes[2].legend(title='Label', labels=['Non-Progressive', 'Progressive'])
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f"{OUT}/01_label_distribution.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 01_label_distribution.png")

# --- 4.2 Feature Distributions ---
fig, axes = plt.subplots(3, 4, figsize=(22, 15))
axes = axes.ravel()
for i, col in enumerate(numeric_cols):
    if i < 12:
        for lab, color in zip([0, 1], COLORS):
            subset = df[df['label'] == lab][col]
            axes[i].hist(subset, bins=30, alpha=0.6, color=color,
                        label=f"{'Non-Prog' if lab==0 else 'Prog'}")
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].legend(fontsize=8)
if len(numeric_cols) < 12:
    for j in range(len(numeric_cols), 12):
        axes[j].set_visible(False)
plt.suptitle('Feature Distributions by Label', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/02_feature_distributions.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 02_feature_distributions.png")

# --- 4.3 Correlation Heatmap ---
feature_cols_for_corr = numeric_cols + ['pachy_diff', 'pachy_ratio', 'pachy_thinnest_displacement',
    'asphericity_diff', 'asphericity_abs_sum', 'astig_abs', 'corneal_power_index',
    'kmax_astig_interaction', 'corneal_risk_score', 'label']
corr_matrix = df[feature_cols_for_corr].corr()

fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 7}, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/03_correlation_heatmap.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 03_correlation_heatmap.png")

# --- 4.4 Box Plots (key features) ---
key_features = ['kmax_value_D', 'astig_abs', 'pachy_central_um', 'pachy_thinnest_um',
                'asphericity_anterior', 'asphericity_posterior', 'pachy_diff', 'corneal_power_index']
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.ravel()
for i, col in enumerate(key_features):
    sns.boxplot(data=df, x="label", y=col, ax=axes[i], palette=COLORS)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xticklabels(['Non-Progressive', 'Progressive'])
plt.suptitle('Key Feature Distributions by Label (Box Plots)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/04_boxplots.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 04_boxplots.png")

# --- 4.5 Violin Plots ---
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.ravel()
for i, col in enumerate(key_features):
    sns.violinplot(data=df, x="label", y=col, ax=axes[i], palette=COLORS, inner="quartile")
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xticklabels(['Non-Progressive', 'Progressive'])
plt.suptitle('Violin Plots — Feature Distribution by Label', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/05_violin_plots.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 05_violin_plots.png")

# --- 4.6 Pair Plot (top features) ---
top_features = ['kmax_value_D', 'astig_abs', 'pachy_central_um', 'asphericity_anterior', 'label']
g = sns.pairplot(df[top_features], hue='label', palette=PALETTE,
                 diag_kind='kde', corner=True, plot_kws={'alpha': 0.4, 's': 15})
g.figure.suptitle('Pair Plot — Top Features', y=1.02, fontsize=16, fontweight='bold')
plt.savefig(f"{OUT}/06_pairplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("  [✓] 06_pairplot.png")

# --- 4.7 Statistical Tests ---
print("\n  Statistical significance tests (Mann-Whitney U):")
stat_results = {}
for col in numeric_cols + ['pachy_diff', 'pachy_ratio', 'astig_abs', 'corneal_power_index',
                           'asphericity_diff', 'corneal_risk_score']:
    g0 = df[df['label'] == 0][col].dropna()
    g1 = df[df['label'] == 1][col].dropna()
    stat, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
    stat_results[col] = {'U-stat': stat, 'p-value': p, 'significant': p < 0.05}
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    {col:30s}  p={p:.6f}  {sig}")

# --- 4.8 Plotly Interactive Charts ---
# Scatter 3D
fig_3d = px.scatter_3d(df, x='kmax_value_D', y='astig_abs', z='pachy_central_um',
                        color=df['label'].map({0: 'Non-Progressive', 1: 'Progressive'}),
                        color_discrete_map={'Non-Progressive': '#2196F3', 'Progressive': '#FF5722'},
                        opacity=0.6, title='3D Feature Space — Myopia Classification',
                        labels={'color': 'Label'})
fig_3d.write_html(f"{OUT}/07_3d_scatter.html")
print("  [✓] 07_3d_scatter.html / .png")

# Sunburst
fig_sun = px.sunburst(df, path=['gender', 'age_group', 'label'], 
                       title='Hierarchical View: Gender → Age Group → Label',
                       color='label', color_continuous_scale='RdBu_r')
fig_sun.write_html(f"{OUT}/08_sunburst.html")
print("  [✓] 08_sunburst.html / .png")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 : DATA PREPARATION FOR MODELING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 5: DATA PREPARATION FOR MODELING")
print("=" * 70)

feature_columns = [
    'age_years', 'gender_encoded', 'eye_encoded',
    'astig_value_D', 'astig_axis_deg', 'kmax_value_D', 'kmax_axis_deg',
    'pachy_central_um', 'pachy_thinnest_um', 'pachy_thinnest_x', 'pachy_thinnest_y',
    'asphericity_anterior', 'asphericity_posterior',
    # Engineered features
    'pachy_diff', 'pachy_ratio', 'pachy_thinnest_displacement',
    'asphericity_diff', 'asphericity_abs_sum', 'astig_abs',
    'astig_axis_sin', 'astig_axis_cos', 'kmax_axis_sin', 'kmax_axis_cos',
    'corneal_power_index', 'kmax_astig_interaction', 'age_kmax_interaction',
    'pachy_asph_interaction', 'corneal_risk_score'
]

X = df[feature_columns].copy()
y = df['label'].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {dict(y.value_counts())}")

# Train-Test Split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Train label dist: {dict(y_train.value_counts())}")
print(f"Test  label dist: {dict(y_test.value_counts())}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE for balanced training
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"\nAfter SMOTE — Train: {X_train_smote.shape[0]}")
print(f"SMOTE label dist: {dict(pd.Series(y_train_smote).value_counts())}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 : MULTI-MODEL TRAINING & CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 6: MODEL TRAINING (10 Models)")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False,
                              eval_metric='logloss', verbosity=0),
    'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                     random_state=42, early_stopping=True),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n  Training: {name}...")
    
    # Cross-validation on SMOTE train
    cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
    
    # Fit and predict
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'cm': cm,
    }
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    results[name]['specificity'] = specificity
    
    print(f"    CV Acc: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test → Acc: {acc:.4f} | Prec: {prec:.4f} | Rec(Sens): {rec:.4f} | Spec: {specificity:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 : MODEL EVALUATION VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 7: MODEL EVALUATION VISUALIZATIONS")
print("=" * 70)

# --- 7.1 Comprehensive Model Comparison Bar Chart ---
metrics_df = pd.DataFrame({
    name: {
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Sensitivity': r['recall'],
        'Specificity': r['specificity'],
        'F1-Score': r['f1'],
        'AUC-ROC': r['auc'],
    } for name, r in results.items()
}).T

fig, ax = plt.subplots(figsize=(20, 8))
metrics_df.plot(kind='bar', ax=ax, width=0.8,
                color=['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#D32F2F', '#00796B'])
ax.set_title('Model Performance Comparison — All Metrics', fontsize=16, fontweight='bold')
ax.set_ylabel('Score', fontsize=13)
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=10)
ax.tick_params(axis='x', rotation=35)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=7, rotation=90, padding=3)
plt.tight_layout()
plt.savefig(f"{OUT}/09_model_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 09_model_comparison.png")

# --- 7.2 ROC Curves (All Models) ---
fig, ax = plt.subplots(figsize=(12, 10))
colors_roc = plt.cm.tab10(np.linspace(0, 1, len(results)))
for i, (name, r) in enumerate(results.items()):
    if r['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.4f})", color=colors_roc[i], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
ax.set_title('ROC Curves — All Models', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/10_roc_curves.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 10_roc_curves.png")

# --- 7.3 Precision-Recall Curves ---
fig, ax = plt.subplots(figsize=(12, 10))
for i, (name, r) in enumerate(results.items()):
    if r['y_prob'] is not None:
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, r['y_prob'])
        ap = average_precision_score(y_test, r['y_prob'])
        ax.plot(rec_vals, prec_vals, label=f"{name} (AP={ap:.4f})", color=colors_roc[i], linewidth=2)
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curves — All Models', fontsize=16, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/11_precision_recall_curves.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 11_precision_recall_curves.png")

# --- 7.4 Confusion Matrices (Grid) ---
fig, axes = plt.subplots(2, 5, figsize=(30, 12))
axes = axes.ravel()
for i, (name, r) in enumerate(results.items()):
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Non-Prog', 'Prog'], yticklabels=['Non-Prog', 'Prog'],
                annot_kws={'size': 14})
    axes[i].set_title(f"{name}\nAcc={r['accuracy']:.3f}", fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')
plt.suptitle('Confusion Matrices — All Models', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/12_confusion_matrices.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 12_confusion_matrices.png")

# --- 7.5 Cross-Validation Box Plot ---
fig, ax = plt.subplots(figsize=(16, 8))
cv_data = []
cv_labels = []
for name, model_obj in models.items():
    scores = cross_val_score(results[name]['model'], X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
    cv_data.append(scores)
    cv_labels.append(name)

bp = ax.boxplot(cv_data, labels=cv_labels, patch_artist=True, showmeans=True)
colors_box = plt.cm.Set3(np.linspace(0, 1, len(cv_labels)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax.set_title('5-Fold Cross-Validation Accuracy Distribution', fontsize=16, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13)
ax.tick_params(axis='x', rotation=35)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/13_cv_boxplot.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 13_cv_boxplot.png")

# --- 7.6 Feature Importance (Top 3 Models) ---
top_models = ['Random Forest', 'XGBoost', 'LightGBM']
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for i, name in enumerate(top_models):
    model = results[name]['model']
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]  # top 15
    axes[i].barh([feature_columns[j] for j in indices], importances[indices], color=COLORS[i % 2])
    axes[i].set_title(f'{name}\nFeature Importance (Top 15)', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Importance')
plt.suptitle('Feature Importance — Top Ensemble Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/14_feature_importance.png", dpi=200, bbox_inches='tight')
plt.close()
print("  [✓] 14_feature_importance.png")

# --- 7.7 Plotly Interactive Radar Chart ---
top5 = sorted(results.keys(), key=lambda x: results[x]['auc'], reverse=True)[:5]
fig_radar = go.Figure()
categories = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
for name in top5:
    r = results[name]
    values = [r['accuracy'], r['precision'], r['recall'], r['specificity'], r['f1'], r['auc']]
    fig_radar.add_trace(go.Scatterpolar(r=values + [values[0]],
                                         theta=categories + [categories[0]],
                                         fill='toself', name=name, opacity=0.6))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                         title='Top 5 Models — Performance Radar', font_size=12)
fig_radar.write_html(f"{OUT}/15_radar_chart.html")
print("  [✓] 15_radar_chart.html / .png")

# --- 7.8 Plotly Heatmap — Model x Metric ---
fig_hm = go.Figure(data=go.Heatmap(
    z=metrics_df.values,
    x=metrics_df.columns.tolist(),
    y=metrics_df.index.tolist(),
    colorscale='RdYlGn',
    text=np.round(metrics_df.values, 4),
    texttemplate='%{text}',
    textfont=dict(size=11),
    zmin=0.5, zmax=1.0
))
fig_hm.update_layout(title='Model × Metric Performance Heatmap',
                      height=500, width=900)
fig_hm.write_html(f"{OUT}/16_model_metric_heatmap.html")
print("  [✓] 16_model_metric_heatmap.html / .png")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 : ENSEMBLE STACKING MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 8: ENSEMBLE STACKING MODEL")
print("=" * 70)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False,
                           eval_metric='logloss', verbosity=0)),
    ('lgbm', LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, n_jobs=-1
)
stacking_model.fit(X_train_smote, y_train_smote)
y_pred_stack = stacking_model.predict(X_test_scaled)
y_prob_stack = stacking_model.predict_proba(X_test_scaled)[:, 1]

stack_acc = accuracy_score(y_test, y_pred_stack)
stack_prec = precision_score(y_test, y_pred_stack)
stack_rec = recall_score(y_test, y_pred_stack)
stack_f1 = f1_score(y_test, y_pred_stack)
stack_auc = roc_auc_score(y_test, y_prob_stack)
stack_cm = confusion_matrix(y_test, y_pred_stack)
tn, fp, fn, tp = stack_cm.ravel()
stack_spec = tn / (tn + fp)

print(f"  Stacking Ensemble Results:")
print(f"    Accuracy   : {stack_acc:.4f}")
print(f"    Precision  : {stack_prec:.4f}")
print(f"    Sensitivity: {stack_rec:.4f}")
print(f"    Specificity: {stack_spec:.4f}")
print(f"    F1-Score   : {stack_f1:.4f}")
print(f"    AUC-ROC    : {stack_auc:.4f}")

results['Stacking Ensemble'] = {
    'accuracy': stack_acc, 'precision': stack_prec, 'recall': stack_rec,
    'specificity': stack_spec, 'f1': stack_f1, 'auc': stack_auc,
    'y_pred': y_pred_stack, 'y_prob': y_prob_stack, 'cm': stack_cm,
    'cv_mean': 0, 'cv_std': 0, 'model': stacking_model,
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9 : FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 9: FINAL RESULTS SUMMARY")
print("=" * 70)

summary_data = []
for name, r in results.items():
    summary_data.append({
        'Model': name,
        'Accuracy': round(r['accuracy'], 4),
        'Precision': round(r['precision'], 4),
        'Sensitivity': round(r['recall'], 4),
        'Specificity': round(r['specificity'], 4),
        'F1-Score': round(r['f1'], 4),
        'AUC-ROC': round(r['auc'], 4),
    })
summary_df = pd.DataFrame(summary_data).sort_values('AUC-ROC', ascending=False)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(f"{OUT}/model_results_summary.csv", index=False)
print(f"\n  [✓] model_results_summary.csv saved")

# Final summary chart with Stacking
fig_final = go.Figure()
for metric in ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']:
    fig_final.add_trace(go.Bar(
        name=metric,
        x=summary_df['Model'],
        y=summary_df[metric],
        text=summary_df[metric].round(3),
        textposition='outside',
        textfont=dict(size=8),
    ))
fig_final.update_layout(
    title='Final Model Comparison (Including Stacking Ensemble)',
    barmode='group', height=600, width=1400,
    yaxis=dict(range=[0, 1.1]),
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
)
fig_final.write_html(f"{OUT}/17_final_comparison.html")
print("  [✓] 17_final_comparison.png")

# Save processed data
df.to_csv(f"{OUT}/processed_data_with_features.csv", index=False)
print("  [✓] processed_data_with_features.csv")

# Save model artifacts for Streamlit
import pickle
best_model_name = summary_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
with open(f"{OUT}/best_model.pkl", 'wb') as f:
    pickle.dump(best_model, f)
with open(f"{OUT}/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
with open(f"{OUT}/feature_columns.json", 'w') as f:
    json.dump(feature_columns, f)

print(f"\n{'='*70}")
print(f"PIPELINE COMPLETE — Best Model: {best_model_name} (AUC={summary_df.iloc[0]['AUC-ROC']:.4f})")
print(f"{'='*70}")
