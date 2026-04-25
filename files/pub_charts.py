#!/usr/bin/env python3
"""
Publication-grade figures for AI-Based Myopia Prediction paper.
Clean, minimal, journal-ready (IEEE/Elsevier/Frontiers style).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import os

OUT = "/home/claude/pub_figures"
os.makedirs(OUT, exist_ok=True)

# ── Academic style config ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.linewidth': 0.6,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.5,
})

TEAL = '#0F6E56'
TEAL_LIGHT = '#5DCAA5'
GRAY = '#5F5E5A'
GRAY_LIGHT = '#B4B2A9'
PURPLE = '#534AB7'
PURPLE_LIGHT = '#AFA9EC'

# ── Load and prep data (same as pipeline.py) ─────────────────────────────────
df = pd.read_csv("/mnt/user-data/uploads/clinical_data_and_labels.csv")
df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
df['eye_encoded'] = LabelEncoder().fit_transform(df['eye'])
df['pachy_diff'] = df['pachy_central_um'] - df['pachy_thinnest_um']
df['pachy_ratio'] = df['pachy_thinnest_um'] / df['pachy_central_um']
df['pachy_thinnest_displacement'] = np.sqrt(df['pachy_thinnest_x']**2 + df['pachy_thinnest_y']**2)
df['asphericity_diff'] = df['asphericity_anterior'] - df['asphericity_posterior']
df['asphericity_abs_sum'] = abs(df['asphericity_anterior']) + abs(df['asphericity_posterior'])
df['astig_abs'] = abs(df['astig_value_D'])
df['astig_axis_sin'] = np.sin(2 * np.radians(df['astig_axis_deg']))
df['astig_axis_cos'] = np.cos(2 * np.radians(df['astig_axis_deg']))
df['kmax_axis_sin'] = np.sin(2 * np.radians(df['kmax_axis_deg']))
df['kmax_axis_cos'] = np.cos(2 * np.radians(df['kmax_axis_deg']))
df['corneal_power_index'] = df['kmax_value_D'] * (1 + df['asphericity_anterior'])
df['kmax_astig_interaction'] = df['kmax_value_D'] * df['astig_abs']
df['age_kmax_interaction'] = df['age_years'] * df['kmax_value_D']
df['pachy_asph_interaction'] = df['pachy_central_um'] * abs(df['asphericity_anterior'])
df['corneal_risk_score'] = (
    (df['kmax_value_D'] > 46).astype(int) + (df['astig_abs'] > 3).astype(int) +
    (df['pachy_central_um'] < 520).astype(int) + (df['asphericity_anterior'] > 0).astype(int)
)
df['asphericity_ratio'] = df['asphericity_anterior'] / df['asphericity_posterior'].replace(0, np.nan)
df['asphericity_ratio'] = df['asphericity_ratio'].fillna(0)

feature_columns = [
    'age_years', 'gender_encoded', 'eye_encoded',
    'astig_value_D', 'astig_axis_deg', 'kmax_value_D', 'kmax_axis_deg',
    'pachy_central_um', 'pachy_thinnest_um', 'pachy_thinnest_x', 'pachy_thinnest_y',
    'asphericity_anterior', 'asphericity_posterior',
    'pachy_diff', 'pachy_ratio', 'pachy_thinnest_displacement',
    'asphericity_diff', 'asphericity_abs_sum', 'astig_abs',
    'astig_axis_sin', 'astig_axis_cos', 'kmax_axis_sin', 'kmax_axis_cos',
    'corneal_power_index', 'kmax_astig_interaction', 'age_kmax_interaction',
    'pachy_asph_interaction', 'corneal_risk_score'
]

X = df[feature_columns]; y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
sm = SMOTE(random_state=42)
X_tr, y_tr = sm.fit_resample(X_train_s, y_train)

models = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'DT': DecisionTreeClassifier(random_state=42, max_depth=10),
    'RF': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'XGB': XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0),
    'LGBM': LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    'GB': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Ada': AdaBoostClassifier(n_estimators=100, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=500, random_state=42, early_stopping=True),
}

results = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    yp = m.predict(X_test_s)
    ypr = m.predict_proba(X_test_s)[:,1] if hasattr(m,'predict_proba') else None
    cm = confusion_matrix(y_test, yp)
    tn,fp,fn,tp = cm.ravel()
    results[name] = {
        'acc': accuracy_score(y_test,yp), 'prec': precision_score(y_test,yp),
        'rec': recall_score(y_test,yp), 'spec': tn/(tn+fp),
        'f1': f1_score(y_test,yp), 'auc': roc_auc_score(y_test,ypr) if ypr is not None else 0,
        'yp': yp, 'ypr': ypr, 'cm': cm, 'model': m,
    }

# Stacking
stack = StackingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)),
    ('lgbm', LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
], final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1)
stack.fit(X_tr, y_tr)
yp_s = stack.predict(X_test_s); ypr_s = stack.predict_proba(X_test_s)[:,1]
cm_s = confusion_matrix(y_test, yp_s); tn,fp,fn,tp = cm_s.ravel()
results['Stack'] = {
    'acc': accuracy_score(y_test,yp_s), 'prec': precision_score(y_test,yp_s),
    'rec': recall_score(y_test,yp_s), 'spec': tn/(tn+fp),
    'f1': f1_score(y_test,yp_s), 'auc': roc_auc_score(y_test,ypr_s),
    'yp': yp_s, 'ypr': ypr_s, 'cm': cm_s, 'model': stack,
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A — ROC CURVES (Publication Grade)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))

line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-']
gray_ramp = ['#1a1a1a','#333333','#4d4d4d','#666666','#808080','#999999','#b3b3b3','#4d4d4d','#1a1a1a','#666666', TEAL]
model_order = ['SVM','LR','GB','MLP','LGBM','XGB','Ada','KNN','RF','DT','Stack']

for i, name in enumerate(model_order):
    r = results[name]
    if r['ypr'] is not None:
        fpr, tpr, _ = roc_curve(y_test, r['ypr'])
        lw = 1.8 if name in ['SVM','Stack','LR'] else 0.9
        alpha = 1.0 if name in ['SVM','Stack','LR'] else 0.5
        ax.plot(fpr, tpr, linestyle=line_styles[i], color=gray_ramp[i],
                linewidth=lw, alpha=alpha,
                label=f"{name} ({r['auc']:.3f})")

ax.plot([0,1],[0,1], 'k:', linewidth=0.5, alpha=0.3)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')
ax.legend(loc='lower right', title='Model (AUC)', title_fontsize=9, frameon=True)
plt.savefig(f"{OUT}/fig_roc_curves.png")
plt.savefig(f"{OUT}/fig_roc_curves.pdf")
plt.close()
print("[✓] ROC curves")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B — PRECISION-RECALL CURVES
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))
for i, name in enumerate(model_order):
    r = results[name]
    if r['ypr'] is not None:
        prec_v, rec_v, _ = precision_recall_curve(y_test, r['ypr'])
        ap = average_precision_score(y_test, r['ypr'])
        lw = 1.8 if name in ['SVM','Stack','LR'] else 0.9
        alpha = 1.0 if name in ['SVM','Stack','LR'] else 0.5
        ax.plot(rec_v, prec_v, linestyle=line_styles[i], color=gray_ramp[i],
                linewidth=lw, alpha=alpha,
                label=f"{name} ({ap:.3f})")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.5, 1.05])
ax.legend(loc='lower left', title='Model (AP)', title_fontsize=9, frameon=True)
plt.savefig(f"{OUT}/fig_pr_curves.png")
plt.savefig(f"{OUT}/fig_pr_curves.pdf")
plt.close()
print("[✓] PR curves")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE C — MODEL COMPARISON (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════════
metrics = ['acc','prec','rec','spec','f1','auc']
metric_labels = ['Accuracy','Precision','Sensitivity','Specificity','F1-score','AUC-ROC']
model_names = list(results.keys())

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(model_names))
w = 0.12
colors_bar = [TEAL, TEAL_LIGHT, GRAY, GRAY_LIGHT, PURPLE, PURPLE_LIGHT]

for i, (met, lab) in enumerate(zip(metrics, metric_labels)):
    vals = [results[m][met] for m in model_names]
    offset = (i - 2.5) * w
    bars = ax.bar(x + offset, vals, w, label=lab, color=colors_bar[i], edgecolor='white', linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=0)
ax.set_ylabel('Score')
ax.set_ylim(0.88, 1.02)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.legend(loc='lower left', ncol=3, frameon=True, fontsize=8)
ax.axhline(y=0.95, color='#cccccc', linewidth=0.5, linestyle='--', zorder=0)
plt.savefig(f"{OUT}/fig_model_comparison.png")
plt.savefig(f"{OUT}/fig_model_comparison.pdf")
plt.close()
print("[✓] Model comparison")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE D — CONFUSION MATRICES (Top 4 + Stacking)
# ═══════════════════════════════════════════════════════════════════════════════
top_models = ['SVM', 'LR', 'GB', 'RF', 'Stack']
fig, axes = plt.subplots(1, 5, figsize=(14, 2.8))
for i, name in enumerate(top_models):
    cm = results[name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', ax=axes[i],
                xticklabels=['NP','P'], yticklabels=['NP','P'],
                cbar=False, annot_kws={'size': 12},
                linewidths=0.5, linecolor='#e0e0e0')
    axes[i].set_title(f"{name}", fontsize=11)
    axes[i].set_ylabel('Actual' if i == 0 else '')
    axes[i].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(f"{OUT}/fig_confusion_matrices.png")
plt.savefig(f"{OUT}/fig_confusion_matrices.pdf")
plt.close()
print("[✓] Confusion matrices")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE E — FEATURE IMPORTANCE (RF, XGB, LGBM side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for idx, (name, color) in enumerate([('RF', TEAL), ('XGB', GRAY), ('LGBM', PURPLE)]):
    imp = results[name]['model'].feature_importances_
    top_idx = np.argsort(imp)[-12:]
    feat_names = [feature_columns[j] for j in top_idx]
    feat_names = [n.replace('_', ' ') for n in feat_names]
    axes[idx].barh(feat_names, imp[top_idx], color=color, height=0.6, edgecolor='white', linewidth=0.3)
    axes[idx].set_title(name, fontsize=11)
    axes[idx].set_xlabel('Importance')
    axes[idx].tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_feature_importance.png")
plt.savefig(f"{OUT}/fig_feature_importance.pdf")
plt.close()
print("[✓] Feature importance")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE F — CORRELATION WITH LABEL (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════════════
corr = df[feature_columns + ['label']].corr()['label'].drop('label').sort_values()
fig, ax = plt.subplots(figsize=(5.5, 7))
colors_corr = [TEAL if v > 0 else GRAY for v in corr.values]
ax.barh([c.replace('_',' ') for c in corr.index], corr.values, color=colors_corr, height=0.6,
        edgecolor='white', linewidth=0.3)
ax.axvline(x=0, color='#333', linewidth=0.5)
ax.set_xlabel('Pearson correlation with progression label')
ax.tick_params(axis='y', labelsize=8)
plt.savefig(f"{OUT}/fig_correlation_bar.png")
plt.savefig(f"{OUT}/fig_correlation_bar.pdf")
plt.close()
print("[✓] Correlation bar")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE G — KEY FEATURE DISTRIBUTIONS (Violin Plots, clean)
# ═══════════════════════════════════════════════════════════════════════════════
key_feats = ['kmax_value_D', 'astig_abs', 'pachy_central_um', 'asphericity_anterior',
             'pachy_diff', 'corneal_power_index']
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.ravel()
for i, col in enumerate(key_feats):
    parts = axes[i].violinplot(
        [df[df['label']==0][col].values, df[df['label']==1][col].values],
        positions=[0, 1], showmeans=True, showmedians=True, widths=0.7
    )
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(TEAL if j == 0 else PURPLE)
        pc.set_alpha(0.4)
    for partname in ['cmeans','cmedians','cbars','cmins','cmaxes']:
        if partname in parts:
            parts[partname].set_color('#333')
            parts[partname].set_linewidth(0.5)
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['NP', 'P'])
    axes[i].set_title(col.replace('_', ' '), fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_violin_distributions.png")
plt.savefig(f"{OUT}/fig_violin_distributions.pdf")
plt.close()
print("[✓] Violin distributions")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE H — CROSS-VALIDATION STABILITY
# ═══════════════════════════════════════════════════════════════════════════════
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fig, ax = plt.subplots(figsize=(8, 4))
cv_data = []
cv_names = []
for name in model_names:
    scores = cross_val_score(results[name]['model'], X_tr, y_tr, cv=cv, scoring='accuracy')
    cv_data.append(scores)
    cv_names.append(name)

bp = ax.boxplot(cv_data, labels=cv_names, patch_artist=True, widths=0.5,
                medianprops=dict(color='#333', linewidth=1),
                whiskerprops=dict(linewidth=0.5),
                capprops=dict(linewidth=0.5),
                flierprops=dict(markersize=3))
for patch in bp['boxes']:
    patch.set_facecolor(TEAL_LIGHT)
    patch.set_alpha(0.5)
    patch.set_edgecolor(TEAL)
    patch.set_linewidth(0.5)
ax.set_ylabel('5-fold CV accuracy')
ax.set_ylim(0.92, 1.01)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig(f"{OUT}/fig_cv_stability.png")
plt.savefig(f"{OUT}/fig_cv_stability.pdf")
plt.close()
print("[✓] CV stability")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I — CORRELATION HEATMAP (Clean Academic)
# ═══════════════════════════════════════════════════════════════════════════════
key_corr_feats = ['kmax_value_D', 'astig_abs', 'pachy_central_um', 'pachy_thinnest_um',
                  'asphericity_anterior', 'asphericity_posterior', 'pachy_diff',
                  'corneal_power_index', 'kmax_astig_interaction', 'corneal_risk_score', 'label']
corr_mat = df[key_corr_feats].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True, linewidths=0.3,
            ax=ax, annot_kws={'size': 8}, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
labels = [c.replace('_',' ') for c in key_corr_feats]
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
plt.savefig(f"{OUT}/fig_correlation_heatmap.png")
plt.savefig(f"{OUT}/fig_correlation_heatmap.pdf")
plt.close()
print("[✓] Correlation heatmap")

print(f"\nAll figures saved to {OUT}/")
