"""
src/eda/visualizer.py — Comprehensive Exploratory Data Analysis
================================================================
Generates 11 static publication figures plus 2 interactive HTML
visualisations covering the entire dataset.

Figures
-------
 1. Label distribution (pie + counts + gender × label + age group × label)
 2. Feature distributions by class (histogram + KDE)
 3. Correlation heatmap (full Pearson matrix)
 4. Ranked feature–target correlations
 5. Box plots with Mann–Whitney U p-values
 6. Violin plots (full distribution shape)
 7. Pair plot (top 4 features)
 8. Statistical significance summary (p-values + Cohen's d)
 9. Clinical grouping analysis (risk scores, axis types)
10. Engineered features by class
11. Descriptive statistics table
12. Interactive 3D scatter (HTML)
13. Interactive sunburst (HTML)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

from src.config import (
    COLOR_NEG,
    COLOR_POS,
    COLORS,
    DPI_PUBLICATION,
    DPI_SCREEN,
    FIG_EDA,
    FIG_PUBLICATION,
    PALETTE,
    PUB_RC,
)

try:
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False


def _save(fig: plt.Figure, path: Path, dpi: int = DPI_SCREEN) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(
        ((n1 - 1) * a.std() ** 2 + (n2 - 1) * b.std() ** 2) / (n1 + n2 - 2)
    )
    return (a.mean() - b.mean()) / (pooled_std + 1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 1. Label distribution dashboard
# ──────────────────────────────────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame, out_dir: Path = FIG_EDA) -> None:
    label_counts = df["label"].value_counts().sort_index()
    label_names = {0: "Non-Progressive", 1: "Progressive"}

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        wedges, texts, autotexts = axes[0].pie(
            label_counts.values,
            labels=[label_names[k] for k in label_counts.index],
            colors=COLORS,
            autopct="%1.1f%%", startangle=90, explode=(0.04, 0.04),
            wedgeprops={"linewidth": 2, "edgecolor": "white"},
            textprops={"fontsize": 14, "fontweight": "bold"},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(14)
        axes[0].set_title("Overall Label Distribution")

        sns.countplot(data=df, x="label", ax=axes[1], palette=COLORS, edgecolor="black", linewidth=0.6)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["Non-Progressive", "Progressive"])
        for p in axes[1].patches:
            axes[1].annotate(
                f"{int(p.get_height()):,}\n({p.get_height() / len(df) * 100:.1f}%)",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )
        axes[1].set_title("Sample Counts")
        axes[1].set_ylabel("Count")

        sns.countplot(data=df, x="gender", hue="label", ax=axes[2],
                      palette=COLORS, edgecolor="black", linewidth=0.6)
        axes[2].set_title("Gender × Label")
        axes[2].legend(title="Label", labels=["Non-Prog", "Prog"], fontsize=11)

        if "age_group" in df.columns:
            order = ["adolescent", "young_adult", "adult", "middle_age", "senior"]
            try:
                cats = list(df["age_group"].cat.categories)
                order = [o for o in order if o in cats]
            except AttributeError:
                pass
            sns.countplot(
                data=df, x="age_group", hue="label", ax=axes[3],
                palette=COLORS, order=order, edgecolor="black", linewidth=0.6,
            )
            axes[3].tick_params(axis="x", rotation=25)
            axes[3].set_title("Age Group × Label")
            axes[3].legend(title="Label", labels=["Non-Prog", "Prog"], fontsize=11)
            for label in axes[3].get_xticklabels():
                label.set_horizontalalignment("right")

        fig.suptitle("Label Distribution Analysis", y=1.02)
        plt.tight_layout()
        _save(fig, out_dir / "01_label_distribution.png")
    print("  [OK] Label distribution")


# ──────────────────────────────────────────────────────────────────────────
# 2. Feature distributions
# ──────────────────────────────────────────────────────────────────────────

def plot_feature_distributions(
    df: pd.DataFrame, numeric_cols: list[str], out_dir: Path = FIG_EDA,
) -> None:
    n = len(numeric_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.5 * nrows))
        axes = axes.ravel()
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            for lab, color, name in zip(
                [0, 1], COLORS, ["Non-Progressive", "Progressive"]
            ):
                subset = df[df["label"] == lab][col].dropna()
                ax.hist(
                    subset, bins=30, alpha=0.55, color=color,
                    label=name, density=True, edgecolor="black", linewidth=0.4,
                )
                if len(subset) > 1:
                    subset.plot.kde(ax=ax, color=color, lw=2.4)
            ax.set_title(col.replace("_", " ").title(), fontsize=12)
            ax.legend(fontsize=9)
            ax.set_xlabel("")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature Distributions by Label (Histogram + KDE)", y=1.01)
        plt.tight_layout()
        _save(fig, out_dir / "02_feature_distributions.png")
    print("  [OK] Feature distributions")


# ──────────────────────────────────────────────────────────────────────────
# 3. Correlation heatmap
# ──────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame, cols: list[str], out_dir: Path = FIG_EDA,
) -> None:
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(
            figsize=(max(13, len(cols) * 0.55), max(11, len(cols) * 0.55))
        )
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.4, ax=ax,
            annot_kws={"size": 9, "weight": "bold"},
            cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        )
        ax.set_title("Feature Correlation Heatmap (Pearson r)")
        plt.tight_layout()
        _save(fig, out_dir / "03_correlation_heatmap.png")
        _save(fig, FIG_PUBLICATION / "fig_correlation_heatmap.pdf", dpi=DPI_PUBLICATION)
    print("  [OK] Correlation heatmap")


# ──────────────────────────────────────────────────────────────────────────
# 4. Correlation with target
# ──────────────────────────────────────────────────────────────────────────

def plot_correlation_with_target(
    df: pd.DataFrame, numeric_cols: list[str], out_dir: Path = FIG_EDA,
) -> None:
    cols = [c for c in numeric_cols if c in df.columns]
    corrs = df[cols + ["label"]].corr()["label"].drop("label").sort_values()

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(12, max(7, len(corrs) * 0.4)))
        colors = [COLOR_POS if v > 0 else COLOR_NEG for v in corrs.values]
        ax.barh(
            corrs.index, corrs.values,
            color=colors, edgecolor="black", linewidth=0.5,
        )
        ax.axvline(0, color="black", lw=1.0)
        ax.set_xlabel("Pearson r with Label")
        ax.set_title("Feature Correlation with Target Label (Ranked)")
        plt.tight_layout()
        _save(fig, out_dir / "04_correlation_with_target.png")
    print("  [OK] Correlation with target")


# ──────────────────────────────────────────────────────────────────────────
# 5. Box plots with Mann-Whitney
# ──────────────────────────────────────────────────────────────────────────

def plot_boxplots(
    df: pd.DataFrame, key_cols: list[str], out_dir: Path = FIG_EDA,
) -> None:
    ncols = 4
    key_cols = [c for c in key_cols if c in df.columns]
    nrows = int(np.ceil(len(key_cols) / ncols))

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.5 * nrows))
        axes = axes.ravel()
        for i, col in enumerate(key_cols):
            sns.boxplot(
                data=df, x="label", y=col, ax=axes[i],
                palette=COLORS,
                showfliers=True,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
                linewidth=1.0,
            )
            axes[i].set_xticklabels(["Non-Prog", "Prog"])
            axes[i].set_title(col.replace("_", " ").title(), fontsize=12)

            g0 = df[df["label"] == 0][col].dropna()
            g1 = df[df["label"] == 1][col].dropna()
            if len(g0) > 0 and len(g1) > 0:
                _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                axes[i].text(
                    0.5, 0.97, f"p={p:.4f} {sig}",
                    ha="center", va="top",
                    transform=axes[i].transAxes,
                    fontsize=11,
                    color="red" if p < 0.05 else "gray",
                    fontweight="bold",
                )

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Feature Distributions by Label — Box Plots (Mann–Whitney U p-values)", y=1.02)
        plt.tight_layout()
        _save(fig, out_dir / "05_boxplots_by_class.png")
    print("  [OK] Box plots")


# ──────────────────────────────────────────────────────────────────────────
# 6. Violin plots
# ──────────────────────────────────────────────────────────────────────────

def plot_violin_plots(
    df: pd.DataFrame, key_cols: list[str], out_dir: Path = FIG_EDA,
) -> None:
    ncols = 4
    key_cols = [c for c in key_cols if c in df.columns]
    nrows = int(np.ceil(len(key_cols) / ncols))

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.5 * nrows))
        axes = axes.ravel()
        for i, col in enumerate(key_cols):
            sns.violinplot(
                data=df, x="label", y=col, ax=axes[i],
                palette=COLORS, inner="quartile", cut=0, linewidth=1.0,
            )
            axes[i].set_xticklabels(["Non-Prog", "Prog"])
            axes[i].set_title(col.replace("_", " ").title(), fontsize=12)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature Violin Plots by Label", y=1.02)
        plt.tight_layout()
        _save(fig, out_dir / "06_violin_plots.png")
        _save(fig, FIG_PUBLICATION / "fig_violin_distributions.pdf", dpi=DPI_PUBLICATION)
    print("  [OK] Violin plots")


# ──────────────────────────────────────────────────────────────────────────
# 7. Pair plot
# ──────────────────────────────────────────────────────────────────────────

def plot_pairplot(
    df: pd.DataFrame, top_features: list[str], out_dir: Path = FIG_EDA,
) -> None:
    top_features = [c for c in top_features if c in df.columns]
    if not top_features:
        return
    pair_df = df[top_features + ["label"]].dropna().copy()
    pair_df["Class"] = pair_df["label"].map({0: "Non-Progressive", 1: "Progressive"})

    g = sns.pairplot(
        pair_df.drop("label", axis=1),
        hue="Class",
        palette={"Non-Progressive": COLOR_NEG, "Progressive": COLOR_POS},
        diag_kind="kde", corner=True,
        plot_kws={"alpha": 0.45, "s": 18, "edgecolor": "white", "linewidth": 0.4},
        diag_kws={"fill": True, "alpha": 0.6, "linewidth": 1.5},
    )
    g.figure.suptitle("Pair Plot — Top Clinical Features", y=1.02, fontsize=18, fontweight="bold")
    _save(g.figure, out_dir / "07_pairplot_top_features.png")
    print("  [OK] Pair plot")


# ──────────────────────────────────────────────────────────────────────────
# 8. Statistical significance
# ──────────────────────────────────────────────────────────────────────────

def plot_statistical_significance(
    df: pd.DataFrame, numeric_cols: list[str], out_dir: Path = FIG_EDA,
) -> None:
    results = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        g0 = df[df["label"] == 0][col].dropna()
        g1 = df[df["label"] == 1][col].dropna()
        if len(g0) == 0 or len(g1) == 0:
            continue
        u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        d = _cohen_d(g0, g1)
        results.append({
            "feature": col,
            "u_stat": u,
            "p_value": p,
            "neg_log_p": -np.log10(max(p, 1e-300)),
            "effect_size_d": d,
        })

    if not results:
        return
    df_stat = pd.DataFrame(results).sort_values("neg_log_p", ascending=True)
    df_stat.to_csv(FIG_EDA.parent.parent / "reports" / "statistical_tests.csv", index=False)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(df_stat) * 0.35)))

        colors = [COLOR_POS if p < 0.05 else COLOR_NEG for p in df_stat["p_value"]]
        axes[0].barh(
            df_stat["feature"], df_stat["neg_log_p"],
            color=colors, edgecolor="black", linewidth=0.5,
        )
        threshold = -np.log10(0.05)
        axes[0].axvline(threshold, color="black", ls="--", lw=1.2, label="p = 0.05")
        axes[0].set_xlabel("−log₁₀(p-value)")
        axes[0].set_title("Statistical Significance (Mann–Whitney U)")
        axes[0].legend()

        d_colors = [COLOR_POS if abs(d) > 0.5 else COLOR_NEG for d in df_stat["effect_size_d"]]
        axes[1].barh(
            df_stat["feature"], df_stat["effect_size_d"].abs(),
            color=d_colors, edgecolor="black", linewidth=0.5,
        )
        axes[1].axvline(0.2, color="orange", ls="--", lw=1, label="Small (0.2)")
        axes[1].axvline(0.5, color="red", ls="--", lw=1, label="Medium (0.5)")
        axes[1].axvline(0.8, color="darkred", ls="--", lw=1, label="Large (0.8)")
        axes[1].set_xlabel("|Cohen's d|")
        axes[1].set_title("Effect Size (Cohen's d)")
        axes[1].legend(fontsize=10)

        plt.tight_layout()
        _save(fig, out_dir / "08_statistical_significance.png")
    print("  [OK] Statistical significance")


# ──────────────────────────────────────────────────────────────────────────
# 9. Clinical groupings
# ──────────────────────────────────────────────────────────────────────────

def plot_clinical_groupings(df: pd.DataFrame, out_dir: Path = FIG_EDA) -> None:
    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

        if "corneal_risk_score" in df.columns:
            sns.countplot(
                data=df, x="corneal_risk_score", hue="label", ax=axes[0],
                palette=COLORS, edgecolor="black", linewidth=0.5,
            )
            axes[0].set_title("Corneal Risk Score × Label")
            axes[0].legend(title="Label", labels=["Non-Prog", "Prog"], fontsize=11)
            axes[0].set_xlabel("Corneal Risk Score (0–4)")

        if "astig_axis_type" in df.columns:
            sns.countplot(
                data=df, x="astig_axis_type", hue="label", ax=axes[1],
                palette=COLORS, order=["WTR", "ATR", "Oblique"],
                edgecolor="black", linewidth=0.5,
            )
            axes[1].set_title("Astigmatism Axis Type × Label")
            axes[1].legend(title="Label", labels=["Non-Prog", "Prog"], fontsize=11)

        if "ectasia_risk_score" in df.columns:
            sns.boxplot(
                data=df, x="label", y="ectasia_risk_score", ax=axes[2],
                palette=COLORS, linewidth=1.0,
            )
            axes[2].set_xticklabels(["Non-Progressive", "Progressive"])
            axes[2].set_title("Ectasia Risk Score Distribution")

        fig.suptitle("Clinical Grouping Analysis", y=1.02)
        plt.tight_layout()
        _save(fig, out_dir / "09_clinical_grouping_analysis.png")
    print("  [OK] Clinical groupings")


# ──────────────────────────────────────────────────────────────────────────
# 10. Engineered features by class
# ──────────────────────────────────────────────────────────────────────────

def plot_engineered_features(
    df: pd.DataFrame, eng_cols: list[str], out_dir: Path = FIG_EDA,
) -> None:
    eng_available = [c for c in eng_cols if c in df.columns]
    if not eng_available:
        return

    ncols = 4
    nrows = int(np.ceil(len(eng_available) / ncols))

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows))
        axes = axes.ravel()
        for i, col in enumerate(eng_available):
            sns.boxplot(
                data=df, x="label", y=col, ax=axes[i],
                palette=COLORS, linewidth=0.9,
            )
            axes[i].set_xticklabels(["Non-Prog", "Prog"])
            axes[i].set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")
            axes[i].tick_params(labelsize=9)
            axes[i].set_xlabel("")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Engineered Features vs Label", y=1.01)
        plt.tight_layout()
        _save(fig, out_dir / "10_engineered_features_by_class.png")
    print("  [OK] Engineered features")


# ──────────────────────────────────────────────────────────────────────────
# 11. Descriptive statistics table
# ──────────────────────────────────────────────────────────────────────────

def plot_descriptive_stats(
    df: pd.DataFrame, numeric_cols: list[str], out_dir: Path = FIG_EDA,
) -> None:
    cols = [c for c in numeric_cols if c in df.columns]
    s0 = df[df["label"] == 0][cols].describe().T.round(2)
    s1 = df[df["label"] == 1][cols].describe().T.round(2)

    s0.columns = [f"NP_{c}" for c in s0.columns]
    s1.columns = [f"P_{c}" for c in s1.columns]
    combined = pd.concat([
        s0[["NP_mean", "NP_std", "NP_min", "NP_max"]],
        s1[["P_mean", "P_std", "P_min", "P_max"]],
    ], axis=1)
    combined.columns = ["NP mean", "NP std", "NP min", "NP max",
                        "P mean", "P std", "P min", "P max"]

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(16, max(5, len(combined) * 0.45)))
        ax.axis("off")
        tbl = ax.table(
            cellText=combined.round(3).values,
            rowLabels=combined.index,
            colLabels=combined.columns,
            cellLoc="center", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 1.7)
        for j in range(len(combined.columns)):
            cell = tbl[(0, j)]
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(weight="bold")
        ax.set_title(
            "Descriptive Statistics: Non-Progressive vs Progressive",
            fontsize=15, fontweight="bold", pad=18,
        )
        _save(fig, out_dir / "11_descriptive_stats_table.png")
    print("  [OK] Descriptive statistics table")


# ──────────────────────────────────────────────────────────────────────────
# Plotly interactive charts
# ──────────────────────────────────────────────────────────────────────────

def plot_interactive_3d(df: pd.DataFrame, out_dir: Path = FIG_EDA) -> None:
    if not PLOTLY:
        return
    label_map = {0: "Non-Progressive", 1: "Progressive"}
    fig = px.scatter_3d(
        df, x="kmax_value_D", y="astig_abs", z="pachy_central_um",
        color=df["label"].map(label_map),
        color_discrete_map={"Non-Progressive": COLOR_NEG, "Progressive": COLOR_POS},
        opacity=0.65,
        title="3D Feature Space — Kmax × Astigmatism × Central Pachymetry",
        labels={"color": "Label"},
    )
    fig.update_traces(marker_size=3.5)
    fig.write_html(str(out_dir / "12_3d_scatter.html"))
    print("  [OK] Interactive 3D scatter")


def plot_interactive_sunburst(df: pd.DataFrame, out_dir: Path = FIG_EDA) -> None:
    if not PLOTLY or "age_group" not in df.columns:
        return
    fig = px.sunburst(
        df, path=["gender", "age_group", "label"],
        title="Hierarchical View: Gender → Age Group → Label",
        color="label", color_continuous_scale="RdBu_r",
    )
    fig.write_html(str(out_dir / "13_sunburst.html"))
    print("  [OK] Interactive sunburst")


# ──────────────────────────────────────────────────────────────────────────
# Master function
# ──────────────────────────────────────────────────────────────────────────

def run_full_eda(
    df: pd.DataFrame,
    numeric_cols: list[str],
    engineered_cols: list[str] | None = None,
) -> None:
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    key_cols = [c for c in [
        "kmax_value_D", "astig_abs", "pachy_central_um", "pachy_thinnest_um",
        "asphericity_anterior", "asphericity_posterior",
        "pachy_diff", "corneal_power_index",
        "kisa_proxy", "ectasia_risk_score",
    ] if c in df.columns]

    top_features = [c for c in [
        "kmax_value_D", "astig_abs", "pachy_central_um", "asphericity_anterior",
    ] if c in df.columns]

    all_num = [c for c in numeric_cols if c in df.columns]
    corr_cols = all_num + [c for c in [
        "pachy_diff", "pachy_ratio", "astig_abs", "corneal_power_index",
        "asphericity_diff", "corneal_risk_score", "kisa_proxy", "label",
    ] if c in df.columns]

    plot_label_distribution(df)
    plot_feature_distributions(df, all_num)
    plot_correlation_heatmap(df, corr_cols)
    plot_correlation_with_target(df, all_num)
    plot_boxplots(df, key_cols)
    plot_violin_plots(df, key_cols)
    plot_pairplot(df, top_features)
    plot_statistical_significance(df, all_num)
    plot_clinical_groupings(df)
    if engineered_cols:
        eng_display = [
            c for c in engineered_cols
            if c in df.columns and c not in ("gender_encoded", "eye_encoded")
        ]
        plot_engineered_features(df, eng_display)
    plot_descriptive_stats(df, all_num)
    plot_interactive_3d(df)
    plot_interactive_sunburst(df)

    print(f"\n[EDA complete] -> {FIG_EDA}")
