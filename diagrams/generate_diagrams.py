"""
generate_diagrams.py — Architectural & Methodological Diagrams
===============================================================
Generates publication-quality architectural diagrams using Matplotlib
(no external tools required — fully reproducible).

Diagrams produced:
  1. System Architecture Overview (end-to-end pipeline)
  2. Data Flow Diagram (raw -> engineered -> model -> inference)
  3. Feature Engineering Schema (annotated clinical feature groups)
  4. Multi-Agent Architecture (agent collaboration diagram)
  5. Model Ensemble Architecture (stacking diagram)
  6. Clinical Decision Support Workflow

All diagrams are saved as high-res PNG + PDF in outputs/figures/diagrams/.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import FIG_DIAGRAMS, DPI_PUBLICATION

DPI = DPI_PUBLICATION
OUT = FIG_DIAGRAMS

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "data":      "#1565C0",   # deep blue
    "process":   "#2E7D32",   # forest green
    "model":     "#4527A0",   # deep purple
    "output":    "#BF360C",   # deep orange
    "agent":     "#00695C",   # teal
    "meta":      "#F57F17",   # amber
    "bg":        "#FAFAFA",
    "box_bg":    "#FFFFFF",
    "arrow":     "#424242",
    "text_dark": "#212121",
    "text_light":"#FFFFFF",
    "highlight": "#E8F5E9",
}

PUB_RC = {
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
}


def _save(fig, filename: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    # PDF version
    pdf_path = OUT / filename.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {filename}")


def _fancybox(ax, x, y, w, h, text, color, fontsize=9, text_color="white",
              bold=False, style="round,pad=0.1"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=style,
        facecolor=color, edgecolor="white", linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold" if bold else "normal",
            zorder=4, wrap=True,
            multialignment="center")


def _arrow(ax, x0, y0, x1, y1, color="#424242", lw=1.5):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            mutation_scale=12,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )


# ── Diagram 1: System Architecture Overview ───────────────────────────────────

def diagram_system_architecture():
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(18, 11))
        fig.patch.set_facecolor(C["bg"])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 11)
        ax.axis("off")

        ax.text(9, 10.5, "AI-Based Myopia Progression Prediction — System Architecture",
                ha="center", va="center", fontsize=14, fontweight="bold", color=C["text_dark"])

        # ── Row 1: Input ─────────────────────────────────────────────────────
        # Raw Data
        _fancybox(ax, 2.5, 9.2, 3.5, 0.9, "Raw Clinical Data\n(1,454 patients)", C["data"], bold=True)
        # Synopsis / Domain Knowledge
        _fancybox(ax, 7.5, 9.2, 3.5, 0.9, "Domain Knowledge\n(Ophthalmology / Corneal Topography)", C["data"])
        # Research Objective
        _fancybox(ax, 13, 9.2, 3.8, 0.9, "Research Objective\nBinary Myopia Progression Prediction", C["data"])

        _arrow(ax, 2.5, 8.75, 2.5, 7.65)
        _arrow(ax, 7.5, 8.75, 7.5, 7.65)
        _arrow(ax, 4.25, 9.2, 6.75, 9.2)   # data -> domain

        # ── Row 2: Preprocessing ─────────────────────────────────────────────
        _fancybox(ax, 2.5, 7.3, 3.5, 0.8, "Data Cleaning\n(Dedup + Outlier Cap + Imputation)", C["process"])
        _fancybox(ax, 7.5, 7.3, 3.5, 0.8, "Feature Engineering\n(25 Clinical Features)", C["process"], bold=True)
        _fancybox(ax, 12.5, 7.3, 3.5, 0.8, "SMOTE Augmentation\n(Balanced Classes)", C["process"])

        _arrow(ax, 4.25, 7.3, 5.75, 7.3)
        _arrow(ax, 9.25, 7.3, 10.75, 7.3)
        _arrow(ax, 2.5, 6.9, 2.5, 5.85)
        _arrow(ax, 7.5, 6.9, 7.5, 5.85)
        _arrow(ax, 12.5, 6.9, 12.5, 5.85)

        # ── Row 3: Data Splits ───────────────────────────────────────────────
        _fancybox(ax, 2.5, 5.5, 2.8, 0.7, "Train Set (70%)\n~1,018 samples", C["meta"])
        _fancybox(ax, 7.5, 5.5, 2.8, 0.7, "Val Set (10%)\n~145 samples", C["meta"])
        _fancybox(ax, 12.5, 5.5, 2.8, 0.7, "Test Set (20%)\n~291 samples", C["meta"])

        # ── Row 4: Models ────────────────────────────────────────────────────
        _arrow(ax, 7.5, 5.15, 7.5, 4.35)
        models_x = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5]
        model_names = ["LR", "DT", "RF", "XGB", "LGB", "GB", "SVM", "MLP"]
        for mx, mn in zip(models_x, model_names):
            _fancybox(ax, mx, 4.0, 1.6, 0.65, mn, C["model"], fontsize=8)
            _arrow(ax, mx, 3.67, mx, 2.85)

        ax.text(8.5, 4.45, "Multi-Model Training (11 Classifiers + Stacking Ensemble)",
                ha="center", fontsize=9, color=C["model"], fontweight="bold")

        # ── Row 5: Evaluation ────────────────────────────────────────────────
        _fancybox(ax, 4.5, 2.5, 4.0, 0.75, "Cross-Validation\n5-Fold Stratified CV", C["agent"])
        _fancybox(ax, 9.5, 2.5, 3.5, 0.75, "Test Evaluation\nAUC, F1, Sens, Spec", C["agent"])
        _fancybox(ax, 14.5, 2.5, 3.0, 0.75, "Explainability\nSHAP Analysis", C["agent"])

        _arrow(ax, 4.5, 2.12, 4.5, 1.35)
        _arrow(ax, 9.5, 2.12, 9.5, 1.35)

        # ── Row 6: Output ────────────────────────────────────────────────────
        _fancybox(ax, 3, 1.0, 3.5, 0.7, "Best Model\n(Saved Artifact)", C["output"], bold=True)
        _fancybox(ax, 8.0, 1.0, 3.5, 0.7, "Publication Figures\n& Research Report", C["output"])
        _fancybox(ax, 13.5, 1.0, 4.0, 0.7, "Streamlit\nClinical Decision App", C["output"], bold=True)

        _arrow(ax, 9.5, 1.0, 11.25, 1.0)
        _arrow(ax, 4.75, 1.0, 6.25, 1.0)

        plt.tight_layout()
        _save(fig, "01_system_architecture.png")


# ── Diagram 2: Feature Engineering Schema ─────────────────────────────────────

def diagram_feature_engineering():
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(16, 14))
        fig.patch.set_facecolor(C["bg"])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 14)
        ax.axis("off")

        ax.text(8, 13.5, "Clinical Feature Engineering Schema",
                ha="center", fontsize=14, fontweight="bold", color=C["text_dark"])

        # Raw features central box
        _fancybox(ax, 8, 11.5, 7, 1.4, "RAW CLINICAL INPUT (14 Features)\n"
                  "age, gender, eye, astigmatism (D,°), Kmax (D,°),\n"
                  "pachymetry central/thinnest (μm, x, y), asphericity ant/post",
                  C["data"], bold=True, fontsize=10)

        groups = [
            (2.0, 9.0, C["process"], "PACHYMETRY\n(5 features)",
             "- Pachy diff (central - thinnest)\n- Pachy ratio\n- Thinnest displacement\n- Thin flag (<500μm)\n- Diff flag (>30μm)"),
            (6.0, 9.0, C["model"], "ASPHERICITY\n(5 features)",
             "- Asphericity diff (ant - post)\n- Asphericity ratio\n- Absolute sum\n- Oblate ant. flag\n- Surface imbalance"),
            (10.0, 9.0, C["agent"], "ASTIGMATISM\n(5 features)",
             "- Absolute magnitude |D|\n- Axis sin(2θ), cos(2θ)\n- Axis type: WTR/ATR/Oblique\n- High astig flag (>2.5D)\n- WTR axis flag"),
            (14.0, 9.0, C["meta"], "KERATOMETRY\n(3 features)",
             "- Kmax axis sin/cos\n- Kmax high flag (>47.2D)\n- Steep flag (>46.0D)"),
            (3.0, 5.5, "#5C4033", "COMPOSITE INDICES\n(4 features)",
             "- Corneal Power Index\n  Kmax x (1 + Q_ant)\n- Irregularity Index\n  |astig| x Δpachy / 100\n- KISA Proxy\n- Cone Location Magnitude"),
            (9.0, 5.5, "#1A237E", "INTERACTION TERMS\n(4 features)",
             "- Kmax x |astigmatism|\n- Age x Kmax\n- Pachy_central x |Q_ant|\n- Age x Pachy_central"),
            (13.5, 5.5, "#880E4F", "RISK SCORES\n(2 features)",
             "- Corneal Risk (0–4)\n  Kmax>46, astig>2.5,\n  pachy<510, Q>0\n- Ectasia Risk (0–7)\n  Extended ERSS proxy"),
        ]

        for gx, gy, gc, title, desc in groups:
            _fancybox(ax, gx, gy, 3.6, 1.4, title, gc, bold=True, fontsize=9)
            ax.text(gx, gy - 1.0, desc, ha="center", va="top",
                    fontsize=7.5, color=C["text_dark"], style="italic",
                    multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=gc, alpha=0.85))
            _arrow(ax, 8, 10.8, gx, gy + 0.7)

        # Final feature count
        _fancybox(ax, 8, 1.5, 9, 1.1,
                  "FINAL FEATURE MATRIX: 35 Features per Patient\n"
                  "(11 raw numeric + 2 encoded categorical + 22 engineered)",
                  C["output"], bold=True, fontsize=11)
        for gx, gy, *_ in groups:
            _arrow(ax, gx, gy - 1.75, 8, 2.05)

        plt.tight_layout()
        _save(fig, "02_feature_engineering_schema.png")


# ── Diagram 3: Multi-Agent Architecture ───────────────────────────────────────

def diagram_multi_agent():
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.patch.set_facecolor("#F3F4F6")
        ax.set_facecolor("#F3F4F6")
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis("off")

        ax.text(8, 9.5, "Multi-Agent Research Pipeline Architecture",
                ha="center", fontsize=14, fontweight="bold", color=C["text_dark"])

        # Orchestrator
        _fancybox(ax, 8, 8.5, 6, 0.85, "ORCHESTRATOR AGENT\n(run_pipeline.py)", "#212121",
                  bold=True, fontsize=11)

        # 6 specialist agents
        agents = [
            (1.5, 6.8, C["data"],    "AGENT 1\nData Analyzer",
             "- Load & validate data\n- Inspection report\n- Schema check"),
            (4.5, 6.8, C["process"], "AGENT 2\nPreprocessor",
             "- Cleaning & imputation\n- Feature engineering\n- Outlier capping"),
            (7.5, 6.8, "#1565C0",   "AGENT 3\nEDA Visualizer",
             "- 15 EDA charts\n- Statistical tests\n- Interactive plots"),
            (10.5, 6.8, C["model"],  "AGENT 4\nModel Trainer",
             "- 11 models + ensemble\n- 5-fold CV\n- SMOTE augmentation"),
            (13.5, 6.8, C["agent"],  "AGENT 5\nEvaluator",
             "- ROC, PR, CM\n- Calibration curves\n- Learning curves"),
            (8, 4.2, C["output"],    "AGENT 6\nInference & App",
             "- SHAP explanations\n- Best model export\n- Streamlit app"),
        ]

        for ax_pos, ay_pos, ac, title, desc in agents:
            _fancybox(ax, ax_pos, ay_pos, 2.7, 0.85, title, ac, bold=True, fontsize=9)
            ax.text(ax_pos, ay_pos - 0.7, desc, ha="center", va="top",
                    fontsize=7.5, color=C["text_dark"],
                    multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=ac, alpha=0.9))
            _arrow(ax, 8, 8.08, ax_pos, ay_pos + 0.42)

        # Shared data store
        _fancybox(ax, 8, 2.0, 8, 0.75, "SHARED OUTPUT STORE\n"
                  "outputs/ (models, figures, reports, splits)",
                  "#37474F", bold=True)
        for ax_pos, ay_pos, *_ in agents:
            _arrow(ax, ax_pos, ay_pos - 1.1, 8, 2.38)

        # Publication outputs
        _fancybox(ax, 3, 0.8, 3.5, 0.65, "Research Paper\nPublication Figures", C["output"])
        _fancybox(ax, 8, 0.8, 3, 0.65, "MPhil Thesis\n& Notebook", C["output"])
        _fancybox(ax, 13, 0.8, 3.5, 0.65, "Streamlit\nClinical App", C["output"])
        _arrow(ax, 5.5, 1.62, 3, 1.13)
        _arrow(ax, 8, 1.62, 8, 1.13)
        _arrow(ax, 10.5, 1.62, 13, 1.13)

        plt.tight_layout()
        _save(fig, "03_multi_agent_architecture.png")


# ── Diagram 4: Ensemble Architecture ─────────────────────────────────────────

def diagram_ensemble_architecture():
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor(C["bg"])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis("off")

        ax.text(7, 9.5, "Stacking Ensemble Model Architecture",
                ha="center", fontsize=14, fontweight="bold", color=C["text_dark"])

        # Input
        _fancybox(ax, 7, 8.6, 7, 0.8, "Input: Scaled Feature Vector (35 features)", C["data"], bold=True)

        # Level-0 base learners
        base = [
            (2, 6.8, "Random Forest\n(300 trees)"),
            (5, 6.8, "XGBoost\n(300 trees, η=0.05)"),
            (9, 6.8, "LightGBM\n(300 trees, η=0.05)"),
            (12, 6.8, "SVM-RBF\n(C=10, γ=scale)"),
        ]
        for bx, by, bt in base:
            _fancybox(ax, bx, by, 2.5, 0.9, bt, C["model"], fontsize=9)
            _arrow(ax, 7, 8.2, bx, by + 0.45)

        # Probability outputs
        ax.text(7, 5.85, "Level-0 Probability Outputs\n[P(prog|X) from each base learner]",
                ha="center", fontsize=9, color=C["agent"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["highlight"],
                          edgecolor=C["agent"], alpha=0.9))
        for bx, by, _ in base:
            _arrow(ax, bx, by - 0.45, bx, 5.3)
            _arrow(ax, bx, 5.3, 7, 5.3)

        # Cross-val strategy
        _fancybox(ax, 7, 4.5, 5, 0.7, "5-Fold Cross-Validation\n(OOF predictions to prevent data leakage)", C["agent"])
        _arrow(ax, 7, 5.1, 7, 4.85)
        _arrow(ax, 7, 4.15, 7, 3.65)

        # Meta-learner
        _fancybox(ax, 7, 3.3, 4.5, 0.7,
                  "Meta-Learner: Logistic Regression\n(C=0.1, learns optimal combination)",
                  "#BF360C", bold=True)
        _arrow(ax, 7, 2.95, 7, 2.25)

        # Final output
        _fancybox(ax, 7, 1.9, 5, 0.65,
                  "Final Prediction\nP(Myopia Progression) in [0, 1]",
                  C["output"], bold=True, fontsize=11)

        # Other ensembles note
        ax.text(7, 1.0, "Alternative: Soft-Voting (RF + XGB + LGB + MLP) - AdaBoost - GBM",
                ha="center", fontsize=8.5, color="gray", style="italic")

        plt.tight_layout()
        _save(fig, "04_ensemble_architecture.png")


# ── Diagram 5: Clinical Decision Workflow ─────────────────────────────────────

def diagram_clinical_workflow():
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(14, 11))
        fig.patch.set_facecolor(C["bg"])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 11)
        ax.axis("off")

        ax.text(7, 10.5, "Clinical Decision Support Workflow",
                ha="center", fontsize=14, fontweight="bold", color=C["text_dark"])

        steps = [
            (7, 9.3, C["data"],    "Patient Presents\nCorneal Topography Measurement"),
            (7, 7.8, C["process"], "Extract Clinical Parameters\n"
             "(Kmax, Pachymetry, Astigmatism, Asphericity)"),
            (7, 6.3, C["agent"],   "AI Feature Engineering\n(25 derived clinical indices)"),
            (7, 4.8, C["model"],   "Best Model Inference\n"
             "(Stacking Ensemble / XGBoost)"),
            (7, 3.3, C["meta"],    "Risk Probability Output\nP(Myopia Progression)"),
        ]
        for x, y, c, txt in steps:
            _fancybox(ax, x, y, 7.5, 0.85, txt, c, fontsize=10, bold=True)
            if y > 3.3:
                _arrow(ax, x, y - 0.42, x, y - 0.73 - 0.42)

        # Decision diamond approximation
        _arrow(ax, 7, 2.88, 7, 2.45)
        diamond_pts = np.array([[7, 2.4], [8.5, 2.0], [7, 1.6], [5.5, 2.0], [7, 2.4]])
        ax.fill(diamond_pts[:, 0], diamond_pts[:, 1], color=C["meta"], alpha=0.9)
        ax.text(7, 2.0, "P > 0.5?", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

        # Yes / No branches
        _fancybox(ax, 11, 1.0, 3.5, 0.75,
                  "HIGH RISK\nRefer for specialist review\nConsider intervention",
                  COLOR_POS := "#BF360C", bold=True)
        _fancybox(ax, 3, 1.0, 3.5, 0.75,
                  "LOW RISK\nRoutine monitoring\n6-month follow-up",
                  "#1B5E20", bold=True)
        ax.annotate("YES", xy=(9.25, 1.3), xytext=(8.5, 1.7),
                    arrowprops=dict(arrowstyle="-|>", color="#BF360C", lw=1.5),
                    fontsize=9, color="#BF360C", fontweight="bold")
        ax.annotate("NO", xy=(4.75, 1.3), xytext=(5.5, 1.7),
                    arrowprops=dict(arrowstyle="-|>", color="#1B5E20", lw=1.5),
                    fontsize=9, color="#1B5E20", fontweight="bold")

        # SHAP note
        ax.text(7, 0.25, "SHAP explanations provided for clinical transparency & interpretability",
                ha="center", fontsize=8.5, color="gray", style="italic")

        plt.tight_layout()
        _save(fig, "05_clinical_decision_workflow.png")


# ── Diagram 6: Data Pipeline ──────────────────────────────────────────────────

def diagram_data_pipeline():
    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(18, 5))
        fig.patch.set_facecolor(C["bg"])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 5)
        ax.axis("off")

        ax.text(9, 4.6, "Data Processing Pipeline",
                ha="center", fontsize=13, fontweight="bold", color=C["text_dark"])

        stages = [
            (1.5, C["data"],    "Raw Data\n1,454 rows\n14 columns"),
            (4.0, C["process"], "Cleaning\nDedup\nOutlier Cap"),
            (6.5, C["process"], "Encoding\ngender->0/1\neye->0/1"),
            (9.0, C["process"], "Feature Eng.\n25 new\nfeatures"),
            (11.5, C["model"],  "Splitting\n70/10/20\nStratified"),
            (14.0, C["model"],  "SMOTE\nBalanced\ntrain set"),
            (16.5, C["output"], "Scaled\nInput\nMatrix"),
        ]
        for x, c, txt in stages:
            _fancybox(ax, x, 2.5, 2.2, 1.8, txt, c, fontsize=9, bold=True)
            if x < 16.5:
                _arrow(ax, x + 1.1, 2.5, x + 1.8, 2.5)

        # Bottom labels
        for i, (x, c, txt) in enumerate(stages):
            ax.text(x, 1.4, f"Step {i+1}", ha="center", fontsize=7.5,
                    color=c, fontweight="bold")

        plt.tight_layout()
        _save(fig, "06_data_pipeline.png")


# ── Run all diagrams ──────────────────────────────────────────────────────────

def generate_all_diagrams():
    print("\n" + "=" * 60)
    print("GENERATING ARCHITECTURAL DIAGRAMS")
    print("=" * 60)
    diagram_system_architecture()
    diagram_feature_engineering()
    diagram_multi_agent()
    diagram_ensemble_architecture()
    diagram_clinical_workflow()
    diagram_data_pipeline()
    print(f"\n[Diagrams complete] -> {OUT}")


if __name__ == "__main__":
    generate_all_diagrams()
