"""
streamlit_app.py — AI Myopia Progression Prediction — Clinical Decision Support
================================================================================
Run: streamlit run app/streamlit_app.py

Pages
-----
  1. Patient Prediction  — enter clinical measurements, get AI risk score
  2. Model Performance   — view all evaluation metrics and charts
  3. Feature Importance  — SHAP-based explanations
  4. Dataset Explorer    — interactive EDA dashboard
  5. About               — research background and methodology
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT    = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Myopia Predictor",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #1565C0, #00897B);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title { text-align: center; color: #666; margin-bottom: 1.5rem; font-size: 0.95rem; }
    .risk-high {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        border-left: 5px solid #C62828; padding: 1.2rem; border-radius: 10px;
    }
    .risk-low {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-left: 5px solid #2E7D32; padding: 1.2rem; border-radius: 10px;
    }
    .metric-box {
        background: #F5F7FA; border-radius: 10px;
        padding: 0.8rem; text-align: center; margin: 0.2rem;
        border: 1px solid #E0E0E0;
    }
    .metric-val { font-size: 1.6rem; font-weight: 800; color: #1565C0; }
    .metric-lbl { font-size: 0.75rem; color: #555; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model_path   = APP_DIR / "best_model.joblib"
    scaler_path  = APP_DIR / "scaler.joblib"
    feature_path = APP_DIR / "feature_columns.json"

    if not model_path.exists():
        return None, None, None

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_path) as f:
        features = json.load(f)
    return model, scaler, features


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    path = APP_DIR / "processed_data.csv"
    if path.exists():
        return pd.read_csv(path)
    raw = ROOT / "clinical_data_and_labels.csv"
    if raw.exists():
        return pd.read_csv(raw)
    return None


@st.cache_data(show_spinner="Loading results...")
def load_results():
    path = ROOT / "outputs" / "reports" / "model_results_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Feature engineering (must match pipeline) ─────────────────────────────────
def compute_engineered_features(row: dict) -> dict:
    """Apply the same feature engineering as the training pipeline."""
    import numpy as _np
    r = dict(row)

    # Encodings
    r["gender_encoded"] = 0 if r.get("gender", "f").lower() == "f" else 1
    r["eye_encoded"]    = 0 if r.get("eye", "OD").upper() == "OD" else 1

    # Pachymetry
    r["pachy_diff"] = r["pachy_central_um"] - r["pachy_thinnest_um"]
    r["pachy_ratio"] = (r["pachy_thinnest_um"] / r["pachy_central_um"]
                        if r["pachy_central_um"] != 0 else 1.0)
    r["pachy_thinnest_displacement"] = _np.sqrt(
        r["pachy_thinnest_x"] ** 2 + r["pachy_thinnest_y"] ** 2
    )
    r["pachy_thin_flag"]  = 1 if r["pachy_central_um"] < 500 else 0
    r["pachy_diff_flag"]  = 1 if r["pachy_diff"] > 30 else 0

    # Asphericity
    r["asphericity_diff"] = r["asphericity_anterior"] - r["asphericity_posterior"]
    denom = r["asphericity_posterior"] if r["asphericity_posterior"] != 0 else 1e-9
    r["asphericity_ratio"]   = r["asphericity_anterior"] / denom
    r["asphericity_abs_sum"] = abs(r["asphericity_anterior"]) + abs(r["asphericity_posterior"])
    r["anterior_oblate_flag"] = 1 if r["asphericity_anterior"] > 0 else 0

    # Astigmatism
    r["astig_abs"]      = abs(r["astig_value_D"])
    r["astig_axis_sin"] = _np.sin(2 * _np.radians(r["astig_axis_deg"]))
    r["astig_axis_cos"] = _np.cos(2 * _np.radians(r["astig_axis_deg"]))
    r["astig_wtr_flag"] = 1 if (r["astig_axis_deg"] <= 30 or r["astig_axis_deg"] > 150) else 0
    r["astig_high_flag"] = 1 if r["astig_abs"] > 2.5 else 0

    # Keratometry
    r["kmax_axis_sin"]  = _np.sin(2 * _np.radians(r["kmax_axis_deg"]))
    r["kmax_axis_cos"]  = _np.cos(2 * _np.radians(r["kmax_axis_deg"]))
    r["kmax_high_flag"] = 1 if r["kmax_value_D"] > 47.2 else 0
    r["kmax_steep_flag"]= 1 if r["kmax_value_D"] > 46.0 else 0

    # Composite indices
    r["corneal_power_index"] = r["kmax_value_D"] * (1 + r["asphericity_anterior"])
    r["corneal_irregularity_index"] = r["astig_abs"] * r["pachy_diff"] / 100.0
    k_comp = max(0, r["kmax_value_D"] - 45)
    r["kisa_proxy"] = k_comp * r["astig_abs"] * (r["pachy_thinnest_displacement"] * 10) / 100.0
    r["cone_location_magnitude_index"] = (
        r["pachy_thinnest_displacement"] * (1 - r["pachy_ratio"]) * r["kmax_value_D"]
    )

    # Interactions
    r["kmax_astig_interaction"] = r["kmax_value_D"] * r["astig_abs"]
    r["age_kmax_interaction"]   = r["age_years"] * r["kmax_value_D"]
    r["pachy_asph_interaction"] = r["pachy_central_um"] * abs(r["asphericity_anterior"])
    r["age_pachy_interaction"]  = r["age_years"] * r["pachy_central_um"]

    # Risk scores
    r["corneal_risk_score"] = int(
        (r["kmax_value_D"] > 46.0) + (r["astig_abs"] > 2.5)
        + (r["pachy_central_um"] < 510) + (r["asphericity_anterior"] > 0)
    )
    r["ectasia_risk_score"] = int(
        (r["kmax_value_D"] > 47.2) * 2
        + (r["pachy_central_um"] < 500) * 2
        + (r["pachy_diff"] > 30)
        + (r["astig_abs"] > 3.0)
        + (r["asphericity_anterior"] > 0.5)
        + (r["pachy_thinnest_displacement"] > 1.0)
        + (r["age_years"] < 25)
    )
    return r


# ── Sidebar navigation ────────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "Navigation",
    ["Patient Prediction", "Model Performance", "Dataset Explorer", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Research Project**")
st.sidebar.markdown("AI-Based Myopia Progression Prediction")
st.sidebar.markdown("MPhil Ophthalmology — 2024")
st.sidebar.markdown("Syed Ahmad Hassan")

model, scaler, feature_cols = load_artifacts()
df = load_data()
df_results = load_results()

model_loaded = model is not None

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: PATIENT PREDICTION
# ════════════════════════════════════════════════════════════════════════════
if page == "Patient Prediction":
    st.markdown('<div class="main-title">👁 AI Myopia Progression Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter corneal topography measurements to predict myopia progression risk</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.warning("Model not found. Please run `python run_pipeline.py` first to train and save the model.")
        st.stop()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Patient Demographics")
        age        = st.number_input("Age (years)", 5, 80, 25, step=1)
        gender     = st.selectbox("Gender", ["Female (f)", "Male (m)"])
        eye        = st.selectbox("Eye", ["Right Eye (OD)", "Left Eye (OS)"])

        st.subheader("Astigmatism")
        astig_val  = st.number_input("Astigmatism (D)", -12.0, 0.0, -1.5, step=0.1,
                                      help="Refractive astigmatism in diopters (negative value)")
        astig_axis = st.number_input("Astigmatism Axis (°)", 0, 180, 90, step=1)

    with col2:
        st.subheader("Keratometry")
        kmax_val   = st.number_input("Kmax (D)", 38.0, 70.0, 44.0, step=0.1,
                                      help="Maximum keratometry — normal < 47.2 D")
        kmax_axis  = st.number_input("Kmax Axis (°)", 0, 180, 90, step=1)

        st.subheader("Pachymetry — Thickness")
        pachy_c    = st.number_input("Central Pachymetry (μm)", 300, 700, 520, step=1,
                                      help="Central corneal thickness — normal > 500 μm")
        pachy_t    = st.number_input("Thinnest Pachymetry (μm)", 280, 700, 510, step=1)

    with col3:
        st.subheader("Pachymetry — Location")
        pachy_x    = st.number_input("Thinnest Point X (mm)", -5.0, 5.0, 0.0, step=0.1)
        pachy_y    = st.number_input("Thinnest Point Y (mm)", -5.0, 5.0, 0.0, step=0.1)

        st.subheader("Asphericity")
        asp_ant    = st.number_input("Anterior Asphericity (Q)", -1.5, 3.0, -0.25, step=0.01,
                                      help="Normal: Q < 0 (prolate). Q > 0 = oblate (risk)")
        asp_post   = st.number_input("Posterior Asphericity (Q)", -1.5, 3.0, -0.25, step=0.01)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Myopia Progression Risk", use_container_width=True, type="primary")

    if predict_btn:
        raw_input = {
            "age_years": age,
            "gender": "f" if "Female" in gender else "m",
            "eye": "OD" if "OD" in eye else "OS",
            "astig_value_D": astig_val,
            "astig_axis_deg": astig_axis,
            "kmax_value_D": kmax_val,
            "kmax_axis_deg": kmax_axis,
            "pachy_central_um": pachy_c,
            "pachy_thinnest_um": pachy_t,
            "pachy_thinnest_x": pachy_x,
            "pachy_thinnest_y": pachy_y,
            "asphericity_anterior": asp_ant,
            "asphericity_posterior": asp_post,
        }
        feat_dict = compute_engineered_features(raw_input)
        feat_vals = [feat_dict.get(col, 0.0) for col in feature_cols]
        X_input   = np.array(feat_vals).reshape(1, -1)
        X_scaled  = scaler.transform(X_input)

        prob_prog = float(model.predict_proba(X_scaled)[0][1])
        pred_class = 1 if prob_prog >= 0.5 else 0

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Risk Probability", f"{prob_prog:.1%}")
        with c2:
            st.metric("Prediction", "PROGRESSIVE ⚠" if pred_class == 1 else "NON-PROGRESSIVE ✓")
        with c3:
            risk_cat = "HIGH" if prob_prog >= 0.7 else ("MODERATE" if prob_prog >= 0.4 else "LOW")
            st.metric("Risk Category", risk_cat)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_prog * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Myopia Progression Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#C62828" if prob_prog > 0.5 else "#2E7D32"},
                "steps": [
                    {"range": [0, 30],  "color": "#E8F5E9"},
                    {"range": [30, 60], "color": "#FFF9C4"},
                    {"range": [60, 100],"color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75, "value": 50
                },
            },
        ))
        fig_gauge.update_layout(height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        if pred_class == 1:
            st.markdown(f"""
<div class="risk-high">
<h3>⚠ HIGH RISK — Myopia Progression Likely</h3>
<p>Probability: <strong>{prob_prog:.1%}</strong></p>
<p><strong>Clinical Recommendations:</strong></p>
<ul>
<li>Urgent referral to corneal specialist</li>
<li>Consider orthokeratology or myopia control intervention</li>
<li>3-month topography follow-up</li>
<li>Evaluate for keratoconus suspects</li>
</ul>
<p><em>Corneal Risk Score: {feat_dict.get('corneal_risk_score', 0)}/4 | Ectasia Risk: {feat_dict.get('ectasia_risk_score', 0)}/7</em></p>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="risk-low">
<h3>✓ LOW RISK — Stable Myopia Profile</h3>
<p>Probability: <strong>{prob_prog:.1%}</strong></p>
<p><strong>Clinical Recommendations:</strong></p>
<ul>
<li>Routine annual follow-up</li>
<li>Standard refractive management</li>
<li>Monitor for changes in Kmax and pachymetry</li>
</ul>
<p><em>Corneal Risk Score: {feat_dict.get('corneal_risk_score', 0)}/4 | Ectasia Risk: {feat_dict.get('ectasia_risk_score', 0)}/7</em></p>
</div>""", unsafe_allow_html=True)

        # Key clinical flags
        st.markdown("#### Key Clinical Indicators")
        flags = {
            "Kmax > 47.2 D (Keratoconus threshold)": feat_dict.get("kmax_high_flag", 0) == 1,
            "Kmax > 46.0 D (Steep cornea)": feat_dict.get("kmax_steep_flag", 0) == 1,
            "Pachy < 500 μm (Thin cornea)": feat_dict.get("pachy_thin_flag", 0) == 1,
            "Pachy diff > 30 μm (Thinning gradient)": feat_dict.get("pachy_diff_flag", 0) == 1,
            "Anterior Q > 0 (Oblate shape)": feat_dict.get("anterior_oblate_flag", 0) == 1,
            "|Astig| > 2.5 D (High astigmatism)": feat_dict.get("astig_high_flag", 0) == 1,
        }
        cols = st.columns(3)
        for i, (flag_name, is_set) in enumerate(flags.items()):
            with cols[i % 3]:
                icon = "🔴" if is_set else "🟢"
                st.write(f"{icon} {flag_name}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<div class="main-title">Model Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comprehensive evaluation of all trained classifiers</div>', unsafe_allow_html=True)

    if df_results is None:
        st.warning("Results not found. Please run `python run_pipeline.py` first.")
        st.stop()

    # Summary metrics for best model
    best = df_results.iloc[0]
    st.markdown("---")
    st.subheader(f"Best Model: {best['Model']}")
    cols = st.columns(6)
    for i, (metric, val) in enumerate([
        ("AUC-ROC", best.get("AUC-ROC", 0)),
        ("F1-Score", best.get("F1-Score", 0)),
        ("Sensitivity", best.get("Sensitivity", 0)),
        ("Specificity", best.get("Specificity", 0)),
        ("Precision", best.get("Precision", 0)),
        ("Accuracy", best.get("Accuracy", 0)),
    ]):
        with cols[i]:
            st.markdown(f"""
<div class="metric-box">
  <div class="metric-val">{val:.3f}</div>
  <div class="metric-lbl">{metric}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Interactive bar chart
    metrics = ["AUC-ROC", "F1-Score", "Sensitivity", "Specificity", "Accuracy"]
    available_metrics = [m for m in metrics if m in df_results.columns]
    df_plot = df_results.sort_values("AUC-ROC", ascending=False)

    fig_bar = go.Figure()
    colors_bar = ["#1565C0", "#2E7D32", "#BF360C", "#4527A0", "#F57F17"]
    for j, m in enumerate(available_metrics):
        if m in df_plot.columns:
            fig_bar.add_trace(go.Bar(
                name=m, x=df_plot["Model"], y=df_plot[m],
                text=df_plot[m].round(3), textposition="outside",
                marker_color=colors_bar[j % len(colors_bar)],
            ))
    fig_bar.update_layout(
        barmode="group", title="Model Performance Comparison",
        height=500, yaxis_range=[0, 1.1],
        legend=dict(orientation="h", y=1.05),
        xaxis_tickangle=-35,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap
    st.subheader("Performance Heatmap")
    heat_metrics = [m for m in ["AUC-ROC", "F1-Score", "Sensitivity", "Specificity", "Accuracy", "Precision"]
                    if m in df_results.columns]
    heat_data = df_results[["Model"] + heat_metrics].set_index("Model")
    fig_hm = px.imshow(
        heat_data, text_auto=".3f",
        color_continuous_scale="RdYlGn",
        zmin=0.5, zmax=1.0,
        title="Model × Metric Heatmap",
        aspect="auto",
    )
    fig_hm.update_layout(height=400)
    st.plotly_chart(fig_hm, use_container_width=True)

    # Full table
    st.subheader("Full Results Table")
    st.dataframe(df_results.style.background_gradient(
        subset=heat_metrics, cmap="RdYlGn", vmin=0.5, vmax=1.0
    ), use_container_width=True)

    # Evaluation figures
    st.subheader("Evaluation Figures")
    fig_dir = ROOT / "outputs" / "figures" / "evaluation"
    pub_dir = ROOT / "outputs" / "figures" / "publication"

    figs_to_show = {
        "ROC Curves": fig_dir / "roc_curves.png",
        "Precision-Recall Curves": fig_dir / "pr_curves.png",
        "Confusion Matrices": fig_dir / "confusion_matrices.png",
        "Calibration Curves": fig_dir / "calibration_curves.png",
        "Feature Importance": fig_dir / "feature_importance.png",
        "Learning Curves": fig_dir / "learning_curves.png",
    }
    for title, fpath in figs_to_show.items():
        if fpath.exists():
            with st.expander(title):
                st.image(str(fpath), use_column_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Explorer":
    st.markdown('<div class="main-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive exploration of the clinical dataset</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset not found.")
        st.stop()

    label_map = {0: "Non-Progressive", 1: "Progressive"}
    df_display = df.copy()
    if "label" in df_display.columns:
        df_display["label_name"] = df_display["label"].map(label_map)

    # Summary
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(df))
    if "label" in df.columns:
        c2.metric("Non-Progressive", int((df["label"] == 0).sum()))
        c3.metric("Progressive", int((df["label"] == 1).sum()))
        c4.metric("Class Balance", f"{df['label'].mean():.1%} positive")

    # Label distribution
    if "label" in df.columns:
        fig_pie = px.pie(
            names=[label_map[k] for k in df["label"].value_counts().index],
            values=df["label"].value_counts().values,
            color_discrete_sequence=["#1565C0", "#BF360C"],
            title="Label Distribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Feature selection for scatter
    st.subheader("Feature Relationships")
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ("label", "gender_encoded", "eye_encoded")]
    col_x = st.selectbox("X axis", num_cols, index=num_cols.index("kmax_value_D") if "kmax_value_D" in num_cols else 0)
    col_y = st.selectbox("Y axis", num_cols, index=num_cols.index("pachy_central_um") if "pachy_central_um" in num_cols else 1)

    color_col = "label_name" if "label_name" in df_display.columns else None
    fig_scatter = px.scatter(
        df_display, x=col_x, y=col_y, color=color_col,
        color_discrete_map={"Non-Progressive": "#1565C0", "Progressive": "#BF360C"},
        opacity=0.6, hover_data=["age_years", "gender"] if "gender" in df.columns else None,
        title=f"{col_x} vs {col_y}",
        marginal_x="histogram", marginal_y="histogram",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_cols = [c for c in num_cols if c in df.columns][:15]
    corr = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix",
        aspect="auto",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Raw data table
    with st.expander("View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown('<div class="main-title">About This Research</div>', unsafe_allow_html=True)

    st.markdown("""
---
## Research Background

This application is part of the MPhil research:

> **"Development and Validation of an AI-Based Model for Myopia Progression Prediction Using Clinical Corneal Topography Data"**
>
> Syed Ahmad Hassan (2024-MPhil-OP-037)

### Clinical Context

Myopia (short-sightedness) is the leading cause of visual impairment worldwide, affecting ~2.6 billion people.
High myopia (≥ −6 D) carries significant risks:
- Retinal detachment
- Glaucoma
- Myopic maculopathy (leading cause of blindness in Asia)

**Early identification of progressive myopia is critical** for timely intervention
(orthokeratology, atropine therapy, refractive surgery eligibility).

---
## Dataset

| Property | Value |
|----------|-------|
| Total samples | 1,454 patients |
| Features (raw) | 14 clinical parameters |
| Features (engineered) | 35 (after feature engineering) |
| Label | Binary — Progressive (1) vs Non-Progressive (0) |
| Data type | Clinical corneal topography |

**Key clinical measurements:**
- **Kmax** — Maximum keratometry (corneal curvature)
- **Pachymetry** — Central and thinnest corneal thickness
- **Asphericity** — Anterior/posterior corneal shape
- **Astigmatism** — Magnitude and axis

---
## Methodology

### 1. Feature Engineering (25 derived features)
- Pachymetry differential and ratio
- KISA proxy index (keratoconus screening composite)
- Cone Location Magnitude Index
- Ectasia Risk Score (7-factor ERSS-inspired)
- Cyclic encoding of astigmatism/Kmax axes

### 2. Data Augmentation
- SMOTE (Synthetic Minority Over-sampling Technique)
- Addresses class imbalance without introducing real data

### 3. Multi-Model Comparison (11 classifiers)
Logistic Regression, Decision Tree, Random Forest, Extra Trees,
Gradient Boosting, XGBoost, LightGBM, AdaBoost, SVM, KNN, MLP

### 4. Stacking Ensemble
RF + XGB + LGB + SVM as base learners with Logistic Regression meta-learner

### 5. Explainability
SHAP (SHapley Additive exPlanations) for clinical transparency

---
## Key References

1. Rabinowitz, Y.S. (2002). KISA% Index. *Cornea*, 21(S1), S60.
2. Randleman, J.B. et al. (2008). Ectasia risk scoring system. *JRCS*, 24(5), 1-12.
3. Chawla, N.V. et al. (2002). SMOTE. *JAIR*, 16, 321-357.
4. Lundberg, S.M. & Lee, S-I. (2017). SHAP. *NeurIPS*.
5. Belin, M.W. et al. (2011). Belin/Ambrosio enhanced ectasia display. *JRCS*.

---
*This tool is for research purposes only and does not replace clinical judgement.*
""")

    # Show architectural diagrams if available
    diag_dir = ROOT / "outputs" / "figures" / "diagrams"
    if diag_dir.exists():
        diag_files = sorted(diag_dir.glob("*.png"))
        if diag_files:
            st.markdown("---")
            st.subheader("Architectural Diagrams")
            for d in diag_files:
                with st.expander(d.stem.replace("_", " ").title()):
                    st.image(str(d), use_column_width=True)
