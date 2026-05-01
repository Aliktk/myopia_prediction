"""
app/streamlit_app.py — Clinical Decision Support Web App
==========================================================
A theme-safe, multi-page Streamlit interface for the myopia progression
predictor. Uses CSS variables that adapt to light and dark themes so
text contrast remains readable in either mode.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# Path setup so app can import src/* modules
# ──────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.feature_engineering import engineer_all_features  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Myopia Progression Predictor",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Theme-safe CSS — uses Streamlit's CSS variables and prefers-color-scheme
# Avoids hard-coded white backgrounds with white text bug.
# ──────────────────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
    :root {
        --card-bg: rgba(0, 0, 0, 0.03);
        --card-border: rgba(0, 0, 0, 0.10);
        --text-strong: #1A1A1A;
        --text-muted: #555;
        --primary: #1565C0;
        --success: #2E7D32;
        --warning: #E65100;
        --danger: #C62828;
        --info-bg: rgba(21, 101, 192, 0.08);
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.15);
            --text-strong: #FFFFFF;
            --text-muted: #BBB;
            --info-bg: rgba(100, 181, 246, 0.10);
        }
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 1.6rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.92) !important;
        margin: 0.4rem 0 0 0;
        font-size: 1.0rem;
    }

    /* Generic card — adapts to theme */
    .info-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin: 0.6rem 0;
    }
    .info-card h4 {
        margin: 0 0 0.5rem 0;
        color: var(--text-strong) !important;
        font-size: 1.05rem;
    }
    .info-card p {
        margin: 0;
        color: var(--text-muted) !important;
        font-size: 0.92rem;
    }

    /* Result boxes — coloured BG with explicit white text (always readable) */
    .result-box {
        padding: 1.4rem 1.6rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.18);
    }
    .result-box * { color: #FFFFFF !important; }
    .result-low      { background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%); }
    .result-moderate { background: linear-gradient(135deg, #E65100 0%, #BF360C 100%); }
    .result-high     { background: linear-gradient(135deg, #C62828 0%, #8E0000 100%); }

    /* Risk flag chips */
    .flag-chip {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 14px;
        font-size: 0.82rem;
        font-weight: 600;
        background: var(--info-bg);
        color: var(--text-strong) !important;
        border: 1px solid var(--card-border);
    }
    .flag-chip.danger { background: rgba(198, 40, 40, 0.12); color: #C62828 !important; border-color: rgba(198, 40, 40, 0.4); }
    .flag-chip.warning { background: rgba(230, 81, 0, 0.12); color: #E65100 !important; border-color: rgba(230, 81, 0, 0.4); }
    .flag-chip.success { background: rgba(46, 125, 50, 0.12); color: #2E7D32 !important; border-color: rgba(46, 125, 50, 0.4); }

    /* Metric cards — theme-safe */
    [data-testid="stMetricValue"] {
        color: var(--text-strong) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255, 193, 7, 0.10);
        border-left: 4px solid #FFA000;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: var(--text-strong) !important;
        margin: 1rem 0;
    }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Artifact loaders (cached)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts() -> tuple[object, object, list[str]] | tuple[None, None, None]:
    try:
        model = joblib.load(APP_DIR / "best_model.joblib")
        scaler = joblib.load(APP_DIR / "scaler.joblib")
        with open(APP_DIR / "feature_columns.json") as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except FileNotFoundError as e:
        st.error(f"Required artifact missing: {e}. Run `python run_pipeline.py` first.")
        return None, None, None


@st.cache_data
def load_processed_data() -> pd.DataFrame | None:
    csv = APP_DIR / "processed_data.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return None


@st.cache_data
def load_results_summary() -> pd.DataFrame | None:
    csv = ROOT / "outputs" / "reports" / "model_results_summary.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return None


model, scaler, feature_cols = load_artifacts()


# ──────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👁 Navigation")
    page = st.radio(
        "Page",
        ["Patient Prediction", "Model Performance", "Dataset Explorer", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**System Status**")
    if model is not None:
        st.success(f"Model: `{type(model).__name__}` ✓")
        st.success(f"Features: {len(feature_cols)}")
    else:
        st.error("Model not loaded")
    st.markdown("---")
    st.caption("Research project — MPhil Ophthalmology")
    st.caption("v2.0.0 — combined dataset (1,642 rows)")


# ──────────────────────────────────────────────────────────────────────────
# Hero banner
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>AI Myopia Progression Predictor</h1>
        <p>Clinical Decision Support — Corneal Topography-Based Risk Stratification</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────
def predict_progression(inputs: dict) -> tuple[float, np.ndarray]:
    """Apply feature engineering, scale, and predict."""
    df = pd.DataFrame([inputs])
    df_eng = engineer_all_features(df)
    X = df_eng[feature_cols].values
    X_scaled = scaler.transform(X)
    proba = float(model.predict_proba(X_scaled)[0][1])
    return proba, X_scaled


def risk_band(proba: float) -> tuple[str, str, str]:
    if proba < 0.30:
        return "LOW RISK", "result-low", "Continue routine monitoring."
    if proba < 0.65:
        return "MODERATE RISK", "result-moderate", "Recommend closer follow-up and consider preventive measures."
    return "HIGH RISK", "result-high", "Recommend immediate clinical review and consider intervention."


# ──────────────────────────────────────────────────────────────────────────
# PAGE 1 — Patient Prediction
# ──────────────────────────────────────────────────────────────────────────
if page == "Patient Prediction":
    st.subheader("Enter Clinical Measurements")
    st.markdown(
        '<div class="info-card"><h4>Required Inputs</h4>'
        '<p>All fields are corneal topography or refraction values. '
        'Hover over each field for clinical context.</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age (years)", 5, 100, 30, 1)
        gender = st.selectbox("Gender", ["f", "m"])
        eye = st.selectbox("Eye", ["OD", "OS"], help="OD = right, OS = left")

    with col2:
        st.markdown("**Refraction & Keratometry**")
        astig_value = st.number_input("Astigmatism (D)", -10.0, 10.0, -1.0, 0.25)
        astig_axis = st.number_input("Astigmatism Axis (°)", 0.0, 180.0, 90.0, 1.0)
        kmax_value = st.number_input("Kmax (D)", 30.0, 70.0, 44.5, 0.1)
        kmax_axis = st.number_input("Kmax Axis (°)", 0.0, 180.0, 90.0, 1.0)

    with col3:
        st.markdown("**Pachymetry & Asphericity**")
        pachy_central = st.number_input("Central Pachymetry (μm)", 350, 700, 540, 1)
        pachy_thinnest = st.number_input("Thinnest Pachymetry (μm)", 350, 700, 530, 1)
        pachy_x = st.number_input("Thinnest X (mm)", -3.0, 3.0, 0.0, 0.05)
        pachy_y = st.number_input("Thinnest Y (mm)", -3.0, 3.0, 0.0, 0.05)
        asph_ant = st.number_input("Anterior Asphericity (Q)", -2.0, 2.0, -0.20, 0.01)
        asph_post = st.number_input("Posterior Asphericity (Q)", -2.0, 2.0, -0.20, 0.01)

    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded — run the pipeline first.")
        else:
            inputs = {
                "age_years": age, "gender": gender, "eye": eye,
                "astig_value_D": astig_value, "astig_axis_deg": astig_axis,
                "kmax_value_D": kmax_value, "kmax_axis_deg": kmax_axis,
                "pachy_central_um": pachy_central, "pachy_thinnest_um": pachy_thinnest,
                "pachy_thinnest_x": pachy_x, "pachy_thinnest_y": pachy_y,
                "asphericity_anterior": asph_ant, "asphericity_posterior": asph_post,
            }

            proba, _ = predict_progression(inputs)
            band, css_class, recommendation = risk_band(proba)

            st.markdown(
                f'<div class="result-box {css_class}">'
                f'<h2 style="margin:0; font-size:1.8rem;">{band}</h2>'
                f'<p style="font-size:2.6rem; margin:0.4rem 0; font-weight:700;">{proba * 100:.1f}%</p>'
                f'<p style="font-size:0.95rem; margin:0;">Probability of progressive myopia</p>'
                f'<p style="margin-top:0.6rem; font-size:0.95rem;"><em>{recommendation}</em></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Probability gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 38}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1.4},
                    "bar": {"color": "#1565C0", "thickness": 0.32},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(46,125,50,0.18)"},
                        {"range": [30, 65], "color": "rgba(230,81,0,0.18)"},
                        {"range": [65, 100], "color": "rgba(198,40,40,0.20)"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.78,
                        "value": proba * 100,
                    },
                },
                title={"text": "Risk Score", "font": {"size": 18}},
            ))
            gauge.update_layout(height=300, margin=dict(l=18, r=18, t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)

            # Clinical flag chips
            st.markdown("#### Clinical Risk Flags")
            flags = []
            if kmax_value > 47.2:
                flags.append(("Kmax > 47.2 D (keratoconus suspect)", "danger"))
            elif kmax_value > 46.0:
                flags.append(("Kmax > 46.0 D (steep cornea)", "warning"))
            if pachy_central < 500:
                flags.append(("Central pachymetry < 500 μm (thin)", "danger"))
            elif pachy_central < 510:
                flags.append(("Central pachymetry < 510 μm (borderline)", "warning"))
            if pachy_central - pachy_thinnest > 30:
                flags.append((f"Pachy diff = {pachy_central - pachy_thinnest} μm (thinning gradient)", "warning"))
            if abs(astig_value) > 2.5:
                flags.append((f"|Astigmatism| = {abs(astig_value):.2f} D", "warning"))
            if asph_ant > 0:
                flags.append((f"Anterior Q = {asph_ant:.2f} > 0 (oblate, abnormal)", "warning"))

            if flags:
                chips_html = " ".join(
                    f'<span class="flag-chip {cls}">{txt}</span>' for txt, cls in flags
                )
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<span class="flag-chip success">✓ No clinical risk flags triggered</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="disclaimer">'
                '<strong>Clinical Disclaimer:</strong> This tool is for research and decision '
                'support only. All predictions must be reviewed by a qualified ophthalmologist '
                'before any clinical action is taken.'
                '</div>',
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────
# PAGE 2 — Model Performance
# ──────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.subheader("Model Performance Summary")
    summary = load_results_summary()
    if summary is None:
        st.warning("No results summary found. Run the pipeline first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        best = summary.iloc[0]
        c1.metric("Best Model", best["Model"])
        c2.metric("AUC-ROC", f"{best['AUC-ROC']:.4f}")
        c3.metric("F1-Score", f"{best['F1-Score']:.4f}")
        c4.metric("Sensitivity", f"{best['Sensitivity']:.4f}")

        st.markdown("#### Full Metrics Table")
        st.dataframe(
            summary.style.background_gradient(
                subset=["Accuracy", "Precision", "Sensitivity", "Specificity",
                        "F1-Score", "AUC-ROC"],
                cmap="RdYlGn", vmin=0.85, vmax=1.0,
            ).format(precision=4),
            use_container_width=True,
            height=460,
        )

        st.markdown("#### Visual Summary")
        eval_dir = ROOT / "outputs" / "figures" / "evaluation"
        figs = [
            ("ROC Curves", "roc_curves.png"),
            ("Precision-Recall Curves", "pr_curves.png"),
            ("Confusion Matrices", "confusion_matrices.png"),
            ("Calibration Curves", "calibration_curves.png"),
            ("Cross-Validation Box Plots", "cv_boxplots.png"),
            ("Feature Importance", "feature_importance.png"),
            ("Model × Metric Heatmap", "model_metric_heatmap.png"),
            ("Model Comparison", "model_comparison_bars.png"),
            ("Learning Curves", "learning_curves.png"),
        ]
        tabs = st.tabs([f[0] for f in figs])
        for tab, (_, fname) in zip(tabs, figs):
            with tab:
                p = eval_dir / fname
                if p.exists():
                    st.image(str(p), use_column_width=True)
                else:
                    st.info(f"`{fname}` not yet generated.")


# ──────────────────────────────────────────────────────────────────────────
# PAGE 3 — Dataset Explorer
# ──────────────────────────────────────────────────────────────────────────
elif page == "Dataset Explorer":
    st.subheader("Processed Dataset Explorer")
    df = load_processed_data()
    if df is None:
        st.warning("Processed data not found. Run the pipeline.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Total Features", df.shape[1])
        if "label" in df.columns:
            c3.metric("Class Balance", f"{df['label'].mean() * 100:.1f}% Progressive")

        st.markdown("#### Filters")
        f1, f2, f3 = st.columns(3)
        with f1:
            label_filter = st.multiselect(
                "Class", [0, 1], default=[0, 1],
                format_func=lambda x: "Non-Progressive" if x == 0 else "Progressive",
            )
        with f2:
            if "gender" in df.columns:
                gender_filter = st.multiselect(
                    "Gender", df["gender"].unique().tolist(),
                    default=df["gender"].unique().tolist(),
                )
            else:
                gender_filter = None
        with f3:
            age_range = st.slider(
                "Age",
                int(df["age_years"].min()), int(df["age_years"].max()),
                (int(df["age_years"].min()), int(df["age_years"].max())),
            )

        f = df[df["label"].isin(label_filter)]
        if gender_filter is not None:
            f = f[f["gender"].isin(gender_filter)]
        f = f[(f["age_years"] >= age_range[0]) & (f["age_years"] <= age_range[1])]

        st.dataframe(f.head(200), use_container_width=True, height=300)

        if len(f) > 0:
            st.markdown("#### Distribution Explorer")
            cols = [c for c in f.columns if pd.api.types.is_numeric_dtype(f[c]) and c != "label"]
            chosen = st.selectbox("Feature", cols, index=cols.index("kmax_value_D") if "kmax_value_D" in cols else 0)
            label_str = f["label"].astype(str).map({"0": "Non-Progressive", "1": "Progressive"})
            fig = px.histogram(
                f, x=chosen, color=label_str,
                color_discrete_map={"Non-Progressive": "#1565C0", "Progressive": "#E65100"},
                marginal="box", barmode="overlay", opacity=0.65, nbins=40,
            )
            fig.update_layout(height=440, legend_title="Class")
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 4 — About
# ──────────────────────────────────────────────────────────────────────────
elif page == "About":
    st.subheader("About This Project")
    st.markdown(
        """
        ### Development and Validation of an AI-Based Model for Myopia Progression

        **Research project** — MPhil Ophthalmology
        *Investigator:* Syed Ahmad Hassan (2024-MPhil-OP-037)
        *AI Engineering:* Ali Nawaz

        ---

        #### Objective
        To develop and validate a machine learning model that classifies myopic
        eyes into **progressive** and **non-progressive** categories based on
        routine clinical corneal topography measurements.

        #### Dataset
        - **1,642 records** (combined v1 and v2 collections)
        - **15 raw clinical fields** including age, gender, eye laterality,
          astigmatism, keratometry, pachymetry, and asphericity
        - **0 missing values, 0 duplicates** after de-duplication
        - **Class distribution:** 1,077 non-progressive (65.6%), 565 progressive (34.4%)

        #### Methodology
        1. **Feature Engineering** — 32 engineered features derived from
           keratoconus screening literature (KISA, Randleman ERSS, CLMI proxy)
        2. **Preprocessing** — duplicate removal, IQR × 3 outlier capping,
           median imputation, stratified 70/10/20 train/val/test split,
           StandardScaler fit on training only
        3. **Class Imbalance** — SMOTE applied to training only
        4. **Model Comparison** — 11 classifiers + stacking ensemble,
           5-fold stratified cross-validation
        5. **Evaluation** — AUC-ROC, F1, sensitivity, specificity, calibration,
           bootstrap 95% confidence intervals, SHAP explainability

        #### Best Model
        **LightGBM** — AUC-ROC = 0.9996, 95% CI [0.9987, 1.0000]

        ---

        #### References
        - Rabinowitz, Y.S. (1998). Keratoconus. *Surv Ophthalmol* 42(4)
        - Randleman, J.B. (2008). Risk Assessment for Ectasia. *J Refract Surg*
        - Chawla, N.V. (2002). SMOTE. *JAIR* 16, 321–357
        - Lundberg, S.M. (2017). Unified Approach to Interpreting Model Predictions. *NeurIPS*
        - Ke, G. (2017). LightGBM. *NeurIPS*
        """
    )

    st.markdown(
        '<div class="disclaimer">'
        '<strong>Important:</strong> This tool is intended for research and clinical '
        'decision support only. It does not replace ophthalmologist judgement. '
        'Final diagnosis and management decisions must be made by qualified clinicians.'
        '</div>',
        unsafe_allow_html=True,
    )
