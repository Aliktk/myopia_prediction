"""
=============================================================================
Streamlit Application — AI-Based Myopia Progression Prediction
=============================================================================
Run: streamlit run streamlit_app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Myopia Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1976D2, #00BCD4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.5rem;
    }
    .sub-header { text-align: center; color: #666; font-size: 1rem; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem; border-radius: 12px; text-align: center; margin: 0.3rem;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #1976D2; }
    .metric-label { font-size: 0.8rem; color: #555; text-transform: uppercase; }
    .risk-high { background: linear-gradient(135deg, #FFEBEE, #FFCDD2); border-left: 5px solid #D32F2F; padding: 1rem; border-radius: 8px; }
    .risk-low { background: linear-gradient(135deg, #E8F5E9, #C8E6C9); border-left: 5px solid #388E3C; padding: 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.json", "r") as f:
        features = json.load(f)
    return model, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv("processed_data_with_features.csv")

@st.cache_data
def load_results():
    return pd.read_csv("model_results_summary.csv")

try:
    model, scaler, feature_columns = load_model()
    df = load_data()
    results_df = load_results()
    MODEL_LOADED = True
except:
    MODEL_LOADED = False

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/eye.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Dashboard",
    "🔬 Predict Myopia",
    "📊 Data Explorer",
    "🏆 Model Performance",
    "📋 About Research"
])

# ── Helper Functions ─────────────────────────────────────────────────────────
def engineer_features(row):
    """Apply same feature engineering as training pipeline."""
    features = {}
    features['age_years'] = row['age_years']
    features['gender_encoded'] = 0 if row['gender'] == 'Female' else 1
    features['eye_encoded'] = 0 if row['eye'] == 'OD (Right)' else 1
    features['astig_value_D'] = row['astig_value_D']
    features['astig_axis_deg'] = row['astig_axis_deg']
    features['kmax_value_D'] = row['kmax_value_D']
    features['kmax_axis_deg'] = row['kmax_axis_deg']
    features['pachy_central_um'] = row['pachy_central_um']
    features['pachy_thinnest_um'] = row['pachy_thinnest_um']
    features['pachy_thinnest_x'] = row['pachy_thinnest_x']
    features['pachy_thinnest_y'] = row['pachy_thinnest_y']
    features['asphericity_anterior'] = row['asphericity_anterior']
    features['asphericity_posterior'] = row['asphericity_posterior']
    
    # Engineered
    features['pachy_diff'] = row['pachy_central_um'] - row['pachy_thinnest_um']
    features['pachy_ratio'] = row['pachy_thinnest_um'] / row['pachy_central_um']
    features['pachy_thinnest_displacement'] = np.sqrt(row['pachy_thinnest_x']**2 + row['pachy_thinnest_y']**2)
    features['asphericity_diff'] = row['asphericity_anterior'] - row['asphericity_posterior']
    features['asphericity_abs_sum'] = abs(row['asphericity_anterior']) + abs(row['asphericity_posterior'])
    features['astig_abs'] = abs(row['astig_value_D'])
    features['astig_axis_sin'] = np.sin(2 * np.radians(row['astig_axis_deg']))
    features['astig_axis_cos'] = np.cos(2 * np.radians(row['astig_axis_deg']))
    features['kmax_axis_sin'] = np.sin(2 * np.radians(row['kmax_axis_deg']))
    features['kmax_axis_cos'] = np.cos(2 * np.radians(row['kmax_axis_deg']))
    features['corneal_power_index'] = row['kmax_value_D'] * (1 + row['asphericity_anterior'])
    features['kmax_astig_interaction'] = row['kmax_value_D'] * abs(row['astig_value_D'])
    features['age_kmax_interaction'] = row['age_years'] * row['kmax_value_D']
    features['pachy_asph_interaction'] = row['pachy_central_um'] * abs(row['asphericity_anterior'])
    
    risk = 0
    if row['kmax_value_D'] > 46: risk += 1
    if abs(row['astig_value_D']) > 3: risk += 1
    if row['pachy_central_um'] < 520: risk += 1
    if row['asphericity_anterior'] > 0: risk += 1
    features['corneal_risk_score'] = risk
    
    return features

# ════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">AI-Based Myopia Progression Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Development and Validation of AI Model for Myopia Prediction<br>Syed Ahmad Hassan — MPhil Optometry — University of Faisalabad</div>', unsafe_allow_html=True)
    
    if MODEL_LOADED:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", f"{len(df):,}")
        with col2:
            st.metric("Features Engineered", "28")
        with col3:
            st.metric("Models Trained", "11")
        with col4:
            best = results_df.iloc[0]
            st.metric("Best AUC-ROC", f"{best['AUC-ROC']:.4f}")
        
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df, names=df['label'].map({0: 'Non-Progressive', 1: 'Progressive'}),
                            title='Dataset Label Distribution',
                            color_discrete_sequence=['#2196F3', '#FF5722'])
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with c2:
            fig_bar = px.bar(results_df, x='Model', y=['Accuracy', 'F1-Score', 'AUC-ROC'],
                            title='Model Performance Overview', barmode='group',
                            color_discrete_sequence=['#1976D2', '#F57C00', '#388E3C'])
            fig_bar.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_bar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT MYOPIA
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Predict Myopia":
    st.markdown('<div class="main-header">Myopia Progression Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter patient corneal topography data to predict progression risk.")
    
    if not MODEL_LOADED:
        st.error("Model files not found. Place best_model.pkl, scaler.pkl, feature_columns.json in app directory.")
    else:
        with st.form("prediction_form"):
            st.subheader("Patient Demographics")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age (years)", 13, 65, 25)
            with col2:
                gender = st.selectbox("Gender", ["Female", "Male"])
            with col3:
                eye = st.selectbox("Eye", ["OD (Right)", "OS (Left)"])
            
            st.subheader("Corneal Topography Parameters")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                astig_val = st.number_input("Astigmatism (D)", -21.0, 13.0, -2.0, 0.1)
                kmax_val = st.number_input("Kmax (D)", 35.0, 60.0, 44.0, 0.1)
            with col2:
                astig_axis = st.number_input("Astig Axis (°)", 0, 180, 90)
                kmax_axis = st.number_input("Kmax Axis (°)", 0, 180, 90)
            with col3:
                pachy_central = st.number_input("Central Pachymetry (µm)", 400, 650, 530)
                pachy_thinnest = st.number_input("Thinnest Pachymetry (µm)", 380, 640, 520)
            with col4:
                pachy_x = st.number_input("Thinnest Point X", -3.0, 3.0, 0.0, 0.1)
                pachy_y = st.number_input("Thinnest Point Y", -3.0, 3.0, 0.0, 0.1)
            
            col1, col2 = st.columns(2)
            with col1:
                asph_ant = st.number_input("Asphericity (Anterior)", -5.0, 5.0, -0.25, 0.01)
            with col2:
                asph_post = st.number_input("Asphericity (Posterior)", -5.0, 5.0, -0.50, 0.01)
            
            submitted = st.form_submit_button("🔍 Predict Progression Risk", use_container_width=True)
        
        if submitted:
            row = {
                'age_years': age, 'gender': gender, 'eye': eye,
                'astig_value_D': astig_val, 'astig_axis_deg': astig_axis,
                'kmax_value_D': kmax_val, 'kmax_axis_deg': kmax_axis,
                'pachy_central_um': pachy_central, 'pachy_thinnest_um': pachy_thinnest,
                'pachy_thinnest_x': pachy_x, 'pachy_thinnest_y': pachy_y,
                'asphericity_anterior': asph_ant, 'asphericity_posterior': asph_post,
            }
            features = engineer_features(row)
            X_input = pd.DataFrame([features])[feature_columns]
            X_scaled = scaler.transform(X_input)
            
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            
            st.divider()
            
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h2>⚠️ HIGH RISK — Myopia Progression Predicted</h2>
                    <p>Confidence: <strong>{proba[1]*100:.1f}%</strong></p>
                    <p>Recommendation: Close monitoring and active myopia management intervention advised.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2>✅ LOW RISK — Stable Refraction Predicted</h2>
                    <p>Confidence: <strong>{proba[0]*100:.1f}%</strong></p>
                    <p>Recommendation: Standard follow-up schedule. Continue routine monitoring.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba[1] * 100,
                title={'text': "Progression Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF5722" if prediction == 1 else "#4CAF50"},
                    'steps': [
                        {'range': [0, 30], 'color': '#E8F5E9'},
                        {'range': [30, 60], 'color': '#FFF9C4'},
                        {'range': [60, 100], 'color': '#FFEBEE'},
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<div class="main-header">Data Explorer</div>', unsafe_allow_html=True)
    
    if MODEL_LOADED:
        tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "📋 Raw Data"])
        
        with tab1:
            feature = st.selectbox("Select Feature", 
                ['kmax_value_D', 'astig_abs', 'pachy_central_um', 'pachy_thinnest_um',
                 'asphericity_anterior', 'asphericity_posterior', 'pachy_diff',
                 'corneal_power_index', 'age_years', 'corneal_risk_score'])
            fig = px.histogram(df, x=feature, color=df['label'].map({0: 'Non-Progressive', 1: 'Progressive'}),
                              barmode='overlay', opacity=0.7, nbins=40,
                              color_discrete_map={'Non-Progressive': '#2196F3', 'Progressive': '#FF5722'},
                              title=f'Distribution of {feature} by Label')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c != 'label']
            corr = df[num_cols + ['label']].corr()['label'].drop('label').sort_values()
            fig_corr = px.bar(x=corr.values, y=corr.index, orientation='h',
                             title='Feature Correlation with Myopia Progression (Label)',
                             color=corr.values, color_continuous_scale='RdBu_r')
            fig_corr.update_layout(height=700)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            st.dataframe(df.head(100), use_container_width=True, height=500)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Performance":
    st.markdown('<div class="main-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    if MODEL_LOADED:
        st.dataframe(results_df.style.highlight_max(axis=0, color='#C8E6C9',
                      subset=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']),
                     use_container_width=True)
        
        # Radar
        fig_radar = go.Figure()
        categories = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
        for _, row in results_df.head(5).iterrows():
            values = [row[c] for c in categories]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]], theta=categories + [categories[0]],
                fill='toself', name=row['Model'], opacity=0.5
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.9, 1])),
                                title='Top 5 Models — Radar Chart')
        st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 About Research":
    st.markdown('<div class="main-header">About This Research</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Development and Validation of AI Based Model for Myopia Prediction
    
    **Researcher:** Syed Ahmad Hassan (2024-MPhil-OP-037)  
    **Supervisor:** Dr. Memoona Arshad  
    **Institution:** The University of Faisalabad — Department of Allied Health Sciences  
    
    ---
    
    **Objective:** To develop AI models analyzing corneal topography clinical data for early 
    myopia progression detection, and compare performance against conventional clinical assessment.
    
    **Methodology:**
    - Cross-sectional analytical study design
    - 1,454 eye records from Galilei G6 corneal topographer
    - 28 engineered features from 12 clinical parameters
    - 11 ML models including ensemble stacking
    - SMOTE for class balancing, 5-fold stratified CV
    
    **Key Findings:**
    - Multiple models achieved >97% accuracy on test data
    - SVM achieved highest test accuracy (98.97%) with perfect specificity
    - Logistic Regression achieved highest AUC-ROC (0.9939)
    - Feature engineering improved model discriminability significantly
    """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("© 2026 — AI Myopia Prediction Research | University of Faisalabad")
