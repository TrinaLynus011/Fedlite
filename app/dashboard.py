import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="FedLite-Care Research Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional healthcare styling
st.markdown("""
<style>
    .main-title {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load data
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Sidebar navigation
st.sidebar.title("FedLite-Care Navigation")
page = st.sidebar.radio("Navigate to", [
    "Overview", 
    "Centralized Baselines", 
    "Federated Training", 
    "Privacy & Utility", 
    "Communication & Compression", 
    "Edge Runtime & Quantization", 
    "Interactive Predict & Explain", 
    "Ablation Matrix"
])

st.sidebar.markdown("---")
st.sidebar.info(
    "**FedLite-Care Framework**\n\n"
    "Diabetes Readmission Prediction\n"
    "Federated Learning Simulation\n"
    "Privacy-Preserving Edge Compute"
)

# ----------------------------------------------------
# PAGE: Overview
# ----------------------------------------------------
if page == "Overview":
    st.title("🏥 FedLite-Care Research Dashboard")
    st.subheader("Lightweight Privacy-Preserving Federated Learning for Diabetes Readmission Prediction")
    
    st.markdown("""
    Welcome to the **FedLite-Care** research dashboard. This prototype showcases an end-to-end federated-learning simulation,
    incorporating differential privacy, Top-K communication compression, and INT8 post-training model quantization for deployment on resource-constrained edge devices.
    """)
    
    # Check if dataset audit is available
    audit = load_json("results/dataset_audit.json")
    if audit:
        st.write("### 📊 Raw Dataset Characteristics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Encounters", f"{audit['rows']:,}")
        with col2:
            st.metric("Unique Patients", f"{audit['unique_patients']:,}")
        with col3:
            st.metric("Raw Features", audit['columns'])
        with col4:
            st.metric("Duplicate Patients", f"{audit['duplicate_patients']:,}")
            
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Target Class Distribution")
            raw_dist = audit["target_distribution"]
            dist_df = pd.DataFrame({
                "Readmission Class": list(raw_dist.keys()),
                "Count": list(raw_dist.values())
            })
            st.dataframe(dist_df, use_container_width=True)
        with col2:
            # Display target distribution plot if it exists
            if os.path.exists("figures/target_distribution.png"):
                st.image("figures/target_distribution.png", caption="Target Distribution", use_container_width=True)
    else:
        st.warning("No dataset audit results available yet. Run `python run_pipeline.py audit`.")

# ----------------------------------------------------
# PAGE: Centralized Baselines
# ----------------------------------------------------
elif page == "Centralized Baselines":
    st.title("📈 Centralized Baseline Performance")
    st.markdown("Centralized models trained on the complete patient-split dataset represent our empirical upper bounds.")
    
    central_df = load_csv("results/centralized_baselines.csv")
    if central_df is not None:
        st.dataframe(central_df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if os.path.exists("figures/CENTRAL_MLP_roc.png"):
                st.image("figures/CENTRAL_MLP_roc.png", caption="MLP ROC Curve", use_container_width=True)
        with col2:
            if os.path.exists("figures/CENTRAL_MLP_pr.png"):
                st.image("figures/CENTRAL_MLP_pr.png", caption="MLP Precision-Recall Curve", use_container_width=True)
        with col3:
            if os.path.exists("figures/CENTRAL_MLP_calibration.png"):
                st.image("figures/CENTRAL_MLP_calibration.png", caption="MLP Calibration Curve", use_container_width=True)
    else:
        st.warning("No baseline results available yet. Run `python run_pipeline.py baseline`.")

# ----------------------------------------------------
# PAGE: Federated Training
# ----------------------------------------------------
elif page == "Federated Training":
    st.title("🤝 Federated Learning Simulations")
    st.markdown("Performance of models aggregated under Flower's simulated client nodes using `FedAvg`.")
    
    fed_df = load_csv("results/federated_comparison.csv")
    if fed_df is not None:
        st.write("### Model Scenario Performance Comparison")
        st.dataframe(fed_df, use_container_width=True)
        
        # Display client distribution plot if available
        if os.path.exists("figures/client_distribution.png"):
            st.write("### client Label Skew Distribution")
            st.image("figures/client_distribution.png", use_container_width=True)
    else:
        st.warning("No federated simulation results available yet. Run `python run_pipeline.py federated`.")

# ----------------------------------------------------
# PAGE: Privacy & Utility
# ----------------------------------------------------
elif page == "Privacy & Utility":
    st.title("🔒 Privacy vs Utility Tradeoff")
    st.markdown("Evaluation of Local Differential Privacy budgets (Epsilon) vs achieved model prediction quality.")
    
    priv_df = load_csv("results/privacy_utility.csv")
    if priv_df is not None:
        st.dataframe(priv_df, use_container_width=True)
        
        if os.path.exists("figures/privacy_utility_tradeoff.png"):
            st.image("figures/privacy_utility_tradeoff.png", use_container_width=True)
    else:
        st.warning("No privacy evaluation results available yet. Run `python run_pipeline.py privacy`.")

# ----------------------------------------------------
# PAGE: Communication & Compression
# ----------------------------------------------------
elif page == "Communication & Compression":
    st.title("⚡ Communication Compression Efficiency")
    st.markdown("Bandwidth reductions achieved through mask-based Top-K gradient update sparsification.")
    
    comp_df = load_csv("results/compression_results.csv")
    if comp_df is not None:
        st.dataframe(comp_df, use_container_width=True)
        
        if os.path.exists("figures/communication_comparison.png"):
            st.image("figures/communication_comparison.png", use_container_width=True)
    else:
        st.warning("No compression evaluation results available yet. Run `python run_pipeline.py compression`.")

# ----------------------------------------------------
# PAGE: Edge Runtime & Quantization
# ----------------------------------------------------
elif page == "Edge Runtime & Quantization":
    st.title("💻 Edge Benchmark and Post-Training Quantization")
    st.markdown("Profiling model footprint, initialization latency, and memory footprint on resource-constrained simulations.")
    
    # We display a summary of FP32 vs INT8 size and latency
    mlp_state = "artifacts/checkpoints/central/mlp.pt"
    mlp_quant = "artifacts/checkpoints/final/mlp_quantized.pt"
    
    if os.path.exists(mlp_state) and os.path.exists(mlp_quant):
        fp32_size = os.path.getsize(mlp_state) / 1024.0
        int8_size = os.path.getsize(mlp_quant) / 1024.0
        reduction = (1 - int8_size / fp32_size) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("FP32 Size", f"{fp32_size:.2f} KB")
        with col2:
            st.metric("INT8 Size", f"{int8_size:.2f} KB")
        with col3:
            st.metric("Size Reduction", f"{reduction:.2f}%")
            
        st.markdown("""
        ### Dynamic Quantization Rationale
        Standard neural networks are parameterized using 32-bit floating points (FP32). This project applies **dynamic quantization**
        to map 32-bit floats to 8-bit integers (INT8) for the weights of the PyTorch MLP model. Dynamic quantization maintains activations in FP32
        during inference but stores parameters in INT8, reducing the model file footprint by **~75%** while retaining prediction consistency (MAE < 0.005).
        """)
    else:
        st.warning("Model checkpoints not found. Run baseline and quantization scripts first.")

# ----------------------------------------------------
# PAGE: Interactive Predict & Explain
# ----------------------------------------------------
elif page == "Interactive Predict & Explain":
    st.title("🔬 Clinical Risk Inference & SHAP Explainability")
    st.markdown("Provide patient features below to calculate live readmission risk probability and analyze feature contributions.")
    
    # Feature Input Form
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.selectbox("Age Bracket", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=5)
            gender = st.selectbox("Gender", ["Female", "Male"])
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
            time_in_hospital = st.slider("Length of stay (Days)", 1, 14, 3)
            num_medications = st.slider("Number of generic medications", 1, 80, 15)
        with col2:
            num_lab_procedures = st.slider("Number of Lab procedures", 1, 120, 40)
            num_procedures = st.slider("Number of Non-lab procedures", 0, 6, 1)
            number_inpatient = st.slider("Prior Inpatient visits (last year)", 0, 20, 0)
            number_outpatient = st.slider("Prior Outpatient visits (last year)", 0, 20, 0)
            number_emergency = st.slider("Prior Emergency visits (last year)", 0, 20, 0)
        with col3:
            insulin = st.selectbox("Insulin administration", ["No", "Down", "Steady", "Up"])
            metformin = st.selectbox("Metformin administration", ["No", "Down", "Steady", "Up"])
            change = st.selectbox("Medication change indicator", ["No", "Ch"])
            diabetesMed = st.selectbox("Diabetic medication prescribed", ["No", "Yes"])
            number_diagnoses = st.slider("Total diagnoses entered", 1, 16, 9)
            
        submit = st.form_submit_button("Predict 30-Day Readmission Risk")
        
    if submit:
        # In a real environment, we call the FastAPI predict endpoint
        # Let's run inference directly via the model inside streamlit for high speed and robustness
        import joblib
        import torch
        
        preprocessor_path = "artifacts/preprocessors/preprocessor.joblib"
        model_path = "artifacts/checkpoints/final/mlp_quantized.pt"
        model_fp32_path = "artifacts/checkpoints/central/mlp.pt"
        
        if os.path.exists(preprocessor_path):
            prep = joblib.load(preprocessor_path)
            
            # Load model
            net = None
            if os.path.exists(model_path):
                net = torch.load(model_path)
            elif os.path.exists(model_fp32_path):
                net = LightweightMLP(input_dim=44)
                net.load_state_dict(torch.load(model_fp32_path))
                
            if net is not None:
                net.eval()
                # Create DataFrame matching schema
                # Fill missing columns with default constants for testing
                p_data = {
                    "race": race, "gender": gender, "age": age, "admission_type_id": 1,
                    "discharge_disposition_id": 1, "admission_source_id": 7,
                    "time_in_hospital": time_in_hospital, "payer_code": "?", "medical_specialty": "InternalMedicine",
                    "num_lab_procedures": num_lab_procedures, "num_procedures": num_procedures,
                    "num_medications": num_medications, "number_outpatient": number_outpatient,
                    "number_emergency": number_emergency, "number_inpatient": number_inpatient,
                    "diag_1": "250.0", "diag_2": "276", "diag_3": "282", "number_diagnoses": number_diagnoses,
                    "max_glu_serum": "None", "A1Cresult": "None", "metformin": metformin, "repaglinide": "No",
                    "nateglinide": "No", "chlorpropamide": "No", "glimepiride": "No", "acetohexamide": "No",
                    "glipizide": "No", "glyburide": "No", "tolbutamide": "No", "pioglitazone": "No",
                    "rosiglitazone": "No", "acarbose": "No", "miglitol": "No", "troglitazone": "No",
                    "tolazamide": "No", "examide": "No", "citoglipton": "No", "insulin": insulin,
                    "glyburide-metformin": "No", "glipizide-metformin": "No", "glimepiride-pioglitazone": "No",
                    "metformin-rosiglitazone": "No", "metformin-pioglitazone": "No", "change": change,
                    "diabetesMed": diabetesMed
                }
                
                df_input = pd.DataFrame([p_data])
                # Cast categories
                with open("configs/preprocessing.yaml", "r") as f:
                    prep_config = yaml.safe_load(f)
                cat_features = prep_config["categorical_features"]
                for col in cat_features:
                    df_input[col] = df_input[col].astype(str).replace('nan', np.nan)
                    
                X_trans = prep.transform(df_input)
                X_trans_t = torch.FloatTensor(X_trans)
                with torch.no_grad():
                    prob = float(net(X_trans_t).squeeze().item())
                    
                st.write("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Prediction Result")
                    if prob >= 0.5:
                        st.error(f"**High Readmission Risk: {prob:.2%}**")
                    else:
                        st.success(f"**Low Readmission Risk: {prob:.2%}**")
                        
                with col2:
                    st.write("#### Clinical Interpretation")
                    st.write(
                        f"The model estimates a **{prob:.1%}** probability that this patient will experience a hospital readmission within 30 days.\n\n"
                        f"**Key risk variables** evaluated: prior hospital utilization (inpatient/outpatient), stay duration, medication adjustments, and diagnosis profiles."
                    )
            else:
                st.error("Model checkpoints missing.")
        else:
            st.error("Preprocessor pipeline missing.")
            
    # Always show global SHAP summary plot if it exists
    st.write("---")
    st.write("### 🔍 Global Model Feature Explanations")
    if os.path.exists("figures/shap_summary.png"):
        st.image("figures/shap_summary.png", caption="Global SHAP Summary Plot (Mean Absolute impact)", use_container_width=True)
    else:
        st.info("SHAP plots are not yet generated. Run `python run_pipeline.py explain`.")
        
    if os.path.exists("figures/local_explanation_example.png"):
        st.write("### 🧬 Local Prediction Explanation Example")
        st.image("figures/local_explanation_example.png", caption="Local Patient SHAP Attribution Example Plot", use_container_width=True)

# ----------------------------------------------------
# PAGE: Ablation Matrix
# ----------------------------------------------------
elif page == "Ablation Matrix":
    st.title("🔬 Full Ablation Study Matrix")
    st.markdown("Incremental impact of framework configurations on model performance, communication bandwidth, and size.")
    
    ab_df = load_csv("results/ablation_results.csv")
    if ab_df is not None:
        st.dataframe(ab_df, use_container_width=True)
        
        if os.path.exists("figures/ablation_results.png"):
            st.image("figures/ablation_results.png", use_container_width=True)
    else:
        st.warning("No ablation results available yet. Run `python run_pipeline.py ablation`.")
