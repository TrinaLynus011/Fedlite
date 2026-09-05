import pandas as pd
import numpy as np
import os
import sys
import yaml
import shap
import matplotlib.pyplot as plt
import torch
import json

# Add src/ to path
sys.path.append(os.path.abspath("src"))
from models.mlp import LightweightMLP

def load_feature_dictionary():
    dict_path = "configs/feature_dictionary.yaml"
    if os.path.exists(dict_path):
        with open(dict_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def run_explainability(model_type="mlp", checkpoint_path=None):
    processed_dir = "data/processed"
    train_df = pd.read_csv(os.path.join(processed_dir, "train_preprocessed.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val_preprocessed.csv"))
    
    target_col = "readmitted"
    features = [col for col in train_df.columns if col != target_col]
    
    X_train = train_df[features].values
    X_val = val_df[features].values
    
    # Load feature dictionary for clinical descriptions
    feat_dict = load_feature_dictionary()
    readable_features = [feat_dict.get(f, f) for f in features]
    
    # 1. Load Model
    if model_type == "mlp":
        if checkpoint_path is None:
            checkpoint_path = "artifacts/checkpoints/central/mlp.pt"
        input_dim = len(features)
        model = LightweightMLP(input_dim=input_dim)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        # PyTorch model wrapper for SHAP
        def predict_fn(x):
            x_t = torch.FloatTensor(x)
            with torch.no_grad():
                return torch.sigmoid(model(x_t)).numpy()
    elif model_type == "xgboost":
        import joblib
        if checkpoint_path is None:
            checkpoint_path = "artifacts/checkpoints/central/xgboost.joblib"
        model = joblib.load(checkpoint_path)
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
    else:
        raise ValueError(f"Unsupported model type for explainability: {model_type}")
        
    print(f"Running SHAP explainability on {model_type}...")
    
    # 2. Set up background and explain sample
    # KernelExplainer requires background data. We select 50 samples for speed.
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), 50, replace=False)
    background = X_train[bg_idx]
    
    # We explain 100 validation samples for global analysis
    explain_idx = np.random.choice(len(X_val), 100, replace=False)
    explain_data = X_val[explain_idx]
    
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain_data)
    
    # Handle SHAP output shapes
    # (sometimes shap_values returns a list for binary classification, or a single array)
    if isinstance(shap_values, list):
        # class 1 probabilities
        shap_vals_arr = shap_values[0]
    else:
        shap_vals_arr = shap_values
        
    # Remove dimension if it is shape (N, D, 1) or similar
    if len(shap_vals_arr.shape) == 3:
        shap_vals_arr = shap_vals_arr.squeeze(-1)
    elif len(shap_vals_arr.shape) == 2 and shap_vals_arr.shape[1] == 1:
        # If it returned shape (100, 1) but we have 44 features, something is wrong.
        # Let's re-run explain on a single feature if needed or shape it
        pass
        
    print("Computed SHAP values.")
    
    # 3. Global Summary Plot
    os.makedirs("figures", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    # Map raw features to human-readable names for global summary plot
    shap.summary_plot(
        shap_vals_arr, 
        explain_data, 
        feature_names=readable_features, 
        show=False
    )
    plt.title("FedLite-Care Global Feature Importance (SHAP Summary)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("figures/shap_summary.png", dpi=300)
    plt.close()
    print("Saved figures/shap_summary.png")
    
    # Calculate mean absolute SHAP values for quantitative report
    mean_abs_shap = np.mean(np.abs(shap_vals_arr), axis=0)
    shap_importance = pd.DataFrame({
        "Feature": features,
        "Readable_Feature": readable_features,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)
    
    os.makedirs("results", exist_ok=True)
    shap_importance.to_csv("results/shap_feature_importance.csv", index=False)
    print("Saved results/shap_feature_importance.csv")
    
    # 4. Local Explanation Example
    # Select a patient who has high predicted risk
    val_df_subset = val_df.iloc[explain_idx].copy()
    probs = predict_fn(explain_data).flatten()
    val_df_subset["predicted_risk"] = probs
    
    # Find a patient readmitted with high predicted probability
    high_risk_idx = val_df_subset[val_df_subset[target_col] == 1]["predicted_risk"].idxmax()
    # Map index back to explain_data relative index
    rel_idx = np.where(explain_idx == high_risk_idx)[0][0]
    
    sample_shap = shap_vals_arr[rel_idx]
    sample_features = explain_data[rel_idx]
    
    # Let's plot local explanation bar chart
    local_df = pd.DataFrame({
        "Feature": readable_features,
        "SHAP_Value": sample_shap,
        "Feature_Value": sample_features
    })
    
    # Sort by absolute SHAP value and take top 8 contributors
    local_df["Abs_SHAP"] = local_df["SHAP_Value"].abs()
    top_contributors = local_df.sort_values(by="Abs_SHAP", ascending=False).head(8).copy()
    
    # Sort so positive is at the top or in logical order
    top_contributors = top_contributors.sort_values(by="SHAP_Value", ascending=True)
    
    plt.figure(figsize=(8, 5))
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_contributors["SHAP_Value"]]
    plt.barh(top_contributors["Feature"], top_contributors["SHAP_Value"], color=colors)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel("SHAP Value (Impact on Prediction Risk)")
    plt.title(f"Local Patient Risk Explanation (Predicted Risk: {probs[rel_idx]:.2%})")
    plt.tight_layout()
    plt.savefig("figures/local_explanation_example.png", dpi=300)
    plt.close()
    print("Saved figures/local_explanation_example.png")
    
    # Save a machine-readable local explanation summary
    local_explanation = {
        "encounter_id": "Anonymized Encounter",
        "predicted_risk_probability": float(probs[rel_idx]),
        "risk_classification": int(probs[rel_idx] >= 0.5),
        "top_positive_contributors": top_contributors[top_contributors["SHAP_Value"] > 0][["Feature", "SHAP_Value"]].to_dict(orient="records"),
        "top_negative_contributors": top_contributors[top_contributors["SHAP_Value"] < 0][["Feature", "SHAP_Value"]].to_dict(orient="records")
    }
    
    with open("results/local_explanation.json", "w") as f:
        json.dump(local_explanation, f, indent=2)
    print("Saved results/local_explanation.json")

if __name__ == "__main__":
    run_explainability()
