import os
import sys
import pandas as pd
import numpy as np
import torch
import json

# Add src/ to path
sys.path.append(os.path.abspath("src"))
from models.mlp import LightweightMLP
from utils.metrics import calculate_metrics

def run_subgroup_analysis(model_path="artifacts/checkpoints/federated/mlp_FL_MLP_NONIID_5.pt"):
    processed_dir = "data/processed"
    test_preprocessed_path = os.path.join(processed_dir, "test_preprocessed.csv")
    test_raw_path = os.path.join(processed_dir, "test.csv")
    
    if not os.path.exists(test_preprocessed_path) or not os.path.exists(test_raw_path):
        print("Missing processed datasets. Run preprocessing first.")
        return
        
    test_df_prep = pd.read_csv(test_preprocessed_path)
    test_df_raw = pd.read_csv(test_raw_path)
    
    # Verify dimensions match
    assert len(test_df_prep) == len(test_df_raw), "Preprocessed and raw test sets differ in length!"
    
    target_col = "readmitted"
    features = [col for col in test_df_prep.columns if col != target_col]
    
    # 1. Load Model
    input_dim = len(features)
    # MLP model is saved as state dict in federated run, let's load it
    model = LightweightMLP(input_dim=input_dim)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=False))
    elif os.path.exists("artifacts/checkpoints/central/mlp.pt"):
        print("Federated model checkpoint not found. Using Centralized MLP instead.")
        model.load_state_dict(torch.load("artifacts/checkpoints/central/mlp.pt", weights_only=False))
    else:
        print("No MLP checkpoint found. Skipping subgroup analysis.")
        return
        
    model.eval()
    
    # Predictions
    X_test = test_df_prep[features].values
    y_test = test_df_prep[target_col].values
    
    with torch.no_grad():
        test_probs = torch.sigmoid(model(torch.FloatTensor(X_test))).squeeze().numpy()
        
    # Put predictions back into the raw dataframe to slice by original categories
    test_df_raw["y_prob"] = test_probs
    test_df_raw["y_true"] = y_test
    
    subgroups = ["race", "gender", "age"]
    rows = []
    
    for category in subgroups:
        if category not in test_df_raw.columns:
            continue
            
        # Group by category and compute metrics
        for group_name, group_data in test_df_raw.groupby(category):
            n_samples = len(group_data)
            # Skip subgroups with extremely small sample sizes for statistical stability
            if n_samples < 50:
                continue
                
            y_t = group_data["y_true"].values
            y_p = group_data["y_prob"].values
            
            # Check if we have both classes present in the subgroup
            if len(np.unique(y_t)) < 2:
                # Can't calculate AUROC if only one class is present
                auroc = 0.5
            else:
                from sklearn.metrics import roc_auc_score
                auroc = float(roc_auc_score(y_t, y_p))
                
            metrics = calculate_metrics(y_t, y_p)
            
            rows.append({
                "Demographic_Category": category,
                "Subgroup_Name": str(group_name),
                "Sample_Count": n_samples,
                "Positive_Prevalence": float(np.mean(y_t)),
                "AUROC": auroc,
                "AUPRC": metrics["auprc"],
                "F1": metrics["f1"],
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
                "Brier_Score": metrics["brier_score"]
            })
            
    subgroup_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    subgroup_df.to_csv("results/subgroup_metrics.csv", index=False)
    print("Saved results/subgroup_metrics.csv")
    print(subgroup_df.to_string(index=False))

if __name__ == "__main__":
    run_subgroup_analysis()
