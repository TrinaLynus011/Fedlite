import pandas as pd
import numpy as np
import os
import json
import joblib
import torch
import sys

# Add src/ to path
sys.path.append(os.path.abspath("src"))

from models.logistic_regression import train_sklearn_lr
from models.mlp import train_pytorch_mlp
from models.xgboost_baseline import train_xgboost
from utils.metrics import calculate_metrics, save_evaluation_plots

def run_baselines():
    processed_dir = "data/processed"
    print("Loading preprocessed datasets...")
    train_df = pd.read_csv(os.path.join(processed_dir, "train_preprocessed.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val_preprocessed.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_preprocessed.csv"))
    
    target_col = "readmitted"
    features = [col for col in train_df.columns if col != target_col]
    
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    
    X_val = val_df[features].values
    y_val = val_df[target_col].values
    
    X_test = test_df[features].values
    y_test = test_df[target_col].values
    
    os.makedirs("artifacts/checkpoints/central", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    results = {}
    
    # ----------------------------------------------------
    # 1. Centralized Logistic Regression
    # ----------------------------------------------------
    print("\n--- Training Centralized Logistic Regression ---")
    lr_model, lr_val_prob, lr_time = train_sklearn_lr(X_train, y_train, X_val, y_val)
    lr_test_prob = lr_model.predict_proba(X_test)[:, 1]
    
    # Save checkpoint
    lr_checkpoint_path = "artifacts/checkpoints/central/logistic_regression.joblib"
    joblib.dump(lr_model, lr_checkpoint_path)
    print(f"Saved LR checkpoint to {lr_checkpoint_path}")
    
    # Compute metrics
    lr_metrics = calculate_metrics(y_test, lr_test_prob)
    lr_metrics["training_time_sec"] = lr_time
    lr_metrics["model_size_kb"] = float(os.path.getsize(lr_checkpoint_path) / 1024.0)
    results["CENTRAL_LR"] = lr_metrics
    
    # Save plots
    save_evaluation_plots(y_test, lr_test_prob, "CENTRAL_LR")
    
    # ----------------------------------------------------
    # 2. Centralized PyTorch MLP
    # ----------------------------------------------------
    print("\n--- Training Centralized PyTorch MLP ---")
    mlp_model, mlp_val_prob, mlp_stats = train_pytorch_mlp(X_train, y_train, X_val, y_val)
    
    # Get test predictions
    mlp_model.eval()
    with torch.no_grad():
        mlp_test_prob = torch.sigmoid(
            mlp_model(torch.FloatTensor(X_test))
        ).squeeze().numpy()
        if mlp_test_prob.ndim == 0:
            mlp_test_prob = np.array([mlp_test_prob.item()])
            
    # Save checkpoint
    mlp_checkpoint_path = "artifacts/checkpoints/central/mlp.pt"
    torch.save(mlp_model.state_dict(), mlp_checkpoint_path)
    print(f"Saved MLP checkpoint to {mlp_checkpoint_path}")
    
    # Compute metrics
    mlp_metrics = calculate_metrics(y_test, mlp_test_prob)
    mlp_metrics["training_time_sec"] = mlp_stats["training_time"]
    mlp_metrics["model_size_kb"] = mlp_stats["fp32_size_kb"]
    mlp_metrics["parameter_count"] = mlp_stats["parameter_count"]
    mlp_metrics["estimated_int8_size_kb"] = mlp_stats["estimated_int8_size_kb"]
    mlp_metrics["inference_latency_ms"] = mlp_stats["inference_latency_ms"]
    mlp_metrics["ram_usage_mb"] = mlp_stats["ram_usage_mb"]
    results["CENTRAL_MLP"] = mlp_metrics
    
    # Save plots
    save_evaluation_plots(y_test, mlp_test_prob, "CENTRAL_MLP")
    
    # ----------------------------------------------------
    # 3. Centralized XGBoost
    # ----------------------------------------------------
    print("\n--- Training Centralized XGBoost Benchmark ---")
    xgb_model, xgb_val_prob, xgb_time = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    # Save checkpoint
    xgb_checkpoint_path = "artifacts/checkpoints/central/xgboost.joblib"
    joblib.dump(xgb_model, xgb_checkpoint_path)
    print(f"Saved XGBoost checkpoint to {xgb_checkpoint_path}")
    
    # Compute metrics
    xgb_metrics = calculate_metrics(y_test, xgb_test_prob)
    xgb_metrics["training_time_sec"] = xgb_time
    xgb_metrics["model_size_kb"] = float(os.path.getsize(xgb_checkpoint_path) / 1024.0)
    results["CENTRAL_XGB"] = xgb_metrics
    
    # Save plots
    save_evaluation_plots(y_test, xgb_test_prob, "CENTRAL_XGB")
    
    # ----------------------------------------------------
    # Write JSON and CSV summaries
    # ----------------------------------------------------
    # JSON
    with open("results/centralized_baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/centralized_baselines.json")
    
    # CSV comparison table
    comparison_rows = []
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "AUROC": f"{metrics['auroc']:.4f}",
            "AUPRC": f"{metrics['auprc']:.4f}",
            "F1-Score": f"{metrics['f1']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "Specificity": f"{metrics['specificity']:.4f}",
            "Balanced Accuracy": f"{metrics['balanced_accuracy']:.4f}",
            "Brier Score": f"{metrics['brier_score']:.4f}",
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Training Time (s)": f"{metrics['training_time_sec']:.2f}",
            "Model Size (KB)": f"{metrics['model_size_kb']:.2f}"
        }
        comparison_rows.append(row)
        
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv("results/centralized_baselines.csv", index=False)
    print("Saved results/centralized_baselines.csv")
    
    # Print comparison
    print("\n=== Centralized Baseline Comparison ===")
    print(comp_df.to_string(index=False))

if __name__ == "__main__":
    run_baselines()
