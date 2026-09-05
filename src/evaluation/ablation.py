import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src/ to path
sys.path.append(os.path.abspath("src"))
from federated.server import run_federated_simulation
from models.mlp import train_pytorch_mlp
from quantization.quantizer import quantize_mlp_model
from utils.metrics import calculate_metrics

def run_ablation_study():
    with open("configs/data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    with open("configs/federated.yaml", "r") as f:
        fed_config = yaml.safe_load(f)
        
    rounds = fed_config.get("rounds", 10)
    local_epochs = fed_config.get("local_epochs", 3)
    batch_size = fed_config.get("local_batch_size", 128)
    lr = fed_config.get("local_learning_rate", 0.005)
    seed = data_config["split"]["seed"]
    
    processed_dir = data_config["processed_dir"]
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
    
    # We will compile the results of the 8 configurations: A0 to A7
    ablation_results = []
    
    print("\n==================================================")
    print(" STARTING FEDLITE-CARE ABLATION STUDY MATRIX")
    print("==================================================")
    
    # ----------------------------------------------------
    # A0: Centralized MLP Baseline
    # ----------------------------------------------------
    print("\n--- Ablation A0: Centralized MLP Baseline ---")
    if os.path.exists("results/centralized_baselines.json"):
        with open("results/centralized_baselines.json", "r") as f:
            base_metrics = json.load(f)["CENTRAL_MLP"]
        a0_metrics = base_metrics
    else:
        mlp_model, mlp_val_prob, mlp_stats = train_pytorch_mlp(X_train, y_train, X_val, y_val)
        import torch
        mlp_model.eval()
        with torch.no_grad():
            test_probs = torch.sigmoid(
                mlp_model(torch.FloatTensor(X_test))
            ).squeeze().numpy()
        a0_metrics = calculate_metrics(y_test, test_probs)
        a0_metrics["training_time_sec"] = mlp_stats["training_time"]
        a0_metrics["model_size_kb"] = mlp_stats["fp32_size_kb"]
        
    ablation_results.append({
        "ID": "A0",
        "Name": "Centralized Baseline",
        "AUROC": a0_metrics["auroc"],
        "AUPRC": a0_metrics["auprc"],
        "F1": a0_metrics["f1"],
        "Recall": a0_metrics["recall"],
        "Precision": a0_metrics["precision"],
        "Model Size (KB)": a0_metrics.get("model_size_kb", 19.50),
        "Communication (MB)": 0.0,
        "DP_Epsilon": "N/A",
        "Compression": "None",
        "Quantization": "None"
    })
    
    # ----------------------------------------------------
    # A1: Federated MLP (IID)
    # ----------------------------------------------------
    print("\n--- Ablation A1: Federated MLP (IID) ---")
    res_a1, cp_a1 = run_federated_simulation(
        experiment_id="ABLATION_A1",
        model_type="mlp",
        client_count=5,
        partition_method="iid",
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed
    )
    ablation_results.append({
        "ID": "A1",
        "Name": "Federated Learning (IID)",
        "AUROC": res_a1["auroc"],
        "AUPRC": res_a1["auprc"],
        "F1": res_a1["f1"],
        "Recall": res_a1["recall"],
        "Precision": res_a1["precision"],
        "Model Size (KB)": 19.50,
        "Communication (MB)": res_a1["communication"]["total_bytes"] / (1024*1024),
        "DP_Epsilon": "N/A",
        "Compression": "None",
        "Quantization": "None"
    })
    
    # ----------------------------------------------------
    # A2: Federated MLP + Non-IID
    # ----------------------------------------------------
    print("\n--- Ablation A2: Federated MLP + Non-IID (alpha=0.5) ---")
    res_a2, cp_a2 = run_federated_simulation(
        experiment_id="ABLATION_A2",
        model_type="mlp",
        client_count=5,
        partition_method="non_iid",
        alpha=0.5,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed
    )
    ablation_results.append({
        "ID": "A2",
        "Name": "Federated + Non-IID",
        "AUROC": res_a2["auroc"],
        "AUPRC": res_a2["auprc"],
        "F1": res_a2["f1"],
        "Recall": res_a2["recall"],
        "Precision": res_a2["precision"],
        "Model Size (KB)": 19.50,
        "Communication (MB)": res_a2["communication"]["total_bytes"] / (1024*1024),
        "DP_Epsilon": "N/A",
        "Compression": "None",
        "Quantization": "None"
    })
    
    # ----------------------------------------------------
    # A3: Federated MLP + DP (eps=3)
    # ----------------------------------------------------
    print("\n--- Ablation A3: Federated MLP + DP (eps=3) ---")
    # For eps=3, noise multiplier is 1.5
    res_a3, cp_a3 = run_federated_simulation(
        experiment_id="ABLATION_A3",
        model_type="mlp",
        client_count=5,
        partition_method="non_iid",
        alpha=0.5,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        dp_config={"enabled": True, "clipping_norm": 1.0, "noise_multiplier": 1.5, "delta": 1e-5},
        seed=seed
    )
    ablation_results.append({
        "ID": "A3",
        "Name": "Federated + Non-IID + DP",
        "AUROC": res_a3["auroc"],
        "AUPRC": res_a3["auprc"],
        "F1": res_a3["f1"],
        "Recall": res_a3["recall"],
        "Precision": res_a3["precision"],
        "Model Size (KB)": 19.50,
        "Communication (MB)": res_a3["communication"]["total_bytes"] / (1024*1024),
        "DP_Epsilon": "3.0",
        "Compression": "None",
        "Quantization": "None"
    })
    
    # ----------------------------------------------------
    # A4: Federated MLP + Top-K (10%)
    # ----------------------------------------------------
    print("\n--- Ablation A4: Federated MLP + Top-K (10%) ---")
    res_a4, cp_a4 = run_federated_simulation(
        experiment_id="ABLATION_A4",
        model_type="mlp",
        client_count=5,
        partition_method="non_iid",
        alpha=0.5,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        compression_config={"enabled": True, "ratio": 0.1},
        seed=seed
    )
    ablation_results.append({
        "ID": "A4",
        "Name": "Federated + Non-IID + Top-K 10%",
        "AUROC": res_a4["auroc"],
        "AUPRC": res_a4["auprc"],
        "F1": res_a4["f1"],
        "Recall": res_a4["recall"],
        "Precision": res_a4["precision"],
        "Model Size (KB)": 19.50,
        "Communication (MB)": res_a4["communication"]["total_bytes"] / (1024*1024),
        "DP_Epsilon": "N/A",
        "Compression": "Top-K 10%",
        "Quantization": "None"
    })
    
    # ----------------------------------------------------
    # A5: Federated MLP + INT8 Quantization
    # ----------------------------------------------------
    print("\n--- Ablation A5: Federated MLP + INT8 Quantization ---")
    # Apply post-training dynamic quantization to A2 model
    q_stats_a5 = quantize_mlp_model(cp_a2, input_dim=len(features), X_val=X_test)
    # The quantized model predictions Consistency MAE is computed.
    # To get actual metrics on test set, we load the quantized model object:
    import torch
    quant_model = torch.load("artifacts/checkpoints/final/mlp_quantized.pt", weights_only=False)
    quant_model.eval()
    with torch.no_grad():
        q_probs = torch.sigmoid(
            quant_model(torch.FloatTensor(X_test))
        ).squeeze().numpy()
    res_a5 = calculate_metrics(y_test, q_probs)
    
    ablation_results.append({
        "ID": "A5",
        "Name": "Federated + Non-IID + INT8",
        "AUROC": res_a5["auroc"],
        "AUPRC": res_a5["auprc"],
        "F1": res_a5["f1"],
        "Recall": res_a5["recall"],
        "Precision": res_a5["precision"],
        "Model Size (KB)": q_stats_a5["int8_size_bytes"] / 1024.0,
        "Communication (MB)": res_a2["communication"]["total_bytes"] / (1024*1024),
        "DP_Epsilon": "N/A",
        "Compression": "None",
        "Quantization": "INT8"
    })
    
    # ----------------------------------------------------
    # A6: Federated MLP + DP + Top-K
    # ----------------------------------------------------
    print("\n--- Ablation A6: Federated MLP + DP + Top-K ---")
    res_a6, cp_a6 = run_federated_simulation(
        experiment_id="ABLATION_A6",
        model_type="mlp",
        client_count=5,
        partition_method="non_iid",
        alpha=0.5,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        dp_config={"enabled": True, "clipping_norm": 1.0, "noise_multiplier": 1.5, "delta": 1e-5},
        compression_config={"enabled": True, "ratio": 0.1},
        seed=seed
    )
    ablation_results.append({
        "ID": "A6",
        "Name": "Federated + DP + Top-K",
        "AUROC": res_a6["auroc"],
        "AUPRC": res_a6["auprc"],
        "F1": res_a6["f1"],
        "Recall": res_a6["recall"],
        "Precision": res_a6["precision"],
        "Model Size (KB)": 19.50,
        "Communication (MB)": res_a6["communication"]["total_bytes"] / (1024*1024),
        "DP_Epsilon": "3.0",
        "Compression": "Top-K 10%",
        "Quantization": "None"
    })
    
    # ----------------------------------------------------
    # A7: Federated MLP + DP + Top-K + INT8 (Full Framework)
    # ----------------------------------------------------
    print("\n--- Ablation A7: Federated MLP + DP + Top-K + INT8 (Full Framework) ---")
    # Apply post-training dynamic quantization to A6 model
    q_stats_a7 = quantize_mlp_model(cp_a6, input_dim=len(features), X_val=X_test)
    quant_model_a7 = torch.load("artifacts/checkpoints/final/mlp_quantized.pt", weights_only=False)
    quant_model_a7.eval()
    with torch.no_grad():
        q_probs_a7 = torch.sigmoid(
            quant_model_a7(torch.FloatTensor(X_test))
        ).squeeze().numpy()
    res_a7 = calculate_metrics(y_test, q_probs_a7)
    
    ablation_results.append({
        "ID": "A7",
        "Name": "FedLite-Care (Full Framework)",
        "AUROC": res_a7["auroc"],
        "AUPRC": res_a7["auprc"],
        "F1": res_a7["f1"],
        "Recall": res_a7["recall"],
        "Precision": res_a7["precision"],
        "Model Size (KB)": q_stats_a7["int8_size_bytes"] / 1024.0,
        "Communication (MB)": res_a6["communication"]["total_bytes"] / (1024*1024),
        "DP_Epsilon": "3.0",
        "Compression": "Top-K 10%",
        "Quantization": "INT8"
    })
    
    # Save CSV
    ab_df = pd.DataFrame(ablation_results)
    os.makedirs("results", exist_ok=True)
    ab_df.to_csv("results/ablation_results.csv", index=False)
    print("\nSaved results/ablation_results.csv")
    
    # Print results
    print("\n=== Ablation Matrix ===")
    print(ab_df.to_string(index=False))
    
    # Plot ablation comparative bar chart
    plt.figure(figsize=(10, 6))
    x_labels = [f"{row['ID']}: {row['Name']}" for _, row in ab_df.iterrows()]
    
    x = np.arange(len(ab_df))
    width = 0.35
    
    plt.bar(x - width/2, ab_df["AUROC"], width, label='AUROC', color='#e74c3c')
    plt.bar(x + width/2, ab_df["AUPRC"], width, label='AUPRC', color='#3498db')
    
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.ylabel("Score")
    plt.title("FedLite-Care: Ablation Study Comparison (Utility Metrics)")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower left")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/ablation_results.png", dpi=300)
    plt.close()
    print("Saved figures/ablation_results.png")

if __name__ == "__main__":
    run_ablation_study()
