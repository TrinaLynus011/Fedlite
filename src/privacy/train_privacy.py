import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src/ to path
sys.path.append(os.path.abspath("src"))
from federated.server import run_federated_simulation
from privacy.dp_mechanism import compute_epsilon

def run_privacy_experiments():
    with open("configs/data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    with open("configs/federated.yaml", "r") as f:
        fed_config = yaml.safe_load(f)
    with open("configs/privacy.yaml", "r") as f:
        priv_config = yaml.safe_load(f)
        
    rounds = fed_config.get("rounds", 10)
    local_epochs = fed_config.get("local_epochs", 3)
    batch_size = fed_config.get("local_batch_size", 128)
    lr = fed_config.get("local_learning_rate", 0.005)
    seed = data_config["split"]["seed"]
    
    # 5 clients, non_iid
    client_count = 5
    partition_method = "non_iid"
    alpha = 0.5
    
    # Load dataset to determine size for DP accountant
    processed_dir = data_config["processed_dir"]
    train_df = pd.read_csv(os.path.join(processed_dir, "train_preprocessed.csv"))
    # Estimate average client dataset size
    avg_client_size = len(train_df) / client_count
    
    # Number of steps per client = rounds * local_epochs * (client_samples / batch_size)
    steps_per_client = rounds * local_epochs * int(np.ceil(avg_client_size / batch_size))
    
    # Budgets to evaluate
    budgets = priv_config["budgets"]
    
    results = []
    
    for b in budgets:
        target_eps = b["epsilon"]
        noise_multiplier = float(b["noise_multiplier"])
        
        exp_id = f"FL_MLP_NONIID_5_DP_eps_{target_eps}".replace(" ", "_")
        print(f"\n==================================================")
        print(f" RUNNING PRIVACY EXPERIMENT: {exp_id} (Noise={noise_multiplier})")
        print(f"==================================================")
        
        dp_enabled = target_eps != "No DP"
        dp_config = {
            "enabled": dp_enabled,
            "clipping_norm": float(priv_config["clipping_norm"]),
            "noise_multiplier": noise_multiplier,
            "delta": float(priv_config["delta"])
        }
        
        # Calculate actual epsilon
        if dp_enabled:
            actual_eps = compute_epsilon(
                steps=steps_per_client,
                batch_size=batch_size,
                dataset_size=avg_client_size,
                noise_multiplier=noise_multiplier,
                delta=dp_config["delta"]
            )
        else:
            actual_eps = float('inf')
            
        print(f"Privacy accountant calculated actual epsilon = {actual_eps:.4f} (delta={dp_config['delta']})")
        
        try:
            metrics, checkpoint = run_federated_simulation(
                experiment_id=exp_id,
                model_type="mlp",
                client_count=client_count,
                partition_method=partition_method,
                alpha=alpha,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                dp_config=dp_config,
                compression_config={"enabled": False},
                seed=seed
            )
            
            # Save metadata and update registry
            # We construct a metadata dict
            metadata = {
                "experiment_id": exp_id,
                "timestamp": datetime.now().isoformat(),
                "git_commit": "None",
                "random_seed": seed,
                "Python_version": sys.version,
                "model_type": "mlp",
                "client_count": client_count,
                "partition_method": partition_method,
                "dirichlet_alpha": alpha,
                "dp_enabled": dp_enabled,
                "clipping_norm": dp_config["clipping_norm"],
                "noise_multiplier": noise_multiplier,
                "actual_epsilon": actual_eps,
                "delta": dp_config["delta"],
                "metrics": metrics
            }
            
            os.makedirs("results/experiment_metadata", exist_ok=True)
            meta_path = f"results/experiment_metadata/{exp_id}.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Update registry
            registry_path = "results/experiment_registry.csv"
            row = {
                "experiment_id": exp_id,
                "model": "mlp",
                "training_type": "federated",
                "client_count": client_count,
                "partition": partition_method,
                "alpha": alpha,
                "dp_enabled": dp_enabled,
                "epsilon": f"{actual_eps:.4f}" if dp_enabled else "N/A",
                "delta": f"{dp_config['delta']}" if dp_enabled else "N/A",
                "compression": "None",
                "top_k": 1.0,
                "quantization": "None",
                "seed": seed,
                "status": "FINAL_LOCKED",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "git_commit": "None"
            }
            new_df = pd.DataFrame([row])
            if os.path.exists(registry_path):
                reg_df = pd.read_csv(registry_path)
                reg_df = reg_df[reg_df["experiment_id"] != exp_id]
                reg_df = pd.concat([reg_df, new_df], ignore_index=True)
            else:
                reg_df = new_df
            reg_df.to_csv(registry_path, index=False)
            
            results.append({
                "Epsilon_Target": str(target_eps),
                "Epsilon_Actual": actual_eps,
                "Noise_Multiplier": noise_multiplier,
                "AUROC": metrics["auroc"],
                "AUPRC": metrics["auprc"],
                "F1": metrics["f1"],
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
                "Accuracy": metrics["accuracy"]
            })
            
        except Exception as e:
            print(f"Error running privacy experiment: {str(e)}")
            
    # Save CSV
    if len(results) > 0:
        priv_df = pd.DataFrame(results)
        priv_df.to_csv("results/privacy_utility.csv", index=False)
        print("\n=== Privacy-Utility Evaluation ===")
        print(priv_df.to_string(index=False))
        
        # Plot trade-off
        # Filter out inf values for plotting, but represent them in labels
        plot_df = priv_df.copy()
        plot_df.loc[plot_df["Epsilon_Target"] == "No DP", "Epsilon_Actual"] = 15.0 # Plot placeholder for visual comparison
        
        plt.figure(figsize=(8, 5))
        plt.plot(plot_df["Epsilon_Actual"], plot_df["AUROC"], 'o-', color='#e74c3c', label='AUROC', lw=2)
        plt.plot(plot_df["Epsilon_Actual"], plot_df["AUPRC"], 's-', color='#3498db', label='AUPRC', lw=2)
        
        # Add labels
        for idx, row in plot_df.iterrows():
            label = f"eps={row['Epsilon_Target']}"
            x_val = row['Epsilon_Actual']
            plt.annotate(label, (x_val, row['AUROC']), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
            
        plt.xlabel("Privacy Budget (Epsilon) [Smaller means stronger privacy]")
        plt.ylabel("Performance Metric")
        plt.title("FedLite-Care: Privacy-Utility Tradeoff (Epsilon vs Utility)")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/privacy_utility_tradeoff.png", dpi=300)
        plt.close()
        print("Saved figures/privacy_utility_tradeoff.png")

if __name__ == "__main__":
    run_privacy_experiments()
