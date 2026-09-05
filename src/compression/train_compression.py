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

def run_compression_experiments():
    with open("configs/data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    with open("configs/federated.yaml", "r") as f:
        fed_config = yaml.safe_load(f)
    with open("configs/compression.yaml", "r") as f:
        comp_config = yaml.safe_load(f)
        
    rounds = fed_config.get("rounds", 10)
    local_epochs = fed_config.get("local_epochs", 3)
    batch_size = fed_config.get("local_batch_size", 128)
    lr = fed_config.get("local_learning_rate", 0.005)
    seed = data_config["split"]["seed"]
    
    # 5 clients, non_iid
    client_count = 5
    partition_method = "non_iid"
    alpha = 0.5
    
    ratios = comp_config["ratios"]
    results = []
    
    for r in ratios:
        exp_id = f"FL_MLP_NONIID_5_comp_ratio_{r}".replace(".", "_")
        print(f"\n==================================================")
        print(f" RUNNING COMPRESSION EXPERIMENT: {exp_id} (Ratio={r:.2%})")
        print(f"==================================================")
        
        compression_settings = {
            "enabled": True,
            "ratio": r
        }
        
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
                dp_config={"enabled": False},
                compression_config=compression_settings,
                seed=seed
            )
            
            # Save metadata and update registry
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
                "compression_enabled": True,
                "compression_ratio": r,
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
                "dp_enabled": False,
                "epsilon": "N/A",
                "delta": "N/A",
                "compression": f"Top-K {r:.0%}" if r < 1.0 else "None",
                "top_k": r,
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
            
            bytes_before = metrics["communication"]["bytes_upstream"] + metrics["communication"]["bytes_downstream"]
            # Apply compression ratio to upstream bytes (server downloads are typically uncompressed broadcast)
            # Upstream bytes are what the client sends. Downstream bytes are broadcast.
            # In our client code, self.bytes_sent was compressed, while self.bytes_received was uncompressed.
            # So actual bytes sent by client is compressed, bytes received is original.
            upstream = metrics["communication"]["bytes_upstream"]
            downstream = metrics["communication"]["bytes_downstream"]
            
            total_comm_bytes = upstream + downstream
            reduction = 1.0 - (total_comm_bytes / (upstream / r + downstream if r < 1.0 else total_comm_bytes))
            
            results.append({
                "Top_K_Ratio": r,
                "Upstream_Bytes": upstream,
                "Downstream_Bytes": downstream,
                "Total_Bytes": total_comm_bytes,
                "Compression_Ratio": float(1.0 / r if r > 0 else 1.0),
                "Percent_Reduction": float(reduction * 100),
                "AUROC": metrics["auroc"],
                "AUPRC": metrics["auprc"],
                "F1": metrics["f1"]
            })
            
        except Exception as e:
            print(f"Error running compression experiment: {str(e)}")
            
    # Save CSV
    if len(results) > 0:
        comp_df = pd.DataFrame(results)
        comp_df.to_csv("results/compression_results.csv", index=False)
        print("\n=== Communication Compression Evaluation ===")
        print(comp_df.to_string(index=False))
        
        # Plot communication comparison
        plt.figure(figsize=(8, 5))
        # Bar chart comparing total bytes
        labels = [f"Top-K {r:.0%}" if r < 1.0 else "Uncompressed" for r in comp_df["Top_K_Ratio"]]
        sizes_mb = comp_df["Total_Bytes"] / (1024.0 * 1024.0)
        
        bars = plt.bar(labels, sizes_mb, color='#2980b9', width=0.4)
        for bar, size, pct in zip(bars, sizes_mb, comp_df["Percent_Reduction"]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.1, 
                     f"{size:.2f} MB\n(-{pct:.1f}%)" if pct > 0 else f"{size:.2f} MB", 
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
                     
        plt.ylabel("Total Transmitted Data (MB)")
        plt.title("FedLite-Care: Communication Overhead vs Sparsification Ratio")
        plt.ylim(0, max(sizes_mb) * 1.25)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/communication_comparison.png", dpi=300)
        plt.close()
        print("Saved figures/communication_comparison.png")

if __name__ == "__main__":
    run_compression_experiments()
