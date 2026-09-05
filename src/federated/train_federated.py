import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add src/ to path
sys.path.append(os.path.abspath("src"))
from federated.server import run_federated_simulation

def load_configs():
    with open("configs/data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    with open("configs/federated.yaml", "r") as f:
        fed_config = yaml.safe_load(f)
    return data_config, fed_config

def save_experiment_metadata(experiment_id, model_type, client_count, partition_method,
                             alpha, metrics, duration, seed, dp_config=None, comp_config=None):
    if dp_config is None:
        dp_config = {"enabled": False}
    if comp_config is None:
        comp_config = {"enabled": False}

    # Pull privacy telemetry that was computed by server.py's accountant
    privacy = metrics.get("privacy", {})
    dp_enabled = privacy.get("dp_enabled", False)
    mechanism = privacy.get("mechanism", "none")
    privacy_unit = privacy.get("privacy_unit", "none")
    clipping_norm = privacy.get("clipping_norm", None)
    noise_multiplier = privacy.get("noise_multiplier", None)
    delta = privacy.get("delta", None)
    epsilon = privacy.get("epsilon", None)
    rounds_accounted = privacy.get("rounds_accounted", None)

    # Format for registry display
    epsilon_str = f"{epsilon:.4f}" if (epsilon is not None and np.isfinite(epsilon)) else ("inf" if dp_enabled else "N/A")
    delta_str = str(delta) if delta is not None else "N/A"

    comp_enabled = comp_config.get("enabled", False)
    comp_ratio = comp_config.get("ratio", 1.0) if comp_enabled else 1.0
    compression_str = f"Top-K {comp_ratio:.0%}" if (comp_enabled and comp_ratio < 1.0) else "None"

    # Extract model stats
    model_stats = metrics.get("model_stats", {})
    parameter_count = model_stats.get("parameter_count", 0)
    theoretical_fp32_size_kb = model_stats.get("theoretical_fp32_size_kb", 0.0)
    theoretical_int8_size_kb = model_stats.get("theoretical_int8_size_kb", 0.0)
    serialized_checkpoint_size_kb = model_stats.get("serialized_checkpoint_size_kb", 0.0)

    metadata = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit": "None",
        "random_seed": seed,
        "Python_version": sys.version,
        "model_type": model_type,
        "client_count": client_count,
        "partition_method": partition_method,
        "dirichlet_alpha": alpha if partition_method == "non_iid" else None,
        "simulation_duration_sec": duration,
        "privacy": {
            "dp_enabled": dp_enabled,
            "mechanism": mechanism,
            "privacy_unit": privacy_unit,
            "clipping_norm": clipping_norm,
            "noise_multiplier": noise_multiplier,
            "delta": delta_str,
            "epsilon": epsilon_str,
            "rounds_accounted": rounds_accounted,
            "participating_clients": client_count,
        },
        "compression_enabled": comp_enabled,
        "compression_ratio": comp_ratio,
        "model_stats": {
            "parameter_count": parameter_count,
            "theoretical_fp32_size_kb": theoretical_fp32_size_kb,
            "theoretical_int8_size_kb": theoretical_int8_size_kb,
            "serialized_checkpoint_size_kb": serialized_checkpoint_size_kb
        },
        "metrics": {
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "accuracy": metrics["accuracy"],
            "brier_score": metrics["brier_score"],
            "confusion_matrix": metrics["confusion_matrix"]
        },
        "communication": {
            "total_bytes_transmitted": metrics["communication"]["total_bytes"],
            "bytes_upstream": metrics["communication"]["bytes_upstream"],
            "bytes_downstream": metrics["communication"]["bytes_downstream"]
        }
    }

    os.makedirs("results/experiment_metadata", exist_ok=True)
    meta_path = f"results/experiment_metadata/{experiment_id}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved experiment metadata to {meta_path}")

    # Also update the registry csv
    registry_path = "results/experiment_registry.csv"
    row = {
        "experiment_id": experiment_id,
        "model": model_type,
        "training_type": "federated",
        "client_count": client_count,
        "partition": partition_method,
        "alpha": alpha if partition_method == "non_iid" else "N/A",
        "dp_enabled": dp_enabled,
        "mechanism": mechanism,
        "privacy_unit": privacy_unit,
        "epsilon": epsilon_str,
        "delta": delta_str,
        "clipping_norm": clipping_norm if clipping_norm is not None else "N/A",
        "noise_multiplier": noise_multiplier if noise_multiplier is not None else "N/A",
        "compression": compression_str,
        "top_k": comp_ratio,
        "quantization": "None",
        "parameter_count": parameter_count,
        "theoretical_fp32_size_kb": theoretical_fp32_size_kb,
        "theoretical_int8_size_kb": theoretical_int8_size_kb,
        "serialized_checkpoint_size_kb": serialized_checkpoint_size_kb,
        "seed": seed,
        "status": "FINAL_LOCKED",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": "None"
    }

    new_df = pd.DataFrame([row])
    if os.path.exists(registry_path):
        reg_df = pd.read_csv(registry_path)
        reg_df = reg_df[reg_df["experiment_id"] != experiment_id]
        reg_df = pd.concat([reg_df, new_df], ignore_index=True)
    else:
        reg_df = new_df

    reg_df.to_csv(registry_path, index=False)
    print(f"Updated registry in {registry_path}")

def run_all_federated_experiments():
    data_config, fed_config = load_configs()
    rounds = fed_config.get("rounds", 10)
    local_epochs = fed_config.get("local_epochs", 3)
    batch_size = fed_config.get("local_batch_size", 128)
    lr = fed_config.get("local_learning_rate", 0.005)
    seed = data_config["split"]["seed"]
    
    # Define experiment matrix
    experiments = [
        # LR Experiments
        ("FL_LR_IID_5", "lr", 5, "iid", 0.5),
        ("FL_LR_IID_10", "lr", 10, "iid", 0.5),
        ("FL_LR_NONIID_5", "lr", 5, "non_iid", 0.5),
        ("FL_LR_NONIID_10", "lr", 10, "non_iid", 0.5),
        
        # MLP Experiments
        ("FL_MLP_IID_5", "mlp", 5, "iid", 0.5),
        ("FL_MLP_IID_10", "mlp", 10, "iid", 0.5),
        ("FL_MLP_NONIID_5", "mlp", 5, "non_iid", 0.5),
        ("FL_MLP_NONIID_10", "mlp", 10, "non_iid", 0.5),
    ]
    
    fl_results = []
    
    for exp_id, model, clients, partition, alpha in experiments:
        print(f"\n==================================================")
        print(f" RUNNING EXPERIMENT: {exp_id}")
        print(f"==================================================")
        
        # Configure privacy and compression as Disabled for standard runs
        dp_config = {"enabled": False, "mechanism": "client_update_gaussian"}
        comp_config = {"enabled": False}
        
        try:
            metrics, checkpoint_path = run_federated_simulation(
                experiment_id=exp_id,
                model_type=model,
                client_count=clients,
                partition_method=partition,
                alpha=alpha,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                dp_config=dp_config,
                compression_config=comp_config,
                seed=seed
            )
            
            # Save metadata and update registry
            save_experiment_metadata(
                experiment_id=exp_id,
                model_type=model,
                client_count=clients,
                partition_method=partition,
                alpha=alpha,
                metrics=metrics,
                duration=metrics["simulation_time_sec"],
                seed=seed,
                dp_config=dp_config,
                comp_config=comp_config
            )
            
            fl_results.append({
                "Experiment": exp_id,
                "Model": model,
                "Clients": clients,
                "Partition": partition,
                "Alpha": alpha if partition == "non_iid" else "N/A",
                "AUROC": f"{metrics['auroc']:.4f}",
                "AUPRC": f"{metrics['auprc']:.4f}",
                "F1-Score": f"{metrics['f1']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Communication (MB)": f"{metrics['communication']['total_bytes'] / (1024*1024):.2f}"
            })
            
        except Exception as e:
            print(f"Error running experiment {exp_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Save a comparison summary CSV
    if len(fl_results) > 0:
        summary_df = pd.DataFrame(fl_results)
        summary_df.to_csv("results/federated_comparison.csv", index=False)
        print("\n=== Federated Experiments Comparison ===")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_all_federated_experiments()
