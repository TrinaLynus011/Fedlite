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

def load_adaptive_controller_config():
    try:
        with open("configs/adaptive_controller.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("adaptive_controller", {})
    except Exception:
        return {}


def save_experiment_metadata(experiment_id, model_type, client_count, partition_method,
                             alpha, metrics, duration, seed, dp_config=None, comp_config=None,
                             adaptive_enabled=False):
    if dp_config is None:
        dp_config = {"enabled": False}
    if comp_config is None:
        comp_config = {"enabled": False}

    adaptive_cfg = load_adaptive_controller_config()
    mode = "adaptive" if adaptive_enabled else "static"

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
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "git_commit": "None",
        "random_seed": seed,
        "Python_version": sys.version,
        "model_type": model_type,
        "client_count": client_count,
        "partition_method": partition_method,
        "dirichlet_alpha": alpha if partition_method == "non_iid" else None,
        "simulation_duration_sec": duration,
        "adaptive_controller": {
            "enabled": adaptive_enabled,
            "seed": seed,
            "config": adaptive_cfg,
            "candidate_space": adaptive_cfg.get("candidates", {}),
            "objective_weights": adaptive_cfg.get("objective_weights", {}),
            "round_history": metrics.get("adaptive_controller", {}).get("round_history", [])
        },
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
        "mode": mode,
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

def run_experiment(exp_id, model, clients, partition, alpha, rounds, local_epochs, batch_size, lr, dp_config, comp_config, seed, adaptive_enabled):
    print(f"\n==================================================")
    print(f" RUNNING EXPERIMENT: {exp_id}")
    print(f"==================================================")
    
    # Temporarily override adaptive controller config for this run
    with open("configs/adaptive_controller.yaml", "r") as f:
        adaptive_config = yaml.safe_load(f)
    
    adaptive_config["adaptive_controller"]["enabled"] = adaptive_enabled
    adaptive_config["adaptive_controller"]["seed"] = seed
    
    with open("configs/adaptive_controller.yaml", "w") as f:
        yaml.dump(adaptive_config, f)

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
            comp_config=comp_config,
            adaptive_enabled=adaptive_enabled
        )
        
        return {
            "Experiment": exp_id,
            "Model": model,
            "Clients": clients,
            "Partition": partition,
            "Alpha": alpha if partition == "non_iid" else "N/A",
            "AUROC": f"{metrics['auroc']:.4f}",
            "AUPRC": f"{metrics['auprc']:.4f}",
            "Privacy epsilon": metrics["privacy"]["epsilon"] if metrics["privacy"].get("dp_enabled") else "N/A",
            "Total communication bytes": metrics["communication"]["total_bytes"],
            "Training time": f"{metrics['simulation_time_sec']:.2f}",
        }
        
    except Exception as e:
        print(f"Error running experiment {exp_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_experiment_metadata(exp_id):
    path = os.path.join("results", "experiment_metadata", f"{exp_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def run_comparison():
    static_id = "FL_MLP_NONIID_5_STATIC"
    adaptive_id = "FL_MLP_NONIID_5_ADAPTIVE"

    static_meta = load_experiment_metadata(static_id)
    adaptive_meta = load_experiment_metadata(adaptive_id)

    print("\n==================================================")
    print(" COMPARISON OUTPUT")
    print("==================================================")
    print(f"| Metric                         | Static | Adaptive |")
    print(f"| ------------------------------ | -----: | -------: |")

    def format_metric(static_value, adaptive_value):
        if static_value is None:
            static_display = "NOT YET RUN"
        else:
            static_display = static_value
        if adaptive_value is None:
            adaptive_display = "NOT YET RUN"
        else:
            adaptive_display = adaptive_value
        return static_display, adaptive_display

    static_metrics = static_meta.get("metrics", {}) if static_meta else {}
    adaptive_metrics = adaptive_meta.get("metrics", {}) if adaptive_meta else {}
    static_priv = static_meta.get("privacy", {}) if static_meta else {}
    adaptive_priv = adaptive_meta.get("privacy", {}) if adaptive_meta else {}
    static_comm = static_meta.get("communication", {}) if static_meta else {}
    adaptive_comm = adaptive_meta.get("communication", {}) if adaptive_meta else {}

    static_auroc = f"{static_metrics.get('auroc', 'NOT YET RUN'):.4f}" if static_metrics.get('auroc') is not None else "NOT YET RUN"
    adaptive_auroc = f"{adaptive_metrics.get('auroc', 'NOT YET RUN'):.4f}" if adaptive_metrics.get('auroc') is not None else "NOT YET RUN"
    static_auprc = f"{static_metrics.get('auprc', 'NOT YET RUN'):.4f}" if static_metrics.get('auprc') is not None else "NOT YET RUN"
    adaptive_auprc = f"{adaptive_metrics.get('auprc', 'NOT YET RUN'):.4f}" if adaptive_metrics.get('auprc') is not None else "NOT YET RUN"
    static_eps = static_priv.get("epsilon", "NOT YET RUN")
    adaptive_eps = adaptive_priv.get("epsilon", "NOT YET RUN")
    static_bytes = static_comm.get("total_bytes_transmitted", "NOT YET RUN")
    adaptive_bytes = adaptive_comm.get("total_bytes_transmitted", "NOT YET RUN")
    static_time = static_meta.get("simulation_duration_sec", "NOT YET RUN") if static_meta else "NOT YET RUN"
    adaptive_time = adaptive_meta.get("simulation_duration_sec", "NOT YET RUN") if adaptive_meta else "NOT YET RUN"

    print(f"| AUROC                          | {static_auroc} | {adaptive_auroc} |")
    print(f"| AUPRC                          | {static_auprc} | {adaptive_auprc} |")
    print(f"| Privacy epsilon                | {static_eps} | {adaptive_eps} |")
    print(f"| Total communication bytes      | {static_bytes} | {adaptive_bytes} |")
    print(f"| Training time                  | {static_time} | {adaptive_time} |")
    print("| Mean local epochs              | NOT YET RUN | NOT YET RUN |")
    print("| Mean participation fraction    | NOT YET RUN | NOT YET RUN |")
    print("| Mean compression configuration | NOT YET RUN | NOT YET RUN |")


def run_static_experiment():
    data_config, fed_config = load_configs()
    rounds = fed_config.get("rounds", 10)
    local_epochs = fed_config.get("local_epochs", 3)
    batch_size = fed_config.get("local_batch_size", 128)
    lr = fed_config.get("local_learning_rate", 0.005)
    seed = data_config["split"]["seed"]
    dp_config = {"enabled": True, "mechanism": "dpsgd", "noise_multiplier": 2.3, "clipping_norm": 1.0, "delta": 1e-5}
    comp_config = {"enabled": True, "ratio": 1.0}
    return run_experiment("FL_MLP_NONIID_5_STATIC", "mlp", 5, "non_iid", 0.5, rounds, local_epochs, batch_size, lr, dp_config, comp_config, seed, False)


def run_adaptive_experiment():
    data_config, fed_config = load_configs()
    rounds = fed_config.get("rounds", 10)
    local_epochs = fed_config.get("local_epochs", 3)
    batch_size = fed_config.get("local_batch_size", 128)
    lr = fed_config.get("local_learning_rate", 0.005)
    seed = data_config["split"]["seed"]
    dp_config = {"enabled": True, "mechanism": "dpsgd", "noise_multiplier": 2.3, "clipping_norm": 1.0, "delta": 1e-5}
    comp_config = {"enabled": True, "ratio": 1.0}
    return run_experiment("FL_MLP_NONIID_5_ADAPTIVE", "mlp", 5, "non_iid", 0.5, rounds, local_epochs, batch_size, lr, dp_config, comp_config, seed, True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].lower() in {"compare", "comparison"}:
        run_comparison()
    elif args[0].lower() in {"static", "baseline"}:
        run_static_experiment()
    elif args[0].lower() in {"adaptive"}:
        run_adaptive_experiment()
    else:
        print("Usage: python src/federated/train_federated.py [static|adaptive|compare]")
