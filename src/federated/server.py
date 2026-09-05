"""
FedLite-Care Federated Server — Lightweight Sequential Simulator
================================================================
This module implements a CUSTOM SEQUENTIAL federated learning simulator.
It does NOT use the Flower (flwr) distributed runtime or Ray actors.

Design rationale:
- The project targets 4 GB RAM edge-compatible machines.
- Flower's simulation backend spawns Ray actors per client, which adds
  significant memory overhead not suitable for the target environment.
- The sequential loop is deterministic, reproducible, and memory-efficient.
- The simulation models FedAvg correctly: all clients train serially in
  each round, updates are aggregated by sample-weighted averaging.

IMPORTANT: This is a research simulation, NOT a distributed deployment.
  - Communication cost figures (bytes_upstream / bytes_downstream) are
    ESTIMATED based on parameter count × dtype size.
  - Top-K sparsification reduces the REPORTED byte count for upstream
    communication; actual Python objects are still full-size in memory.
  - To deploy to real federated infrastructure, the client/server split
    would need an actual network transport layer (gRPC, REST, etc.).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys
import yaml
import json
import time

# Add src/ to path
sys.path.append(os.path.abspath("src"))
from models.mlp import LightweightMLP
from models.logistic_regression import PyTorchLR
from utils.metrics import calculate_metrics, save_evaluation_plots
from data.partition import partition_data
from federated.client import FedLiteClient
from privacy.dp_mechanism import compute_client_update_epsilon, compute_dpsgd_epsilon

def get_evaluate_fn(model_type, input_dim, hidden_dims, X_val, y_val):
    """
    Returns an evaluation function for centralized server-side evaluation.
    Runs on the VALIDATION set only. Threshold is tuned on validation to maximise F1.
    Test set is never touched here.
    """
    # Load dropout dynamically from models config
    dropout = 0.2
    try:
        with open("configs/models.yaml", "r") as f:
            dropout = float(yaml.safe_load(f)["mlp"]["dropout"])
    except Exception:
        pass

    def evaluate(server_round, parameters, config):
        if model_type == "mlp":
            model = LightweightMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        else:
            model = PyTorchLR(input_dim=input_dim)
            
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict)
        model.eval()
        
        X_val_t = torch.FloatTensor(np.array(X_val))
        y_val_arr = np.array(y_val)
        y_val_t = torch.FloatTensor(y_val_arr).unsqueeze(1)
        
        with torch.no_grad():
            outputs = model(X_val_t)
            # PyTorchLR applies sigmoid internally; MLP outputs raw logits
            if model_type == "mlp":
                y_prob = torch.sigmoid(outputs).squeeze(1).numpy()
                criterion = nn.BCEWithLogitsLoss()
            else:
                y_prob = outputs.squeeze(1).numpy()   # already probabilities
                criterion = nn.BCELoss()
            val_loss = criterion(outputs, y_val_t).item()

        # Tune threshold on validation set (maximise F1) — never use test set here
        best_thresh, best_f1 = 0.5, -1.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred_t = (y_prob >= thresh).astype(int)
            from sklearn.metrics import f1_score as _f1
            f1_t = _f1(y_val_arr, y_pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = thresh

        metrics = calculate_metrics(y_val_arr, y_prob, threshold=best_thresh)
        metrics["threshold"] = float(best_thresh)
            
        return float(val_loss), {
            "auroc": float(metrics["auroc"]),
            "auprc": float(metrics["auprc"]),
            "f1": float(metrics["f1"]),
            "recall": float(metrics["recall"]),
            "precision": float(metrics["precision"]),
            "specificity": float(metrics["specificity"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "brier_score": float(metrics["brier_score"]),
            "accuracy": float(metrics["accuracy"]),
            "threshold": float(best_thresh)
        }
    return evaluate

def fit_config(server_round):
    # Pass round info to client
    return {"server_round": server_round}

def run_federated_simulation(experiment_id, model_type="mlp", client_count=5,
                             partition_method="iid", alpha=0.5, rounds=10,
                             local_epochs=3, batch_size=128, lr=0.005,
                             dp_config=None, compression_config=None, seed=42):
    
    # 0. Set deterministic seeds globally for PyTorch, NumPy, Python random
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load processed splits
    processed_dir = "data/processed"
    train_df = pd.read_csv(os.path.join(processed_dir, "train_preprocessed.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val_preprocessed.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_preprocessed.csv"))
    
    target_col = "readmitted"
    features = [col for col in train_df.columns if col != target_col]
    input_dim = len(features)
    
    X_train = train_df[features].values
    y_train = train_df[target_col].values
    
    X_val = val_df[features].values
    y_val = val_df[target_col].values
    
    X_test = test_df[features].values
    y_test = test_df[target_col].values
    
    # Hidden dimensions from models config
    with open("configs/models.yaml", "r") as f:
        hidden_dims = yaml.safe_load(f)["mlp"]["hidden_dims"]
        
    # 1. Partition Data
    print(f"Partitioning training dataset ({len(train_df)} rows) into {client_count} clients using '{partition_method}'...")
    client_dfs = partition_data(train_df, client_count, partition_method, alpha, seed)
    
    # 2. Instantiate Clients
    clients = []
    for idx in range(client_count):
        c_df = client_dfs[idx]
        X_c = c_df[features].values
        y_c = c_df[target_col].values
        
        # Subsample validation data deterministically
        np.random.seed(seed + idx)
        v_idx = np.random.choice(len(X_val), int(len(X_val) / client_count), replace=False)
        X_c_val = X_val[v_idx]
        y_c_val = y_val[v_idx]
        
        client = FedLiteClient(
            client_id=idx,
            X_train=X_c,
            y_train=y_c,
            X_val=X_c_val,
            y_val=y_c_val,
            model_type=model_type,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            dp_config=dp_config,
            compression_config=compression_config
        )
        clients.append(client)
        
    # Load dropout dynamically from models config
    dropout = 0.2
    try:
        with open("configs/models.yaml", "r") as f:
            dropout = float(yaml.safe_load(f)["mlp"]["dropout"])
    except Exception:
        pass

    # Initialize global parameters deterministically
    if model_type == "mlp":
        global_model = LightweightMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    else:
        global_model = PyTorchLR(input_dim=input_dim)
        
    global_parameters = [val.cpu().detach().numpy().copy() for _, val in global_model.state_dict().items()]
    
    # Define evaluation function
    evaluate_fn = get_evaluate_fn(model_type, input_dim, hidden_dims, X_val, y_val)
    
    print(f"Starting Custom Sequential FedAvg Simulation: {experiment_id} | Rounds: {rounds}...")
    start_sim_time = time.time()
    
    round_telemetry = []
    federated_round_metrics = []
    
    for r in range(1, rounds + 1):
        client_updates = []
        total_fit_samples = 0
        round_sent = 0
        round_recv = 0
        
        round_start_time = time.time()
        round_mean_grad_norms   = []
        round_median_grad_norms = []
        round_frac_clipped      = []

        for idx, client in enumerate(clients):
            # Broadcast global parameters to client (server -> client)
            # Downstream byte count (all layers parameters as 32-bit floats)
            param_count = sum(p.size for p in global_parameters)
            round_recv += param_count * 4
            
            # Pass a deterministic seed for loader shuffling
            fit_res = client.fit(global_parameters, config={"seed": seed + r * 1000 + idx})
            client_params, num_samples, client_metrics = fit_res
            
            client_updates.append((client_params, num_samples))
            total_fit_samples += num_samples
            
            # Upstream byte count (client -> server)
            round_sent += client_metrics["bytes_sent"]

            # Collect per-round gradient telemetry (DP-SGD)
            if client_metrics.get("mean_grad_norm") is not None:
                round_mean_grad_norms.append(client_metrics["mean_grad_norm"])
                round_median_grad_norms.append(client_metrics["median_grad_norm"])
                round_frac_clipped.append(client_metrics["frac_clipped"])
            
        round_training_time = time.time() - round_start_time
            
        # FedAvg Aggregation
        new_global_parameters = []
        num_layers = len(global_parameters)
        for layer_idx in range(num_layers):
            layer_updates = [u[layer_idx] * weight for u, weight in client_updates]
            aggregated_layer = sum(layer_updates) / sum(w for _, w in client_updates)
            new_global_parameters.append(aggregated_layer)
            
        global_parameters = new_global_parameters
        
        # Evaluate global model on server validation set
        val_loss, val_metrics = evaluate_fn(r, global_parameters, config={})
        print(f"Round {r:02d}/{rounds:02d} - Loss: {val_loss:.4f} | Val AUROC: {val_metrics['auroc']:.4f} | Val AUPRC: {val_metrics['auprc']:.4f}")
        
        # Collect client details
        participating_clients = len(clients)
        total_client_samples = sum(len(c.train_dataset) for c in clients)
        
        # Collect round-by-round validation metrics
        round_metrics_dict = {
            "round": r,
            "validation_loss": val_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_brier_score": val_metrics["brier_score"],
            "val_accuracy": val_metrics["accuracy"],
            "threshold": float(val_metrics.get("threshold", 0.5)),
            "participating_clients": participating_clients,
            "total_client_samples": total_client_samples,
            "bytes_upstream": round_sent,
            "bytes_downstream": round_recv,
            "total_bytes": round_sent + round_recv,
            "compression_ratio": float(1.0 / compression_config.get("ratio", 1.0)) if (compression_config and compression_config.get("enabled", False) and compression_config.get("ratio", 1.0) < 1.0) else 1.0,
            "training_time_sec": round_training_time,
            # DP-SGD gradient telemetry (None when DP-SGD is not active)
            "mean_grad_norm":   float(np.mean(round_mean_grad_norms))   if round_mean_grad_norms   else None,
            "median_grad_norm": float(np.mean(round_median_grad_norms)) if round_median_grad_norms else None,
            "frac_clipped":     float(np.mean(round_frac_clipped))      if round_frac_clipped      else None,
            "clipping_norm":    float((dp_config or {}).get("clipping_norm", 1.0)) if (dp_config or {}).get("enabled", False) else None,
            "noise_multiplier": float((dp_config or {}).get("noise_multiplier", 0.0)) if (dp_config or {}).get("enabled", False) else None,
        }
        federated_round_metrics.append(round_metrics_dict)
        
        round_telemetry.append({
            "round": r,
            "clients_participated": client_count,
            "bytes_upstream": round_sent,
            "bytes_downstream": round_recv,
            "total_bytes": round_sent + round_recv
        })
        
    simulation_duration = time.time() - start_sim_time
    print(f"Simulation completed in {simulation_duration:.2f} seconds.")
    
    # Save round metrics to files
    os.makedirs("results", exist_ok=True)
    round_metrics_df = pd.DataFrame(federated_round_metrics)
    csv_round_path = f"results/{experiment_id}_round_metrics.csv"
    json_round_path = f"results/{experiment_id}_round_metrics.json"
    round_metrics_df.to_csv(csv_round_path, index=False)
    with open(json_round_path, "w") as f:
        json.dump(federated_round_metrics, f, indent=2)
        
    # Also save to generic master file
    round_metrics_df.to_csv("results/federated_round_metrics.csv", index=False)
    with open("results/federated_round_metrics.json", "w") as f:
        json.dump(federated_round_metrics, f, indent=2)
    print(f"Saved round validation telemetry to {csv_round_path} and master files.")
    
    # Save final model weights
    if model_type == "mlp":
        final_model = LightweightMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    else:
        final_model = PyTorchLR(input_dim=input_dim)
        
    params_dict = zip(final_model.state_dict().keys(), global_parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    final_model.load_state_dict(state_dict)
    
    # Save checkpoint
    os.makedirs("artifacts/checkpoints/federated", exist_ok=True)
    if model_type == "mlp":
        checkpoint_path = f"artifacts/checkpoints/federated/mlp_{experiment_id}.pt"
        torch.save(final_model.state_dict(), checkpoint_path)
    else:
        checkpoint_path = f"artifacts/checkpoints/federated/lr_{experiment_id}.pt"
        torch.save(final_model.state_dict(), checkpoint_path)
    print(f"Saved federated checkpoint to {checkpoint_path}")
    
    # Model size calculations (Theoretical vs Serialized)
    model_param_count = sum(p.numel() for p in final_model.parameters())
    theoretical_fp32_size_kb = (model_param_count * 4) / 1024.0
    theoretical_int8_size_kb = (model_param_count * 1) / 1024.0
    serialized_checkpoint_size_kb = float(os.path.getsize(checkpoint_path) / 1024.0)
    
    # Evaluate final model on VALIDATION set to produce final development metrics.
    # Test set is reserved for final locked evaluation only and must NOT be used
    # for model selection, round selection, or threshold tuning.
    final_model.eval()
    X_val_t = torch.FloatTensor(np.array(X_val))
    y_val_arr = np.array(y_val)
    with torch.no_grad():
        val_outputs = final_model(X_val_t)
        if model_type == "mlp":
            val_probs = torch.sigmoid(val_outputs).squeeze(1).numpy()
        else:
            val_probs = val_outputs.squeeze(1).numpy()  # PyTorchLR applies sigmoid internally

    # Find best F1 threshold on validation
    from sklearn.metrics import f1_score as _f1
    best_thresh, best_f1_val = 0.5, -1.0
    for thresh in np.arange(0.1, 0.9, 0.05):
        f1_t = _f1(y_val_arr, (val_probs >= thresh).astype(int), zero_division=0)
        if f1_t > best_f1_val:
            best_f1_val = f1_t
            best_thresh = thresh

    final_val_metrics = calculate_metrics(y_val_arr, val_probs, threshold=best_thresh)
    final_val_metrics["threshold"] = best_thresh
    final_val_metrics["simulation_time_sec"] = simulation_duration
    final_val_metrics["split"] = "validation"  # explicit label

    final_val_metrics["model_stats"] = {
        "parameter_count": model_param_count,
        "theoretical_fp32_size_kb": theoretical_fp32_size_kb,
        "theoretical_int8_size_kb": theoretical_int8_size_kb,
        "serialized_checkpoint_size_kb": serialized_checkpoint_size_kb
    }
    
    # Calculate total communication
    total_sent = sum(item["bytes_upstream"] for item in round_telemetry)
    total_received = sum(item["bytes_downstream"] for item in round_telemetry)

    # ------------------------------------------------------------------
    # Privacy accounting
    # ------------------------------------------------------------------
    dp_enabled    = (dp_config or {}).get("enabled", False)
    mechanism     = (dp_config or {}).get("mechanism", "dpsgd")
    privacy_unit  = (dp_config or {}).get("privacy_unit", "record" if mechanism == "dpsgd" else "client")
    clipping_norm = float((dp_config or {}).get("clipping_norm", 1.0))
    noise_mult    = float((dp_config or {}).get("noise_multiplier", 0.0))
    delta         = float((dp_config or {}).get("delta", 1e-5))

    reported_epsilon = float("inf")
    accounted_steps  = 0

    if dp_enabled and noise_mult > 0.0:
        if mechanism == "dpsgd":
            # Use the per-client average dataset size for accounting.
            # This is a conservative choice (largest dataset → most steps → higher ε).
            # We use the actual sizes if available; fall back to total/clients.
            avg_client_size = int(len(X_train) / client_count)
            reported_epsilon, accounted_steps = compute_dpsgd_epsilon(
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                dataset_size=avg_client_size,
                noise_multiplier=noise_mult,
                delta=delta,
            )
        elif mechanism == "client_update_gaussian":
            reported_epsilon = compute_client_update_epsilon(
                rounds=rounds,
                noise_multiplier=noise_mult,
                delta=delta,
            )

    privacy_telemetry = {
        "dp_enabled":          dp_enabled,
        "mechanism":           mechanism           if dp_enabled else "none",
        "privacy_unit":        privacy_unit        if dp_enabled else "none",
        "clipping_norm":       clipping_norm       if dp_enabled else None,
        "noise_multiplier":    noise_mult          if dp_enabled else None,
        "delta":               delta               if dp_enabled else None,
        "epsilon":             reported_epsilon    if dp_enabled else None,
        "rounds_accounted":    rounds              if dp_enabled else None,
        "steps_accounted":     accounted_steps     if dp_enabled else None,
        "participating_clients": client_count,
        "total_bytes_transmitted": total_sent + total_received,
    }

    final_val_metrics["communication"] = {
        "bytes_upstream": total_sent,
        "bytes_downstream": total_received,
        "total_bytes": total_sent + total_received,
        "telemetry_by_round": round_telemetry,
    }
    final_val_metrics["privacy"] = privacy_telemetry
    
    # Save diagnostic plots against the validation set.
    # NOTE: y_test / X_test are loaded but intentionally NOT used for metric reporting.
    # They remain available only if a separate final_evaluation.py script is run explicitly.
    save_evaluation_plots(y_val_arr, val_probs, experiment_id)
    
    return final_val_metrics, checkpoint_path

