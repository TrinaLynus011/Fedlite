"""
FedLite-Care — DP Calibration Experiments
==========================================
Three controlled IID-5 experiments on MLP:

  A. No DP                  (noise_multiplier=0.0)
  B. Client-Update Gaussian (noise_multiplier=0.8)  -- calibration only
  C. Client-Update Gaussian (noise_multiplier=3.0)

All experiments share:
  model_type  = mlp
  clients     = 5
  partition   = IID
  rounds      = 10
  local_epochs= 3
  batch_size  = 128
  lr          = 0.005
  clipping_norm=1.0
  delta       = 1e-5
  seed        = 42

Telemetry collected per experiment:
  - epsilon (from accountant)
  - AUROC, AUPRC, F1, precision, recall, specificity, balanced_accuracy
  - model parameter count + size
  - total communication bytes
  - training time
  - per-round AUROC / AUPRC
  - average pre-clip norm (mean across clients and rounds)
  - fraction of client updates that were clipped (pre-clip norm > C)

Results are saved to:
  results/dp_calibration_results.json
  results/dp_calibration_summary.csv
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath("src"))

from models.mlp import LightweightMLP
from data.partition import partition_data
from utils.metrics import calculate_metrics
from privacy.dp_mechanism import clip_and_add_noise, compute_client_update_epsilon
from compression.sparsification import compress_updates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_evaluate_fn(input_dim, hidden_dims, dropout, X_val, y_val):
    """Returns a callable that evaluates global params on the validation set."""
    from sklearn.metrics import f1_score as _f1

    def evaluate(parameters):
        model = LightweightMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(np.copy(v)) for k, v in params_dict}
        model.load_state_dict(state_dict)
        model.eval()

        X_t = torch.FloatTensor(np.array(X_val))
        y_arr = np.array(y_val)

        with torch.no_grad():
            logits = model(X_t)
            probs = torch.sigmoid(logits).squeeze(1).numpy()

        best_thresh, best_f1 = 0.5, -1.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            f1_t = _f1(y_arr, (probs >= thresh).astype(int), zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = thresh

        m = calculate_metrics(y_arr, probs, threshold=best_thresh)
        return m, probs

    return evaluate


def train_one_client(X_c, y_c, global_params, input_dim, hidden_dims, dropout,
                     local_epochs, batch_size, lr, seed_offset,
                     dp_enabled, clipping_norm, noise_mult, dp_seed):
    """
    Run one client's local training and return (final_params, num_samples,
    pre_clip_norm, was_clipped).
    """
    model = LightweightMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    # Load global parameters
    params_dict = zip(model.state_dict().keys(), global_params)
    state_dict = {k: torch.tensor(np.copy(v)) for k, v in params_dict}
    model.load_state_dict(state_dict)

    old_params = [np.copy(v.detach().cpu().numpy()) for v in model.parameters()]

    num_pos = int((y_c == 1).sum())
    num_neg = int((y_c == 0).sum())
    if num_pos > 0 and num_neg > 0:
        pw = torch.tensor(num_neg / num_pos, dtype=torch.float32)
    elif num_pos == 0:
        pw = torch.tensor(7.680996884735203, dtype=torch.float32)
    else:
        pw = torch.tensor(1.0 / 7.680996884735203, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.FloatTensor(np.copy(X_c))
    y_t = torch.FloatTensor(np.copy(y_c)).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    g = torch.Generator()
    g.manual_seed(seed_offset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

    model.train()
    for _ in range(local_epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    new_params = [v.detach().cpu().numpy().copy() for v in model.parameters()]
    deltas = [n - o for n, o in zip(new_params, old_params)]

    pre_clip_norm = float(np.sqrt(sum(np.sum(d ** 2) for d in deltas)))
    was_clipped = pre_clip_norm > clipping_norm

    if dp_enabled and noise_mult > 0.0:
        deltas, _ = clip_and_add_noise(deltas, clipping_norm, noise_mult, seed=dp_seed)
    elif dp_enabled and noise_mult == 0.0:
        deltas, _ = clip_and_add_noise(deltas, clipping_norm, 0.0)

    final_params = [o + d for o, d in zip(old_params, deltas)]
    return final_params, len(X_c), pre_clip_norm, was_clipped


def run_calibration_experiment(label, model_type, client_count, partition_method,
                                alpha, rounds, local_epochs, batch_size, lr,
                                dp_enabled, clipping_norm, noise_mult, delta, seed,
                                X_train, y_train, X_val, y_val, input_dim,
                                hidden_dims, dropout, client_dfs, features):
    print(f"\n{'='*60}")
    print(f"  CALIBRATION EXPERIMENT: {label}")
    print(f"  DP={'ON' if dp_enabled else 'OFF'} | sigma={noise_mult} | C={clipping_norm}")
    print(f"{'='*60}")

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Global model
    global_model = LightweightMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    global_params = [v.detach().cpu().numpy().copy() for _, v in global_model.state_dict().items()]

    evaluate_fn = get_evaluate_fn(input_dim, hidden_dims, dropout, X_val, y_val)

    # Per-client validation subsets (same deterministic subsampling as server.py)
    client_val_data = []
    for idx in range(client_count):
        np.random.seed(seed + idx)
        v_idx = np.random.choice(len(X_val), int(len(X_val) / client_count), replace=False)
        client_val_data.append((X_val[v_idx], y_val[v_idx]))

    round_records = []
    all_pre_clip_norms = []
    all_was_clipped = []

    start = time.time()

    for r in range(1, rounds + 1):
        client_updates = []
        round_pre_norms = []
        round_was_clipped = []

        for idx in range(client_count):
            df_c = client_dfs[idx]
            X_c = df_c[features].values
            y_c = df_c["readmitted"].values

            seed_offset = seed + r * 1000 + idx
            dp_seed = (seed_offset + 99991) if (dp_enabled and noise_mult > 0.0) else None

            f_params, n_samples, pre_norm, clipped = train_one_client(
                X_c, y_c, global_params, input_dim, hidden_dims, dropout,
                local_epochs, batch_size, lr, seed_offset,
                dp_enabled, clipping_norm, noise_mult, dp_seed
            )
            client_updates.append((f_params, n_samples))
            round_pre_norms.append(pre_norm)
            round_was_clipped.append(clipped)

        all_pre_clip_norms.extend(round_pre_norms)
        all_was_clipped.extend(round_was_clipped)

        # FedAvg aggregation
        n_layers = len(global_params)
        new_global = []
        total_w = sum(w for _, w in client_updates)
        for li in range(n_layers):
            agg = sum(p[li] * w for p, w in client_updates) / total_w
            new_global.append(agg)
        global_params = new_global

        # Evaluate on validation
        m, _ = evaluate_fn(global_params)
        print(f"  Round {r:02d}/{rounds} | AUROC={m['auroc']:.4f} | "
              f"AUPRC={m['auprc']:.4f} | "
              f"avg_pre_clip_norm={np.mean(round_pre_norms):.4f} | "
              f"frac_clipped={np.mean(round_was_clipped):.2f}")

        round_records.append({
            "round": r,
            "val_auroc": m["auroc"],
            "val_auprc": m["auprc"],
            "val_f1": m["f1"],
            "avg_pre_clip_norm": float(np.mean(round_pre_norms)),
            "frac_clipped": float(np.mean(round_was_clipped)),
        })

    elapsed = time.time() - start

    # Final metrics on validation
    final_m, final_probs = evaluate_fn(global_params)

    # Epsilon
    if dp_enabled and noise_mult > 0.0:
        epsilon = compute_client_update_epsilon(rounds=rounds, noise_multiplier=noise_mult, delta=delta)
    else:
        epsilon = float("inf")

    # Model stats
    final_model = LightweightMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    n_params = sum(p.numel() for p in final_model.parameters())
    fp32_kb = (n_params * 4) / 1024.0
    int8_kb = (n_params * 1) / 1024.0

    # Communication bytes (no compression)
    param_bytes = n_params * 4
    total_bytes = rounds * client_count * param_bytes * 2  # up + down

    result = {
        "label": label,
        "dp_enabled": dp_enabled,
        "mechanism": "client_update_gaussian" if dp_enabled else "none",
        "privacy_unit": "client" if dp_enabled else "none",
        "noise_multiplier": noise_mult,
        "clipping_norm": clipping_norm if dp_enabled else None,
        "delta": delta if dp_enabled else None,
        "epsilon": epsilon if dp_enabled else None,
        "rounds": rounds,
        "auroc": final_m["auroc"],
        "auprc": final_m["auprc"],
        "f1": final_m["f1"],
        "precision": final_m["precision"],
        "recall": final_m["recall"],
        "specificity": final_m["specificity"],
        "balanced_accuracy": final_m["balanced_accuracy"],
        "brier_score": final_m["brier_score"],
        "accuracy": final_m["accuracy"],
        "n_params": n_params,
        "fp32_size_kb": fp32_kb,
        "int8_size_kb": int8_kb,
        "total_bytes_transmitted": total_bytes,
        "training_time_sec": elapsed,
        "avg_pre_clip_norm_all_rounds": float(np.mean(all_pre_clip_norms)),
        "frac_updates_clipped_all_rounds": float(np.mean(all_was_clipped)),
        "per_round": round_records,
    }

    print(f"\n  FINAL | epsilon={epsilon:.4f} | AUROC={final_m['auroc']:.4f} | "
          f"AUPRC={final_m['auprc']:.4f} | F1={final_m['f1']:.4f}")
    print(f"         avg_pre_clip_norm={np.mean(all_pre_clip_norms):.4f} | "
          f"frac_clipped={np.mean(all_was_clipped):.3f}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import random

    # Load data
    processed_dir = "data/processed"
    train_df = pd.read_csv(os.path.join(processed_dir, "train_preprocessed.csv"))
    val_df   = pd.read_csv(os.path.join(processed_dir, "val_preprocessed.csv"))

    target_col = "readmitted"
    features = [c for c in train_df.columns if c != target_col]
    input_dim = len(features)

    X_train = train_df[features].values
    y_train = train_df[target_col].values
    X_val   = val_df[features].values
    y_val   = val_df[target_col].values

    with open("configs/models.yaml") as f:
        mcfg = yaml.safe_load(f)
    hidden_dims = mcfg["mlp"]["hidden_dims"]
    dropout     = float(mcfg["mlp"]["dropout"])

    # Shared config
    client_count   = 5
    partition      = "iid"
    alpha          = 0.5
    rounds         = 10
    local_epochs   = 3
    batch_size     = 128
    lr             = 0.005
    clipping_norm  = 1.0
    delta          = 1e-5
    seed           = 42

    # Partition once — same partition for all three experiments
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    client_dfs = partition_data(train_df, client_count, partition, alpha, seed)
    print(f"IID partition sizes: {[len(df) for df in client_dfs]}")

    experiments = [
        # (label, dp_enabled, noise_mult)
        ("CAL_MLP_IID_5_NoDB",      False, 0.0),
        ("CAL_MLP_IID_5_sigma_0.8", True,  0.8),
        ("CAL_MLP_IID_5_sigma_3.0", True,  3.0),
    ]

    all_results = []
    for label, dp_en, nm in experiments:
        res = run_calibration_experiment(
            label=label,
            model_type="mlp",
            client_count=client_count,
            partition_method=partition,
            alpha=alpha,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            dp_enabled=dp_en,
            clipping_norm=clipping_norm,
            noise_mult=nm,
            delta=delta,
            seed=seed,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            client_dfs=client_dfs,
            features=features,
        )
        all_results.append(res)

    # Save full JSON
    os.makedirs("results", exist_ok=True)
    with open("results/dp_calibration_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/dp_calibration_results.json")

    # Save summary CSV
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "label":             r["label"],
            "dp_enabled":        r["dp_enabled"],
            "noise_multiplier":  r["noise_multiplier"],
            "epsilon":           f"{r['epsilon']:.4f}" if r["epsilon"] else "N/A",
            "auroc":             f"{r['auroc']:.4f}",
            "auprc":             f"{r['auprc']:.4f}",
            "f1":                f"{r['f1']:.4f}",
            "precision":         f"{r['precision']:.4f}",
            "recall":            f"{r['recall']:.4f}",
            "specificity":       f"{r['specificity']:.4f}",
            "balanced_accuracy": f"{r['balanced_accuracy']:.4f}",
            "brier_score":       f"{r['brier_score']:.4f}",
            "n_params":          r["n_params"],
            "fp32_kb":           f"{r['fp32_size_kb']:.2f}",
            "total_bytes_MB":    f"{r['total_bytes_transmitted']/(1024*1024):.2f}",
            "training_time_sec": f"{r['training_time_sec']:.1f}",
            "avg_pre_clip_norm": f"{r['avg_pre_clip_norm_all_rounds']:.4f}",
            "frac_clipped":      f"{r['frac_updates_clipped_all_rounds']:.4f}",
        })
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv("results/dp_calibration_summary.csv", index=False)
    print("Saved results/dp_calibration_summary.csv")

    # Print comparison table
    print("\n" + "="*80)
    print("  CALIBRATION COMPARISON TABLE")
    print("="*80)
    print(df_sum.to_string(index=False))

    # Per-round AUROC table
    print("\n" + "="*80)
    print("  PER-ROUND AUROC")
    print("="*80)
    round_table = {"round": list(range(1, rounds + 1))}
    for res in all_results:
        col_label = f"{res['label'].replace('CAL_MLP_IID_5_','')}_AUROC"
        round_table[col_label] = [r["val_auroc"] for r in res["per_round"]]
        col_auprc = f"{res['label'].replace('CAL_MLP_IID_5_','')}_AUPRC"
        round_table[col_auprc] = [r["val_auprc"] for r in res["per_round"]]
    df_rounds = pd.DataFrame(round_table)
    print(df_rounds.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
