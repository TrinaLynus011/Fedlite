import os
import sys
import json

sys.path.append(os.path.abspath("src"))
from federated.server import run_federated_simulation
from federated.train_federated import save_experiment_metadata

exp_id = "FL_MLP_NONIID_5_DP_sigma_2.3"

dp_config = {
    "enabled": True,
    "mechanism": "dpsgd",
    "privacy_unit": "record",
    "clipping_norm": 1.0,
    "noise_multiplier": 2.3,
    "delta": 1e-5,
}

metrics, checkpoint_path = run_federated_simulation(
    experiment_id=exp_id,
    model_type="mlp",
    client_count=5,
    partition_method="non_iid",
    alpha=0.5,
    rounds=10,
    local_epochs=3,
    batch_size=128,
    lr=0.005,
    dp_config=dp_config,
    compression_config={"enabled": False},
    seed=42,
)

save_experiment_metadata(
    experiment_id=exp_id,
    model_type="mlp",
    client_count=5,
    partition_method="non_iid",
    alpha=0.5,
    metrics=metrics,
    duration=metrics["simulation_time_sec"],
    seed=42,
    dp_config=dp_config,
    comp_config={"enabled": False},
)

with open(f"results/experiment_metadata/{exp_id}.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

print(f"EXPERIMENT_ID={exp_id}")
print(f"AUROC={meta['metrics']['auroc']}")
print(f"AUPRC={meta['metrics']['auprc']}")
print(f"DP_EPSILON={meta['privacy']['epsilon']}")
print(f"CHECKPOINT={checkpoint_path}")
print("ROUND_FILES=results/{0}_round_metrics.csv, results/{0}_round_metrics.json".format(exp_id))
