# 1. Current System

The current implementation is a research-grade, deterministic, sequential federated learning simulator rather than a deployed distributed hospital network. The end-to-end control flow is:

1. Load processed training/validation/test splits from data/processed.
2. Partition data into clients using partition_data() in src/data/partition.py.
3. Instantiate FedLiteClient objects in src/federated/server.py: run_federated_simulation().
4. Each client trains locally with the active DP-SGD or diagnostic client-update Gaussian mechanism in src/federated/client.py: FedLiteClient.fit().
5. The server aggregates client updates using a custom sequential FedAvg loop in src/federated/server.py: run_federated_simulation().
6. Validation metrics are calculated on the validation set only; test data is not used in training or threshold selection. This is explicit in src/federated/server.py: get_evaluate_fn() and the comment block in run_federated_simulation().
7. Privacy and communication telemetry are written to results/*_round_metrics.csv and JSON files, and metadata is written by save_experiment_metadata() in src/federated/train_federated.py.
8. Model artifacts are saved in artifacts/checkpoints/federated.

This is the current system architecture. It is not a real networked multi-hospital deployment and the server code itself states that it is a research simulation, not a deployed federated infrastructure (see header comments in src/federated/server.py).

# 2. Implemented Mechanisms

## 2.1 Local training
- Local model architecture: LightweightMLP in src/models/mlp.py (used by FedLiteClient in src/federated/client.py).
- Local logistic-regression baseline: PyTorchLR in src/models/logistic_regression.py.
- Training loop: FedLiteClient.fit() in src/federated/client.py.

## 2.2 Privacy mechanisms implemented
There are two implemented DP mechanisms, and their implementations are intentionally different:

- Active DP-SGD mechanism: src/federated/client.py: FedLiteClient.fit(). This performs per-example gradient clipping before aggregation and adds Gaussian noise to the summed clipped gradients inside the local optimizer step.
- Diagnostic client-update Gaussian mechanism: src/federated/client.py: FedLiteClient.fit() under if dp_enabled and mechanism == "client_update_gaussian" and src/privacy/dp_mechanism.py: clip_and_add_noise(), compute_client_update_epsilon(). This clips and noised the completed model update delta Δθ and is explicitly marked as calibration/diagnostic only.

## 2.3 Communication/compression implemented
- Top-K sparsification is implemented in src/compression/sparsification.py: compress_updates().
- It is invoked in src/federated/client.py: FedLiteClient.fit(), in the communication compression block when compression_config enabled and ratio < 1.0.
- The code computes bytes_before and bytes_after using mask-based accounting; it does not adaptively change compression ratio based on measured network conditions.

## 2.4 Resource telemetry implemented
- Round-level bytes, validation metrics, and training-time telemetry are collected in src/federated/server.py: run_federated_simulation() around the round loop.
- Gradient telemetry (mean_grad_norm, median_grad_norm, frac_clipped, clipping_norm, noise_multiplier) is included in the round metrics dict.
- Model size / parameter count calculations are performed in src/federated/server.py: run_federated_simulation() after final model creation.
- Dedicated edge benchmarking exists in src/edge/benchmarker.py: run_edge_benchmark(), but it is not used by training or the runtime decision logic.

# 3. Privacy Mechanism

The active implementation is DP-SGD with record-level privacy.

Exact implementation details:
- In src/federated/client.py: FedLiteClient.fit(), within the dp_enabled and mechanism == "dpsgd" branch:
  - Each record is processed individually in a local for i in range(B) loop.
  - A per-example backward pass is run for each record.
  - The gradient is clipped with scale = min(1.0, clipping_norm / (rec_norm + 1e-8)).
  - Clipped gradients are accumulated into summed_clipped.
  - Gaussian noise is added with noise_std = noise_mult * clipping_norm before the optimizer step, using torch.normal(...).
  - The optimizer step is performed with p.grad = (acc + noise) / B.

This is consistent with the comments in the header of src/federated/client.py and with the design notes in configs/privacy.yaml.

Important distinction:
- The returned final_parameters are reconstructed as old_parameters + deltas, where deltas are computed after local training and are not additionally noised. This is intentional and explicit in the file comments in src/privacy/dp_mechanism.py and src/federated/client.py.
- No additional client-update Gaussian noise is added after the per-record DP-SGD step in the active mechanism.

Privacy accounting implementation:
- src/privacy/dp_mechanism.py: compute_dpsgd_epsilon()
- Supporting helper: _rdp_subsampled_gaussian()
- Legacy shim: compute_epsilon() is explicitly deprecated and retained only for compatibility.

The accountant uses the tight second-order subsampled Gaussian RDP bound and composes over steps = rounds × local_epochs × ceil(dataset_size / batch_size). The code comments document this exactly.

# 4. Federated Learning Mechanism

The federated learning mechanism is custom sequential FedAvg, not Flower distributed runtime.

Evidence:
- Header in src/federated/server.py states: “This module implements a CUSTOM SEQUENTIAL federated learning simulator. It does NOT use the Flower (flwr) distributed runtime or Ray actors.”
- Aggregation logic in src/federated/server.py: run_federated_simulation() does:
  - collect client_updates
  - compute weight per client as num_samples
  - aggregate layer-wise as sum(layer_updates) / sum(w for _, w in client_updates)
- This is the effective FedAvg update used by the sequential simulation.

This is a simulation of federated learning, not an actual networked distributed deployment; the code explicitly documents that communication bytes are estimated values rather than actual network transport accounting.

# 5. Resource/Communication Mechanism

## 5.1 Communication accounting
Current implementation measures communication cost by approximating parameter count × dtype size:
- In src/federated/server.py: run_federated_simulation(), per-client round_recv += param_count * 4 and round_sent += client_metrics["bytes_sent"].
- The final communication dictionary is written into final_val_metrics["communication"] and persisted in the metadata via src/federated/train_federated.py: save_experiment_metadata().

## 5.2 Compression
Current implementation:
- src/compression/sparsification.py: compress_updates()
- activation point: src/federated/client.py: FedLiteClient.fit() when compression_config["enabled"] and ratio < 1.0

This is a configuration-based communication reduction, not an adaptive controller. It does not choose the compression ratio based on time, bandwidth, or latency measurements.

## 5.3 Resource measurement only (not optimization)
- src/edge/benchmarker.py: run_edge_benchmark() measures file size, startup time, memory, batch latency, throughput, and CPU-thread simulation.
- It is a benchmark script, not a runtime optimization loop.
- The code does not call this benchmark during training or use it to change model size, compression, or privacy settings in real time.

Classification:
- Resource size / latency measurement: A. Measurement only
- Compression ratio configuration: B. Configuration
- Adaptive control based on resource measurements: none present
- Resource optimization decision-making: none present

# 6. Telemetry

Telemetry currently collected in code includes:

- Per-round validation metrics in src/federated/server.py: round_metrics_dict
  - validation_loss
  - val_auroc
  - val_auprc
  - val_f1
  - val_precision
  - val_recall
  - val_specificity
  - val_balanced_accuracy
  - val_brier_score
  - val_accuracy
  - threshold
  - participating_clients
  - total_client_samples
  - bytes_upstream
  - bytes_downstream
  - total_bytes
  - compression_ratio
  - training_time_sec
  - mean_grad_norm
  - median_grad_norm
  - frac_clipped
  - clipping_norm
  - noise_multiplier

- Privacy telemetry in src/federated/server.py: privacy_telemetry dictionary
  - dp_enabled
  - mechanism
  - privacy_unit
  - clipping_norm
  - noise_multiplier
  - delta
  - epsilon
  - rounds_accounted
  - steps_accounted
  - participating_clients
  - total_bytes_transmitted

- Metadata persistence in src/federated/train_federated.py: save_experiment_metadata()
  - experiment_id
  - timestamp
  - privacy metadata
  - communication volume
  - model stats
  - metrics

- Edge benchmark metrics in src/edge/benchmarker.py: run_edge_benchmark()
  - model_file_size_kb
  - startup_time_sec
  - peak_ram_diff_mb
  - total_ram_usage_mb
  - batch_latency_ms
  - single_sample_latency_ms
  - throughput_fps
  - threads_simulated

Important limitation:
- Telemetry is recorded and saved, but there is no code path that uses those measurements to change model configuration, privacy parameters, communication strategy, or client participation mid-run.

# 7. Adaptive Control

No adaptive resource-aware controller currently exists.

Evidence:
- No code in src/ or configs/ contains logic such as: if bandwidth is high then reduce noise; if memory > X then compress more; if latency > Y then change model capacity; if epsilon exceeds budget then retrain.
- The server loop in src/federated/server.py is a fixed deterministic sequence of rounds.
- The client-side compression logic in src/federated/client.py is triggered only by a fixed static ratio in compression_config.
- The edge benchmarker in src/edge/benchmarker.py runs as a standalone script and does not feed back into training.

Therefore, the implementation is:
- measurement capable
- configuration-driven
- not adaptive control
- not optimization in the patent sense of closed-loop decision-making

# 8. Evidence

## 8.1 Verified code evidence
- DP-SGD active implementation: src/federated/client.py: FedLiteClient.fit()
- DP-SGD accountant: src/privacy/dp_mechanism.py: compute_dpsgd_epsilon()
- client-update Gaussian diagnostic mechanism: src/privacy/dp_mechanism.py: clip_and_add_noise(), compute_client_update_epsilon()
- FedAvg aggregation: src/federated/server.py: run_federated_simulation()
- compression implementation: src/compression/sparsification.py: compress_updates()
- telemetry collection: src/federated/server.py: run_federated_simulation(); src/federated/train_federated.py: save_experiment_metadata()
- benchmark measurement script: src/edge/benchmarker.py: run_edge_benchmark()

## 8.2 Verified project evidence files
- tests/test_fedlite_care.py passed: 40 passed in the project environment
- results/FL_MLP_IID_5_DP_SGD_sigma_0.8_round_metrics.csv
- results/FL_MLP_IID_5_DP_SGD_sigma_0.8_round_metrics.json
- results/dp_calibration_summary.csv
- results/dp_calibration_results.json
- results/experiment_metadata/FL_MLP_NONIID_5_DP_eps_5.0.json (diagnostic client-update Gaussian metadata)

These files support the existence of the mechanisms and the verified sigma=0.8 pilot result; they do not support a claim that a sigma=2.3 final result was completed.

# 9. Documentation-vs-Code Gaps

## 9.1 Documentation that is stronger than code
The following are described in documentation or higher-level project descriptions but are not implemented as active runtime control logic:

- “resource-aware optimization”: not implemented as adaptive decision-making; no runtime controller exists.
- “edge/resource optimization”: measured in benchmarker only; not used in runtime control.
- “distributed deployment”: the server comments explicitly say this is a sequential simulation, not a real network deployment.
- “real hospital deployment”: not implemented.
- “clinical deployment” or “HIPAA compliance”: not implemented as a deployment claim; the code is a research artifact.

## 9.2 Code that is documented but not used adaptively
- The compression mechanism is real, but it is static configuration only.
- The benchmarker is real, but it is not connected to any control policy.
- Privacy telemetry is real, but it is not fed into a controller that changes training decisions.

# 10. Potentially Distinctive Technical Combinations

The code supports a real technical combination, but it is narrower than a broad “resource-aware adaptive federated healthcare platform” claim.

Potentially distinctive combination in code:
- local client-side per-record DP-SGD
- FedAvg aggregation in a custom sequential simulation
- communication accounting and compression toggle
- round-level privacy and utility telemetry
- model-size and benchmark measurements

This is technically meaningful, but the currently implemented combination is not yet a closed-loop multi-objective controller or a real deployed edge optimization system.

# 11. Patent Risks

## 11.1 Over-claim risk
The main patent risk is claiming more than the code implements. Examples:
- claiming an adaptive resource-aware controller when the code only measures resources
- claiming a real edge deployment when the code is sequential simulation only
- claiming communication optimization when compression is static and not adaptive
- claiming the architecture is a deployed healthcare production system when it is a research simulator
- claiming patient-level DP when the code claims record-level DP and the project explicitly notes this distinction

## 11.2 Mechanism-level risk
- FedAvg is conventional.
- DP-SGD is conventional.
- Gaussian noise and clipping are conventional.
- MLPs are conventional.
- The novelty, if any, must be tied to the specific integrated workflow, telemetry, and resource measurement scheme rather than to individual mechanisms.

## 11.3 Best patent framing
A stronger but still supportable framing is: a simulated or prototype federated clinical learning system that integrates local per-record DP-SGD with FedAvg aggregation and resource/communication telemetry in a lightweight, reproducible workflow. This is far stronger than claiming an autonomously adaptive optimization controller without implementation evidence.

# 12. Recommended Next Technical Step

The most defensible next technical step is not a redesign. It is a narrow evidence-building step:

1. Confirm whether the sigma=2.3 run is truly pending compute and preserve the verified sigma=0.8 evidence.
2. Keep the DP-SGD mechanism and accountant unchanged.
3. Add a single explicit, code-backed artifact documenting the exact boundary between:
   - measurement-only telemetry,
   - configuration-based compression,
   - and adaptive control (currently absent).
4. Prepare the patent draft around the integrated workflow and telemetry design, rather than around claimed resource-optimization behavior that is not implemented.

This preserves scientific integrity and avoids stronger patent claims than the implementation can support.

---

Feature | Implemented? | Evidence | Patent relevance
---|---|---|---|
DP-SGD with per-record clipping | Yes | src/federated/client.py: FedLiteClient.fit(); src/privacy/dp_mechanism.py: compute_dpsgd_epsilon() | High, but conventional mechanism; must be framed as integrated workflow rather than standalone novelty |
Client-update Gaussian diagnostic mechanism | Yes | src/federated/client.py: FedLiteClient.fit(); src/privacy/dp_mechanism.py: clip_and_add_noise() | Low novelty; explicitly diagnostic only |
FedAvg aggregation | Yes | src/federated/server.py: run_federated_simulation() | Conventional, but part of integrated workflow |
Compression configuration | Yes | src/compression/sparsification.py: compress_updates(); src/federated/client.py: fit() | Medium; static configuration only, not adaptive |
Resource/communication measurement | Yes | src/federated/server.py: round_metrics_dict; src/edge/benchmarker.py: run_edge_benchmark() | Medium; measurement-only evidence |
Adaptive resource control | No | No controller logic in src/federated/server.py or src/federated/client.py | Low patent support unless implemented |
Runtime optimization from telemetry | No | telemetry is recorded but not used to alter training decisions | Not currently supportable as an adaptive optimization claim |
Real deployment / distributed network | No | header comments in src/federated/server.py explicitly note sequential simulation only | Patent claim should avoid describing production deployment |
