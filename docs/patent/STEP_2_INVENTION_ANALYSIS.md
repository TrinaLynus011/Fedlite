# FedLite-Care — Patent Step 2: Invention Identification

This document identifies the technical invention actually supported by the current codebase, without redesign, without new experiments, and without claiming results that are not verified.

Scope: current implementation only. No mechanism change. No experimental fabrication. No legal conclusion.

---

## 1. INVENTION COMPONENT INVENTORY

The tables below classify relevant technical components using the requested categories:

- A. Conventional prior-art component
- B. Implemented technical contribution
- C. Combination/integration feature
- D. Measurement/telemetry feature
- E. Configuration only
- F. Proposed but NOT implemented
- G. Experimentally validated
- H. Not yet experimentally validated

### 1.1 Component inventory table

| Component | Classification | Source file and function/class | Notes |
|---|---|---|---|
| Federated learning workflow | C, G | src/federated/server.py, run_federated_simulation() | Sequential custom FedAvg simulation; round loop, client selection, update aggregation. |
| Client-side training | B, G | src/federated/client.py, FedLiteClient.fit() | Local training occurs on each client before aggregation. |
| DP-SGD mechanism | B, G | src/federated/client.py, FedLiteClient.fit(); src/privacy/dp_mechanism.py, compute_dpsgd_epsilon() | Active mechanism; per-record clipping and local Gaussian noise inside training loop. |
| Per-record gradient clipping | B, G | src/federated/client.py, FedLiteClient.fit(); src/privacy/dp_mechanism.py, clip_grad_tensor() | Each record gets a separate backward pass and clipping before aggregation. |
| Gaussian noise addition | B, G | src/federated/client.py, FedLiteClient.fit(); src/privacy/dp_mechanism.py, clip_and_add_noise() | Noise is added to summed clipped gradients in DP-SGD. |
| Privacy accounting | B, G | src/privacy/dp_mechanism.py, compute_dpsgd_epsilon(); compute_client_update_epsilon() | Formal epsilon accounting implemented in code. |
| RDP accountant | B, G | src/privacy/dp_mechanism.py, _rdp_subsampled_gaussian(); compute_dpsgd_epsilon() | Tight second-order subsampled Gaussian RDP accounting. |
| FedAvg aggregation | A | src/federated/server.py, run_federated_simulation() | Standard FedAvg model averaging logic in a custom simulator. |
| Preprocessing | B, G | data/processed/*.csv; src/data/partition.py, partition_data() | Data is processed and partitioned into client datasets. |
| MLP architecture | A | src/models/mlp.py, LightweightMLP | Standard feed-forward MLP used in the federated system. |
| Communication measurement | D, G | src/federated/server.py, run_federated_simulation(); src/federated/train_federated.py, save_experiment_metadata() | Tracks bytes_upstream and bytes_downstream per round. |
| Model-size measurement | D, G | src/federated/server.py, run_federated_simulation(); src/models/mlp.py, estimate_model_stats() | Parameter count and FP32/INT8 size estimates. |
| Gradient telemetry | D, G | src/federated/server.py, round_metrics_dict; src/federated/client.py, client_metrics | mean_grad_norm, median_grad_norm, frac_clipped, etc. |
| Clipping telemetry | D, G | src/federated/client.py, client_metrics; src/federated/server.py, round_metrics_dict | frac_clipped and clipping_norm are logged. |
| Edge/resource benchmarking | D, G | src/edge/benchmarker.py, run_edge_benchmark() | Measures model load, memory, latency, throughput, CPU-thread simulation. |
| Sparsification/compression | B, G | src/compression/sparsification.py, compress_updates(); src/federated/client.py, FedLiteClient.fit() | Top-K sparsification is applied when a ratio is configured. |
| Privacy configuration | E, G | configs/privacy.yaml | Declares mechanism, unit, clipping norm, noise multiplier, and expected epsilons. |
| Validation/evaluation | B, G | src/utils/metrics.py, calculate_metrics(); src/federated/server.py, get_evaluate_fn() | Validation metrics are computed and threshold tuned on validation only. |
| Experiment metadata | D, G | src/federated/train_federated.py, save_experiment_metadata() | Registry and metadata generation for each run. |
| Checkpointing | D, G | src/federated/server.py, run_federated_simulation() | Saves final model checkpoint under artifacts/checkpoints/federated. |
| Adaptive resource control | F | Not implemented in src/federated/server.py or src/federated/client.py | Telemetry exists, but no closed-loop optimization logic. |
| Real deployment / hospital network operation | F | Not implemented in codebase | Sequential simulation only; no deployment runtime layer. |
| Hardware optimization loop | F | Not implemented | No controller changing privacy/compression/resources at runtime. |

### 1.2 Additional classification notes

- The implementation is strongest in the following areas: a deterministic custom federated simulator, record-level DP-SGD, privacy accounting, lifecycle telemetry, and model/resource measurement.
- The implementation is not strong in the following areas: adaptive control, network deployment, or production healthcare edge orchestration.
- The project supports a system-level integration of privacy, federated updates, and telemetry, but not a closed-loop adaptive runtime.

---

## 2. INVENTION BOUNDARY

### 2.1 What technical system is actually implemented?

The implemented system is a research-grade sequential federated simulation for binary readmission prediction using a lightweight MLP.

Inputs:
- Feature vectors and labels from processed hospital encounter data loaded from data/processed/train_preprocessed.csv, val_preprocessed.csv, and test_preprocessed.csv.
- Federated configuration from configs/federated.yaml and privacy settings from configs/privacy.yaml.
- Client partitioning parameters from src/data/partition.py and the simulation configuration passed to run_federated_simulation().

Processing performed locally:
- Each client receives the global model parameters from the server.
- Each client trains locally using FedLiteClient.fit() in src/federated/client.py.
- For the active DP-SGD path, each record in a local batch is processed individually with a backward pass, clipping, and accumulation of clipped gradients.
- Gaussian noise is added locally to the summed clipped gradients before the optimizer step.
- The client computes its updated parameter delta and optionally applies Top-K sparsification to the outgoing parameter update.

Information that remains local:
- Raw training records remain on the client dataset instance.
- Local per-record gradient norms are computed locally and logged as telemetry but are not transferred as part of the core protocol.
- Local training state and model weights remain on each client during that round.

Information transmitted:
- The server broadcasts model parameters to clients.
- Each client sends updated model parameters or compressed update deltas back to the server.
- The code estimates bytes_upstream and bytes_downstream based on parameter count and dtype size in src/federated/server.py.

Aggregation performed:
- The server aggregates client updates using a custom FedAvg average in run_federated_simulation().
- The update for each layer is weighted by the number of training samples on each client.

Privacy mechanism:
- Active mechanism is record-level DP-SGD.
- The per-example clipping and Gaussian-noise addition are implemented in FedLiteClient.fit() and accounted for in compute_dpsgd_epsilon().

Resource measurements:
- Communication telemetry: upstream/downstream bytes per round.
- Model size estimation: parameter count, FP32 size, INT8 estimate.
- Edge resources: model load time, memory delta, latency, throughput, and thread simulation measured by run_edge_benchmark().

Outputs:
- Validation metrics computed by calculate_metrics() and get_evaluate_fn()
- Round metrics CSV/JSON files in results/
- Metadata at results/experiment_metadata/*.json and registry at results/experiment_registry.csv
- Model checkpoints at artifacts/checkpoints/federated/

Feedback/monitoring mechanisms:
- Logging of gradient norms, clipping rates, and privacy telemetry after each round.
- Validation metrics monitored round by round.
- No runtime control loop changes privacy, compression, or model configuration based on those measurements.

### 2.2 System boundary conclusion

The implemented system is best described as: a custom sequential federated learning simulator with local record-level DP-SGD, local per-record clipping, Gaussian-noise perturbation, FedAvg aggregation, and round-level telemetry for privacy, communication, and resource behavior. It is not a closed-loop edge deployment system and not a hospital-operational platform.

---

## 3. DISTINCTIVE COMBINATION

For patent analysis, the following combinations are potentially distinctive technical combinations requiring prior-art review; they are not legal conclusions.

### 3.1 Combination 1: local record-level DP-SGD + sequential FedAvg + per-round telemetry

- Component A: DP-SGD with per-record clipping in src/federated/client.py
- Component B: custom FedAvg aggregation in src/federated/server.py
- Component C: privacy and gradient telemetry in round_metrics_dict

How they interact technically:
- Each client locally trains with record-level clipping and noise.
- The server aggregates only the resulting model updates.
- The system records per-round mean/median gradient norms, clipping fraction, validation metrics, and bytes.

Technical problem addressed:
- Preserve training utility while limiting the privacy leakage from individual records at the client level, while also tracking communication cost and model behavior.

Support in code:
- FedLiteClient.fit() defines the local DP-SGD loop.
- run_federated_simulation() records round metrics and computes privacy telemetry.

Measured effect:
- Verified pilot run produced AUROC 0.6340, AUPRC 0.1876, communication 1.90 MB, epsilon 11.6717 for sigma=0.8.

Implementation status:
- Implemented and experimentally validated for the sigma=0.8 pilot.

### 3.2 Combination 2: per-record clipping + privacy accountant + telemetry-driven privacy monitoring

- Component A: per-record gradient clipping in FedLiteClient.fit()
- Component B: compute_dpsgd_epsilon() in src/privacy/dp_mechanism.py
- Component C: privacy telemetry in server round metrics

How they interact technically:
- Clipping controls the sensitivity bound per record.
- The accountant converts the training configuration into an epsilon estimate.
- The telemetry exposes the clipping profile and the privacy parameters used.

Technical problem addressed:
- Make privacy accounting and local clipping visible and auditable in a federated training pipeline.

Support in code:
- range of telemetry fields in client_metrics and privacy_telemetry.

Measured effect:
- Verified telemetry confirms clipping statistics and accountant output are saved.

Implementation status:
- Implemented and recorded; not a closed-loop adaptive controller.

### 3.3 Combination 3: static Top-K compression + privacy-preserving local training + communication accounting

- Component A: Top-K sparsification in src/compression/sparsification.py
- Component B: DP-SGD local training in src/federated/client.py
- Component C: communication accounting in src/federated/server.py

How they interact technically:
- The client applies a configured sparsification ratio to parameter deltas before transmission.
- The system records resulting bytes sent and received.
- The mechanism remains static and not dynamically adjusted to resource state.

Technical problem addressed:
- Reduce the reported communication volume while preserving model updates under local privacy protection.

Support in code:
- compress_updates() and the communication block in FedLiteClient.fit().

Measured effect:
- Communication totals are recorded in results/ and reductions are measured in compression results.

Implementation status:
- Implemented as a static configuration mechanism; not adaptive control.

### 3.4 Combination 4: validation-only threshold tuning + federated metric tracking + metadata packaging

- Component A: get_evaluate_fn() in src/federated/server.py
- Component B: calculate_metrics() in src/utils/metrics.py
- Component C: save_experiment_metadata() in src/federated/train_federated.py

How they interact technically:
- The validation set is used to select the best threshold without touching the test set.
- The final metrics are packaged together with privacy and communication metadata.

Technical problem addressed:
- Maintain disciplined metric reporting and avoid leakage from the locked test set.

Support in code:
- The code comments explicitly state that the test set is never used for threshold tuning.

Measured effect:
- Metadata is consistently saved for verified and diagnostic runs.

Implementation status:
- Implemented and documented in code.

### 3.5 Combination 5: edge benchmarking + model statistics + checkpointing

- Component A: run_edge_benchmark() in src/edge/benchmarker.py
- Component B: final_model state save in src/federated/server.py
- Component C: model size metadata generation in src/federated/train_federated.py

How they interact technically:
- The final model is serialized and characterized in size and latency terms.
- Benchmarks estimate startup time, memory use, and throughput.

Technical problem addressed:
- Characterize edge deployability and resource footprint of the saved model.

Support in code:
- run_edge_benchmark() and model_stats entries.

Measured effect:
- File size and latency metrics are generated for model artifacts.

Implementation status:
- Measurement capability exists, but it is not used as a runtime optimization controller.

---

## 4. CLAIM CANDIDATES

These are not final legal claims. They are candidate invention concepts that could potentially inform claim drafting after prior-art review.

### Candidate 1: Federated privacy-preserving clinical model training system

- Technical mechanism: Sequential federated training combined with client-side record-level DP-SGD and server-side FedAvg aggregation.
- Exact implementation evidence: src/federated/client.py, FedLiteClient.fit(); src/federated/server.py, run_federated_simulation(); src/privacy/dp_mechanism.py, compute_dpsgd_epsilon().
- Problem solved: train a globally useful readmission model without centralizing raw hospital data while limiting disclosure from individual patient records.
- Measurable advantage: privacy accounting and gradient clipping are implemented and telemetry is captured.
- Implementation status: implemented and valid for the verified sigma=0.8 pilot.
- Patent risk: moderate; the underlying mechanisms are conventional individually, but the integrated workflow may be distinctive in context.

### Candidate 2: Record-level DP training pipeline with per-example clipping in federated learning

- Technical mechanism: Local backward pass for each record, clipping by L2 norm, accumulation of clipped gradients, Gaussian noise, and formal RDP accountant.
- Exact implementation evidence: src/federated/client.py, per-record loop in FedLiteClient.fit(); src/privacy/dp_mechanism.py, clip_grad_tensor(); compute_dpsgd_epsilon().
- Problem solved: enforce sensitivity control per individual record in a federated DL pipeline.
- Measurable advantage: explicit clipping fraction and local gradient statistics are captured.
- Implementation status: implemented and logged.
- Patent risk: moderate to high; DP-SGD and Gaussian noise are well-known techniques, so the distinctive value depends on the code’s integration and specific workflow framing.

### Candidate 3: Federated communication and privacy telemetry pipeline

- Technical mechanism: Per-round telemetry collection for bytes sent/received, gradient norms, clipping percentages, privacy epsilon, and validation metrics.
- Exact implementation evidence: src/federated/server.py, round_metrics_dict; src/federated/train_federated.py, save_experiment_metadata(); src/edge/benchmarker.py, run_edge_benchmark().
- Problem solved: quantify communication cost, privacy loss, and training behavior in a reproducible simulation.
- Measurable advantage: transparent metrics and metadata generation for each run.
- Implementation status: implemented.
- Patent risk: lower than mechanism-centric claims; measurement and telemetry are often not the strongest patent core alone, but they support a system description strongly.

### Candidate 4: Resource-aware measurement and characterization framework for federated edge models

- Technical mechanism: Model size estimation, startup time, memory delta, throughput, and latency benchmarking.
- Exact implementation evidence: src/edge/benchmarker.py, run_edge_benchmark(); src/models/mlp.py, estimate_model_stats().
- Problem solved: estimate the runtime behavior of a federated learning model on resource-constrained hardware.
- Measurable advantage: benchmarked hardware characteristics are generated for the model artifact.
- Implementation status: implemented as measurement only; not as a runtime optimizer.
- Patent risk: moderate; useful as system context, but not standalone invention if no adaptive control is implemented.

### Candidate 5: Privacy-calibrated local update compression with metrics collection

- Technical mechanism: Top-K sparsification combined with privacy-preserving training and communication telemetry.
- Exact implementation evidence: src/compression/sparsification.py, compress_updates(); src/federated/client.py, communication compression block; src/federated/server.py, round metrics.
- Problem solved: reduce upstream communication volume while tracking privacy and utility shift.
- Measurable advantage: reduction in reported communication volume and logged telemetry.
- Implementation status: implemented as static configuration, not adaptive control.
- Patent risk: moderate; may be a good supporting feature but not the strongest patent core without a dynamic controller.

---

## 5. DO NOT OVERCLAIM

The following MUST NOT currently be claimed as implemented based on the present codebase and verified evidence.

- Real hospital deployment or production clinical deployment.
- Hardware deployment of a live hospital federated network.
- Adaptive resource control or automatic runtime optimization based on measured resources.
- Dynamic compression ratio adjustment as a function of latency, bandwidth, or memory in real time.
- Clinical efficacy or patient-level outcome improvement claims.
- Guaranteed privacy beyond the implemented accountant and the active mechanism.
- Invention of FedAvg.
- Invention of DP-SGD.
- Invention of Gaussian noise.
- Invention of MLP model architecture.
- Completion of the sigma=2.3 experiment.
- A legally supported claim that the system is “patent-ready” solely from author assumptions or undocumented results.
- A claim that the system is a deployed edge device network without evidence in the code or results.

---

## 6. EVIDENCE TABLE

| Proposed feature | Implemented? | Source file | Function/class | Experimentally validated? | Safe to describe? |
|---|---|---|---|---|---|
| Federated learning workflow | Yes | src/federated/server.py | run_federated_simulation() | Yes | Yes, as sequential simulation |
| Local client training | Yes | src/federated/client.py | FedLiteClient.fit() | Yes | Yes |
| DP-SGD | Yes | src/federated/client.py; src/privacy/dp_mechanism.py | FedLiteClient.fit(); compute_dpsgd_epsilon() | Yes | Yes |
| Per-record gradient clipping | Yes | src/federated/client.py; src/privacy/dp_mechanism.py | FedLiteClient.fit(); clip_grad_tensor() | Yes | Yes |
| Gaussian noise | Yes | src/federated/client.py; src/privacy/dp_mechanism.py | FedLiteClient.fit(); clip_and_add_noise() | Yes | Yes |
| Privacy accounting | Yes | src/privacy/dp_mechanism.py | compute_dpsgd_epsilon(); compute_client_update_epsilon() | Yes | Yes |
| RDP accountant | Yes | src/privacy/dp_mechanism.py | _rdp_subsampled_gaussian(); compute_dpsgd_epsilon() | Yes | Yes |
| FedAvg aggregation | Yes | src/federated/server.py | run_federated_simulation() | Yes | Yes |
| Preprocessing | Yes | data/processed/*.csv; src/data/partition.py | partition_data() | Yes | Yes |
| MLP architecture | Yes | src/models/mlp.py | LightweightMLP | Yes | Yes |
| Communication measurement | Yes | src/federated/server.py | round_metrics_dict | Yes | Yes |
| Model-size measurement | Yes | src/models/mlp.py; src/federated/server.py | estimate_model_stats(); run_federated_simulation() | Yes | Yes |
| Gradient telemetry | Yes | src/federated/client.py; src/federated/server.py | client_metrics; round_metrics_dict | Yes | Yes |
| Clipping telemetry | Yes | src/federated/client.py; src/federated/server.py | client_metrics; round_metrics_dict | Yes | Yes |
| Edge/resource benchmarking | Yes | src/edge/benchmarker.py | run_edge_benchmark() | Yes | Yes, measurement only |
| Sparsification/compression | Yes | src/compression/sparsification.py; src/federated/client.py | compress_updates(); FedLiteClient.fit() | Yes | Yes, static configuration |
| Privacy configuration | Yes | configs/privacy.yaml | privacy.budgets | Yes | Yes |
| Validation/evaluation | Yes | src/utils/metrics.py; src/federated/server.py | calculate_metrics(); get_evaluate_fn() | Yes | Yes |
| Experiment metadata | Yes | src/federated/train_federated.py | save_experiment_metadata() | Yes | Yes |
| Checkpointing | Yes | src/federated/server.py | run_federated_simulation() | Yes | Yes |
| Adaptive resource control | No | src/federated/server.py; src/federated/client.py | no controller logic | No | No |
| Real hospital deployment | No | none | no deployment runtime | No | No |
| sigma=2.3 final result | No | not present in verified artifacts | pending | No | No |

---

## 7. PATENT FIGURE REQUIREMENTS

The minimum patent figures should reflect the actual implemented system, not a hypothetical system.

### Figure 1 — Federated simulation architecture
- Title: Federated training system architecture for FedLite-Care
- Blocks/components:
  - Processed patient data
  - Client partitioning
  - Local client training
  - Global aggregation server
  - Validation and threshold tuning
  - Experiment metadata and checkpoint outputs
- Data flow:
  - Data split into clients; global parameters broadcast; local updates computed; updates aggregated; validation metrics produced.
- Technical purpose:
  - Show the implemented research simulation and data flow.

### Figure 2 — Client-side local training and privacy layer
- Title: Client-side DP-SGD update path
- Blocks/components:
  - local batch
  - record-wise backward pass
  - gradient clipping
  - gradient accumulation
  - Gaussian noise addition
  - optimizer step
- Data flow:
  - single record -> per-example gradient -> clipped gradient -> summed gradients -> noise -> update step.
- Technical purpose:
  - Show the actual local privacy mechanism implemented in code.

### Figure 3 — Privacy accounting and telemetry pipeline
- Title: Privacy and communication telemetry pipeline
- Blocks/components:
  - DP configuration
  - local clipping telemetry
  - per-round privacy accountant
  - communication bytes tracking
  - metadata export
- Data flow:
  - training config -> epsilon calculation -> telemetry capture -> metadata save.
- Technical purpose:
  - Show how the system records privacy and communication data.

### Figure 4 — Model and edge resource characterization
- Title: Model sizing and edge benchmarking flow
- Blocks/components:
  - final model checkpoint
  - parameter count and file-size estimation
  - memory and startup benchmark
  - latency and throughput evaluation
- Data flow:
  - checkpoint -> size estimate -> benchmark -> report metrics.
- Technical purpose:
  - Document the measurement-only resource characterization path.

---

## 8. EXPERIMENTAL EVIDENCE

### VERIFIED

The following are supported by the project’s existing verified results and metadata:

- 40/40 tests from the previously verified project state using the project venv: pytest -q passed with 40 passed.
- sigma=0.8 DP-SGD pilot is verified.
- epsilon = 11.6717.
- AUROC = 0.6340.
- AUPRC = 0.1876.
- communication ≈ 1.90 MB.
- training time ≈ 5085 seconds.
- The verified source of these values is results/FINAL_RESULTS_TABLE.csv and the associated round-metric artifacts in results/FL_MLP_IID_5_DP_SGD_sigma_0.8_round_metrics.*.

### PENDING

- sigma=2.3 experiment is pending and must remain unclaimed.
- The config file states sigma=2.3 is expected to be near epsilon=5 under the tight second-order accountant, but the project must not treat this as completed evidence.
- No final sigma=2.3 output artifact was verified as completed in the current codebase state.

---

## 9. FINAL SECTION

### Recommended Patent Core

The strongest technically defensible invention core based only on the current implementation is a custom sequential federated learning system for clinical readmission prediction that combines local record-level DP-SGD, per-example gradient clipping, Gaussian-noise perturbation, server-side FedAvg aggregation, and round-level privacy/communication telemetry within a lightweight, reproducible training pipeline. This is a concrete integrated system implemented in code, and it is stronger and more defensible than any claim based on isolated standard components alone. It is not a claim of adaptive optimization, real deployment, or broad clinical invention.

### What We Need to Add Before Patent Filing

Only genuinely necessary technical/documentary gaps:

1. Confirm and preserve a single, final verified results package for the strongest implemented run (currently the sigma=0.8 pilot).
2. Document the exact invention boundary between implemented system components and non-implemented features.
3. Prepare a clean patent draft that references the actual code architecture and verified telemetry rather than descriptive assumptions.
4. Perform a proper prior-art review focused on the combination of sequential federated learning, record-level DP-SGD, and telemetry-driven communication/privacy measurement.
5. If the sigma=2.3 result is to be included in a filing, it must be completed and verified before it can be described as evidence.

---

## Completion summary

1. STEP 2 completion status: complete as a code-grounded invention analysis document.
2. Strongest candidate invention: local record-level DP-SGD federated learning system with telemetry and privacy accounting.
3. Weakest/riskiest claim area: any claim implying adaptive resource control, deployment, or completed sigma=2.3 results.
4. Exact next step: conduct a focused prior-art review on the strongest combination and narrow the claim theme to the implemented system boundary.
5. Total patent workflow progress: 2/9.

---

This document is intentionally restricted to the current implementation and verified evidence. It does not make any legal conclusion, and it explicitly avoids overclaiming beyond the codebase.
