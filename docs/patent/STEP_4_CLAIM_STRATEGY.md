# FedLite-Care — Patent Step 4: Claim Strategy

This is a technical claim-architecture exercise only. It is not a final patent application, not a legal opinion, and not a statement that the system is legally patentable.

Scope:
- use the actual code, verified evidence, and the prior-art conclusions in Steps 1–3
- do not modify implementation
- do not include unimplemented features as claim bases
- do not claim sigma=2.3 as experimentally completed
- do not claim hardware deployment or adaptive optimization unless the code implements those behaviors

---

## 1. CLAIMABLE INVENTION CORE

### Recommended narrow technical name

Record-Level Federated DP Clinical Update Pipeline with Privacy and Communication Telemetry

### Narrowest technically meaningful invention core

A privacy-aware federated clinical prediction system in which each participating node retains local patient-level data, performs local preprocessing, computes local model updates using record-level DP-SGD with per-example gradient clipping and Gaussian noise injection, and transmits model updates to a central aggregation component that combines them using a FedAvg-style weighted averaging operation while recording privacy, gradient, and communication telemetry for the training run.

### Why this is the strongest current core

This core is actually implemented and supported by the code:
- local client training loop with per-record DP-SGD exists in src/federated/client.py
- aggregation exists in src/federated/server.py
- privacy accounting exists in src/privacy/dp_mechanism.py
- telemetry is captured and exported in round metrics and experiment metadata
- a verified run exists for sigma=0.8 with documented metrics in results/FINAL_RESULTS_TABLE.csv

### What makes the combination technically different

The combination is not the novelty of any single element. It is the specific integration of:
- local retention of patient data at each client node,
- record-level DP-SGD in the local training path,
- per-example clipping and Gaussian perturbation before aggregation,
- weighted averaging at the server,
- and explicit generation of privacy and communication telemetry within the same training workflow.

This is stronger than a generic privacy-preserving FL description because it ties together local record-level privacy, server aggregation, and measured telemetry in a reproducible clinical ML pipeline.

Important limit:
- passive telemetry is not adaptive control.
- no closed-loop optimization is implemented.
- no hardware deployment is implemented.

---

## 2. CLAIM ELEMENT DECOMPOSITION

### Proposed technical elements

E1 — local clinical dataset retained at a participating node
E2 — local preprocessing and partitioning
E3 — lightweight prediction model
E4 — per-record gradient computation at the client
E5 — per-record gradient clipping
E6 — Gaussian perturbation of the aggregated clipped gradients
E7 — local model update generation
E8 — federated aggregation
E9 — privacy accounting
E10 — gradient/clipping telemetry
E11 — communication telemetry
E12 — model/resource measurement and metadata export

For each element:

| Element | Implementation evidence | Source file | Function/class | Prior-art overlap | Importance |
|---|---|---|---|---|---|
| E1 | local data retained on client and partitioned before training | src/data/partition.py; src/federated/server.py | partition_data(); run_federated_simulation() | MEDIUM | Core to federated design |
| E2 | data preprocessing and feature matrix construction | src/federated/server.py; data/processed/*.csv | run_federated_simulation() | HIGH | standard ML workflow |
| E3 | LightweightMLP used for readmission prediction | src/models/mlp.py | LightweightMLP | HIGH | task model but not core novelty |
| E4 | local per-example backward pass for each record | src/federated/client.py | FedLiteClient.fit() | MEDIUM | essential to record-level DP-SGD |
| E5 | clip_grad_tensor and per-record clipping | src/federated/client.py; src/privacy/dp_mechanism.py | FedLiteClient.fit(); clip_grad_tensor() | HIGH | central to DP mechanism |
| E6 | additive Gaussian noise to summed clipped gradients | src/federated/client.py; src/privacy/dp_mechanism.py | FedLiteClient.fit(); clip_and_add_noise() | HIGH | standard DP primitive |
| E7 | model update returned to server after training | src/federated/client.py | FedLiteClient.fit() | MEDIUM | part of local training loop |
| E8 | weighted averaging of client updates | src/federated/server.py | run_federated_simulation() | HIGH | known FedAvg pattern |
| E9 | subsampled Gaussian RDP accountant | src/privacy/dp_mechanism.py | compute_dpsgd_epsilon(); _rdp_subsampled_gaussian() | MEDIUM | formal privacy accounting |
| E10 | gradient norm and clipping telemetry | src/federated/client.py; src/federated/server.py | client_metrics; round_metrics_dict | LOW | supports technical reporting |
| E11 | upstream/downstream byte telemetry | src/federated/server.py | round_metrics_dict | MEDIUM | communication quantification |
| E12 | model size and benchmark measurements | src/edge/benchmarker.py; src/models/mlp.py | run_edge_benchmark(); estimate_model_stats() | MEDIUM | supporting system characterization |

Notes:
- E5 and E6 are the most important privacy elements for the technical story.
- E8 is conventional in FL literature and should not be overemphasized as a stand-alone invention.
- E10–E12 support the measurement layer, but they are not adaptive control features.

---

## 3. INDEPENDENT SYSTEM CLAIM DIRECTION

### System claim concept

A technical system comprising:
- a plurality of participating clinical nodes, each retaining local patient data and configured to locally preprocess feature vectors,
- a local training component at each participating node configured to train a prediction model on the local data,
- a privacy-preserving training component configured to compute per-record gradients for a local data batch, clip each per-record gradient to a bounded L2 norm, aggregate the clipped gradients, and add Gaussian noise to produce a locally updated model state,
- a central aggregation component configured to receive the locally updated model states and combine them into an aggregated model using weighted averaging,
- a privacy-accounting component configured to compute a privacy-loss estimate for the training process based on the local training schedule and the noise configuration,
- a telemetry component configured to measure at least gradient statistics and communication volume associated with the local training and aggregation process.

### Required technical relationship

The key technical relationship is not simply that these components coexist. The essential relationship is:

1. local data remains at the participating node,
2. the local model update is generated through a privacy-preserving record-level DP mechanism before being transmitted,
3. the transmitted update is then combined by the central aggregation component,
4. the system records operational metadata describing the privacy and communication characteristics of that process.

This relationship matters because it defines a federated privacy-preserving training path rather than merely a central DP training system or a generic FL workflow.

The claim architecture should therefore protect the combination of:
- local retention of patient data,
- local record-level DP-SGD path,
- central model aggregation,
- and telemetry-backed measurement of privacy and communication behavior.

---

## 4. INDEPENDENT METHOD CLAIM DIRECTION

### Candidate method architecture

1. receiving, at each of a plurality of participating clinical nodes, a local dataset comprising patient records and associated labels;
2. preprocessing the local dataset to generate feature vectors for a prediction task;
3. initializing a shared model at a coordinator and distributing the shared model parameters to each participating node;
4. at each participating node, computing a local model update by iteratively processing local records, computing a per-record gradient for each record, clipping each per-record gradient according to a bounded L2 norm, summing the clipped gradients, and adding Gaussian noise to the summed gradients;
5. applying the locally noisy gradient update to the local model to generate a local update state;
6. transmitting the local update state to the coordinator;
7. aggregating the received local update states using a weighted averaging operation to generate an updated shared model;
8. computing a privacy-loss estimate based on the local training configuration and the noise and clipping parameters;
9. measuring communication and gradient metrics associated with the training process and storing the results as telemetry.

### Which sequence/order matters

The most important order is:
- local data retention,
- local record-level gradient clipping and Gaussian perturbation,
- transmission of the locally updated state,
- server-side aggregation,
- privacy accounting,
- measurement export.

This order distinguishes the system from generic FL or generic centralized DP training.

---

## 5. DEPENDENT CLAIM DIRECTIONS

Below are candidate dependent claim directions. Each entry states whether it is implemented and validated, or whether it is weaker and should not be emphasized.

| Proposed dependent claim | Status | Notes |
|---|---|---|
| record-level privacy unit | IMPLEMENTED + VERIFIED | explicit in code and config comments |
| per-record gradient clipping with a bounded L2 norm | IMPLEMENTED + VERIFIED | implemented in FedLiteClient.fit() and clip_grad_tensor() |
| Gaussian perturbation applied to summed clipped gradients | IMPLEMENTED + VERIFIED | active DP-SGD path |
| subsampled Gaussian RDP accounting | IMPLEMENTED + VERIFIED | compute_dpsgd_epsilon() and _rdp_subsampled_gaussian() |
| privacy budget estimation based on rounds/local epochs/batch size | IMPLEMENTED + VERIFIED | accountant uses these variables |
| gradient norm telemetry | IMPLEMENTED + VERIFIED | tracked in round metrics |
| clipping fraction telemetry | IMPLEMENTED + VERIFIED | tracked in client metrics |
| communication byte accounting | IMPLEMENTED + VERIFIED | tracked in server round telemetry |
| model size estimation | IMPLEMENTED + VERIFIED | estimate_model_stats() and result metadata |
| lightweight MLP model architecture | IMPLEMENTED + VERIFIED | architecture exists in src/models/mlp.py |
| clinical readmission prediction task | IMPLEMENTED + VERIFIED | task is a defined prediction objective |
| alternative participating node count or client split | CONFIGURATION ONLY | supported by config/partitioning, but not a unique differentiator |
| alternative privacy configuration levels (e.g., sigma grid) | CONFIGURATION ONLY | configuration exists; not a different mechanism |
| adaptive privacy/compression control based on telemetry | NOT IMPLEMENTED | telemetry is passive measurement |
| hardware deployment to edge devices | NOT IMPLEMENTED | no actual device deployment layer |
| real-time resource optimization loop | NOT IMPLEMENTED | no closed-loop controller |
| generic FedAvg alone | CONVENTIONAL / WEAK | known technique |
| generic DP-SGD alone | CONVENTIONAL / WEAK | known technique |
| generic communication measurement alone | CONVENTIONAL / WEAK | measurement feature, not sufficient by itself |

The dependent claims should be chosen for technical specificity and actual implementation support, not simply for count.

---

## 6. THREE CLAIM LAYERS

### Layer 1 — Broad technical concept

The broadest defensible technical concept is:

A federated clinical prediction system that uses local record-level differential privacy during model training and aggregates locally trained model updates at a central coordinator while preserving local data retention.

This protects the broad architecture rather than the exact implementation details.

### Layer 2 — Specific implementation

The specific FedLite-Care implementation is:

A custom sequential federated training pipeline for readmission prediction in which each client performs local DP-SGD with per-record clipping and Gaussian noise injection, and the server aggregates the client updates using a weighted average while recording privacy, gradient, and communication telemetry.

This is the most defensible technical core supported by code and verified results.

### Layer 3 — Fallback embodiments

Fallback embodiments should preserve the same conceptual architecture while altering specific details, such as:
- different client counts,
- different preprocessing feature sets,
- alternative clinical prediction tasks within the same tabular framework,
- alternative privacy parameter settings,
- alternative resource measurement or byte-counting implementations,
- alternative lightweight model architectures within the same training pattern.

These embodiments are useful if prior art is close on the exact model or task, but the technical idea still remains the privacy-preserving federated training pipeline with telemetry.

---

## 7. PRIOR-ART DIFFERENTIATION

| Claim element | Prior-art overlap | Differentiating feature | Importance |
|---|---|---|---|
| local clinical datasets retained at nodes | MEDIUM | local retention plus clinical prediction workflow | important to architecture |
| local preprocessing | HIGH | standard and non-novel | lower claim weight |
| lightweight prediction model | HIGH | conventional MLP-like models | should not dominate claim |
| record-level DP-SGD | MEDIUM | local per-record gradient clipping and noise within FL loop | critical element |
| per-record gradient clipping | HIGH | standard primitive but integrated into local update path | critical but not standalone |
| Gaussian perturbation | HIGH | standard primitive in DP literature | important but conventional |
| federated aggregation | HIGH | standard FedAvg pattern | not a standalone invention |
| privacy accounting | MEDIUM | RDP-based accounting in a training pipeline | important technical support |
| gradient/clipping telemetry | LOW | operational telemetry, not adaptive control | supporting feature |
| communication telemetry | MEDIUM | common FL monitoring data | supporting feature |
| model/resource measurement | MEDIUM | resource characterization | supporting feature |

### Strongest prior-art threat against the independent claim

The strongest threat is the combination of:
- generic federated learning, especially FedAvg, and
- generic DP-SGD / differentially private deep learning, and
- general healthcare FL privacy literature.

This means the claim should not be drafted as a broad statement about FL+DP+healthcare. Instead, it should emphasize the actual system interaction:
- local record-level training and clipping,
- client update generation before aggregation,
- explicit privacy accounting,
- and measured telemetry in the same pipeline.

### Technical element to emphasize

The most defensible emphasized feature is the local record-level DP-SGD pipeline integrated into the federated aggregation workflow, not the broader FedAvg or privacy primitives individually.

---

## 8. WHAT NOT TO CLAIM

The following should currently be avoided or heavily down-weighted in the claim strategy.

- FedAvg alone
  - reason: known, standard FL technique
- DP-SGD alone
  - reason: known technique; claimed alone is too broad and highly exposed
- Gaussian noise alone
  - reason: standard DP primitive
- MLP alone
  - reason: conventional model family
- healthcare prediction alone
  - reason: broad domain concept
- readmission prediction alone
  - reason: task-level claim without the specific training mechanism is too weak
- communication measurement alone
  - reason: measurement is not a novel technical function unless it drives behavior
- gradient measurement alone
  - reason: telemetry is not a claim-driving invention by itself
- generic privacy accounting
  - reason: generally known; should be described as part of the system but not as the main novelty
- hardware deployment
  - reason: not implemented; not supported by code
- adaptive resource optimization
  - reason: no implemented closed-loop controller
- compression if not part of the active demonstrated mechanism
  - reason: the active mechanism is DP-SGD; compression is a static option, not a central claim feature unless necessary
- sigma=2.3 experimental performance
  - reason: pending; not verified as completed

A careful patent strategy should protect the system architecture and the local record-level DP-SGD integration, not a broad or generic statement about known components.

---

## 9. EXPERIMENTAL SUPPORT MATRIX

| Proposed claim element | Experimental evidence | Verified? |
|---|---|---|
| 40/40 tests passed | pytest run in project venv passed | YES |
| local DP-SGD training path | code path implemented in FedLiteClient.fit() | YES |
| record-level clipping | code and telemetry show clipping and grad norms | YES |
| Gaussian perturbation | implemented in DP-SGD training loop | YES |
| FEDERATED aggregation | sequential server averaging implemented | YES |
| privacy accounting | compute_dpsgd_epsilon() executed and metadata saved | YES |
| sigma=0.8 DP-SGD pilot | verified results in FINAL_RESULTS_TABLE.csv | YES |
| epsilon = 11.6717 | verified result in FINAL_RESULTS_TABLE.csv | YES |
| delta = 1e-5 | explicit in config and accountant | YES |
| AUROC = 0.6340 | verified result in FINAL_RESULTS_TABLE.csv | YES |
| AUPRC = 0.1876 | verified result in FINAL_RESULTS_TABLE.csv | YES |
| communication ≈ 1.90 MB | verified result in FINAL_RESULTS_TABLE.csv | YES |
| training time ≈ 5085 s | verified result in FINAL_RESULTS_TABLE.csv | YES |
| privacy telemetry | code emits and stores gradient/clipping metrics | YES |
| communication telemetry | code emits and stores byte counts | YES |
| sigma=2.3 completion | pending run; no verified final result | PENDING |

The strongest evidence is the verified sigma=0.8 pilot, not the sigma=2.3 configuration target. Sigma=2.3 remains explicitly PENDING.

---

## 10. INVENTION STORY

### Problem

A clinically useful predictive model often requires access to multi-site patient data, but hospitals and clinical sites cannot easily centralize raw patient records because of privacy, governance, and data-sharing constraints. At the same time, purely centralized training is not feasible for many healthcare settings, and generic federated learning without privacy constraints leaves patient records exposed to inference risk from model updates and gradient leakage.

### Existing approaches

Existing approaches include generic federated learning frameworks, standard DP-SGD training, and healthcare-focused FL systems. These approaches address individual pieces of the problem but do not, in themselves, provide a fully integrated clinical workflow that combines local record-level DP-SGD, local clipping and noise injection, server-side aggregation, and explicit privacy/communication telemetry in a reproducible, lightweight training pipeline.

### Technical solution

FedLite-Care implements a custom sequential federated learning system for clinical readmission prediction. Each participating node keeps local data on site, preprocesses it, and trains locally using a record-level DP-SGD loop. During each local training step, the per-record gradient is computed and clipped to a bounded L2 norm, then the summed clipped gradients are perturbed with Gaussian noise before the local optimizer step. The resulting model state is transmitted to a central server, which combines client updates via weighted averaging.

### Technical interaction

The technical interaction is the combination of local privacy protection and federated aggregation. The local gradient clipping and Gaussian-noise addition define the privacy boundary for each record before any update is shared externally. The server still performs aggregation over multiple local updates, thereby allowing shared model learning without centralizing the raw data. The system also tracks privacy, gradient, and communication metrics for the run.

### Technical effect

The verified evidence shows that the system trains successfully under a record-level DP-SGD regime, produces documented validation metrics for the sigma=0.8 pilot, and exports telemetry covering privacy and communication behavior. The measurable effects include the privacy accountant output, gradient clipping statistics, communication estimates, and the final recorded model evaluation metrics.

This is a technical system effect, not a claim of a general clinical deployment or a broad novel algorithmic breakthrough.

---

## 11. PATENT FIGURE MAPPING

### FIG. 1 — overall system architecture
- Claim elements: E1, E2, E7, E8, E9, E11
- Purpose: show the end-to-end training workflow
- Essential: yes

### FIG. 2 — local hospital-node processing
- Claim elements: E1, E2, E3, E4, E5, E6, E7
- Purpose: show local clinical data retention and local DP training path
- Essential: yes

### FIG. 3 — DP-SGD record-level training sequence
- Claim elements: E4, E5, E6, E7
- Purpose: show the core privacy mechanism in sequence
- Essential: yes

### FIG. 4 — federated aggregation
- Claim elements: E8
- Purpose: show the weighted model averaging step
- Essential: yes, but should be described as supporting architecture rather than standalone novelty

### FIG. 5 — privacy accounting
- Claim elements: E9
- Purpose: show how privacy-loss estimation is derived from the local training configuration
- Essential: yes, as a technical support element

### FIG. 6 — resource and communication telemetry
- Claim elements: E10, E11, E12
- Purpose: show measurement of gradient behavior, bytes, model size, and resource metrics
- Optional but useful: yes

### Priority of figures

Essential figures:
- FIG. 1
- FIG. 2
- FIG. 3
- FIG. 5

Supporting figures:
- FIG. 4
- FIG. 6

The measurement figures are useful but should not be framed as the core invention if the claim is driven by the privacy-preserving local training architecture.

---

## 12. FINAL RECOMMENDATION

### Recommended independent claim focus

The best independent claim focus is a federated clinical prediction system in which each participating node retains local patient data, applies local record-level DP-SGD with per-record gradient clipping and Gaussian noise to generate a local update, and transmits the update to a coordinating aggregator that combines the updates into a shared model while computing privacy and communication metrics for the training process. This is narrow enough to match the implementation and broad enough to cover the meaningful technical architecture without overclaiming generic FL or generic DP components.

### Recommended dependent claim family

The strongest 8–12 dependent claim directions are:
1. record-level privacy unit
2. per-record gradient clipping to bounded L2 norm
3. aggregate clipped-gradient Gaussian perturbation
4. local optimizer step using noisy gradient
5. weighted server aggregation of client model updates
6. privacy-accounting based on subsampled Gaussian RDP
7. gradient norm telemetry
8. clipping fraction telemetry
9. communication byte accounting
10. model size and resource estimation
11. clinical readmission prediction task
12. lightweight MLP model class or equivalent tabular model

### Strongest fallback position

If the independent claim faces prior-art pressure, the fallback position should shift from a general healthcare FL claim to a narrower system claim focused on the local record-level DP-SGD training sequence and the privacy-accounting + telemetry integration in a federated clinical prediction pipeline. The fallback should keep the architecture but avoid broad statements about generic FL or generic healthcare ML.

### Major patentability risk

The major patentability risk is that the individual ingredients are highly standard: federated learning, FedAvg, DP-SGD, Gaussian noise, MLPs, and healthcare prediction are all well-established in the literature. The real risk is not that the code is unimplemented; it is that the claim must be tailored closely to the specific integration and measurement architecture so that it does not collapse into broad statements about known techniques.

### Evidence gap

The most material evidence gaps before filing would be:
- a clearer, claim-oriented comparison to the exact combination of record-level DP-SGD and federated clinical prediction in prior art,
- a stronger technical distinction between passive telemetry and actual adaptive optimization,
- a final verified result package for the strongest implemented run, if the claim will rely on measured performance beyond the pilot result,
- a formal prior-art search as part of an examiner-facing claim chart.

### NEXT STEP

STEP 5 should produce a claim chart and claim drafting scaffold based on this architecture: it should map each candidate claim element to the code, the verified technical evidence, and the strongest cited prior-art references, then narrow the draft to the strongest defensible invention core without writing the final patent application.

---

PATENT WORKFLOW: 4/9

This document remains a technical claim-architecture exercise and should be reviewed by a patent professional before any legal or filing decision.
