# FedLite-Care — Patent Step 5: Patent Specification + Claim Draft

This document is a technical patent drafting scaffold only. It is not legal advice, not a final patent application, and not a legal opinion regarding patentability. All statements are limited to the implemented code, the verified project evidence, and the technical audit records in the preceding patent steps.

The following technical boundaries are maintained throughout:
- no source-code changes
- no unnecessary experiments
- no fabrication of evidence
- no claim that sigma=2.3 is experimentally complete
- no claim of hospital deployment
- no claim of adaptive optimization unless implemented in the code
- no claim that FedAvg, DP-SGD, Gaussian noise, or MLPs are inventions by themselves
- no inventors, applicants, assignees, dates, or legal status are specified

---

# PART A — PATENT SPECIFICATION

## 1. TITLE

FedLite-Care: Privacy-Preserving Federated Learning for Clinical Prediction with Record-Level DP-SGD and Telemetry

This title is the strongest technically accurate title supported by the current implementation and the prior claim strategy analysis.

## 2. TECHNICAL FIELD

The present disclosure relates generally to machine learning systems for privacy-aware clinical prediction, federated learning, differential privacy, and measurement of communication and resource characteristics during training. More particularly, the present disclosure relates to systems and methods for training a clinical prediction model across multiple local datasets while retaining local data at participating nodes, applying record-level differential privacy during local training, aggregating local updates at a central coordinator, and tracking privacy and communication telemetry for the training workflow.

The disclosed system is directed to the technical field of distributed learning and privacy-preserving model training for healthcare-related prediction tasks, including readmission-risk prediction, without requiring raw patient data to be centrally aggregated.

## 3. BACKGROUND

### 3.1 Fragmented clinical data

Clinical data is often fragmented across multiple institutions, departments, and healthcare data silos. Each participating institution may maintain its own local patient records and associated clinical features. A centralized approach to model training can require pooling sensitive patient-level data into a single location, which introduces technical and governance challenges.

### 3.2 Constraints on centralized aggregation

Centralizing patient data may be impractical because of data-sharing restrictions, local governance policies, variability in local data quality, and the cost of transferring large datasets. In clinical settings, institutions may prefer to keep records locally while still contributing to a shared predictive model.

### 3.3 Computational limitations at participating healthcare nodes

Healthcare nodes may have limited memory, limited local compute capacity, and varied networking conditions when participating in a collaborative model-training process. A system that trains locally while preserving a lightweight model footprint and tracking communication load can be useful in settings where a full-scale distributed infrastructure is not available.

### 3.4 Privacy requirements

Because patient records contain sensitive information, local training procedures may be expected to provide formal privacy protections for individual records. Simple model sharing without a privacy mechanism can leak information through parameter updates or gradient information. A privacy-preserving local update mechanism can reduce the risk that a local training record is revealed or inferred from the transmitted update.

### 3.5 Limitations of conventional federated learning

Federated learning allows a model to be trained across local datasets without transmitting raw data. However, conventional federated learning workflows can still expose sensitive information through model updates and can entail substantial communication overhead. In addition, naive federated workflows may not provide detailed accounting of the effective privacy loss, the gradient sensitivity bound, or the communication cost incurred during the training process.

### 3.6 Need for measurable privacy/resource characteristics

In research and system design, it is often useful to quantify training behavior in terms of privacy loss, gradient statistics, clipping behavior, communication volume, and model size. A technical workflow that records these quantities as telemetry can support reproducibility and system-level understanding, even when the telemetry remains passive and does not change the training configuration mid-run.

This disclosure addresses the technical problem of enabling a privacy-aware federated training pipeline for clinical data while retaining local patient records, tracking privacy and communication characteristics, and avoiding central data pooling in a research-grade simulation environment.

## 4. PROBLEM TO BE SOLVED

A technical problem addressed by the present disclosure is how to train a shared clinical prediction model across multiple local datasets while preserving local data retention, reducing exposure of individual record information in model updates, and providing technical accounting of privacy and communication characteristics. Conventional approaches may rely on either centralized training, standard federated averaging without privacy protection, or generic differential privacy mechanisms that are not integrated into a local clinical learning workflow with measurable telemetry.

Accordingly, there is a need for a system that:
- keeps local patient records at their respective participating nodes,
- computes local model updates using a record-level privacy mechanism,
- transmits only the resulting update state rather than raw patient records,
- aggregates the updates into a shared model,
- and records the privacy and communication characteristics of the training process in a reproducible manner.

The solution is not a deployment claim and does not require a public healthcare production system; the implemented system is a research-grade, sequential federated simulation configured to demonstrate the technical architecture and its recorded privacy/communication behavior.

## 5. SUMMARY OF THE INVENTION

The present disclosure provides a federated learning architecture and method for privacy-aware clinical prediction. In a disclosed implementation, each participating clinical node retains local patient records and performs local preprocessing to generate training features and labels. A shared prediction model is initialized and distributed to the participating nodes. At each node, a local training phase is performed using a record-level differentially private stochastic gradient descent procedure.

In the active implementation, each individual record in a local batch is processed separately via a local backward pass. The gradient corresponding to each record is computed, its L2 norm is evaluated, and the gradient is clipped to a bounded norm according to a clipping threshold. A clipped gradient accumulator is maintained for the batch. After processing all records in the batch, a Gaussian noise term is added to the accumulated clipped gradients before the local optimizer step. The local training procedure thus produces a model update that is privacy-protected before it is transmitted to a central coordinator.

The central coordinator aggregates the received local updates using a weighted averaging procedure, which is a FedAvg-style aggregation in the implemented custom sequential simulator. The resulting shared model is then evaluated on a validation set, and the system records privacy, gradient, and communication telemetry associated with the training process. The telemetry includes gradient norm statistics, clipping fraction, privacy parameters, byte counts, and model metadata.

The invention is not based on the novelty of standard FedAvg alone, DP-SGD alone, Gaussian noise alone, or an MLP alone. The specific combination and interoperation of local record-level DP-SGD, server-side aggregation, and privacy/communication telemetry forms the meaningful disclosed architecture.

## 6. BRIEF DESCRIPTION OF DRAWINGS

FIG. 1 — Overall FedLite-Care architecture:
A high-level block diagram showing participating clinical nodes, local data retention, a shared model coordinator, local DP-SGD training, aggregation, validation, and telemetry output.

FIG. 2 — Participating clinical node:
A schematic of a node containing local clinical data, preprocessing, a model instance, local training loop, clipping/noise procedure, and a communication pathway to the coordinator.

FIG. 3 — Record-level DP-SGD processing:
A flow diagram showing per-record forward/backward computation, L2 norm calculation, gradient clipping, clipped-gradient accumulation, Gaussian-noise addition, and optimizer update.

FIG. 4 — Federated aggregation workflow:
A diagram showing the weight computation for client updates, the server-side FedAvg aggregation, and the generation of an updated global model.

FIG. 5 — Privacy accounting:
A schematic of the privacy accountant using the local training configuration, sampling rate, number of steps, noise multiplier, and delta to compute an epsilon estimate from the subsampled Gaussian RDP account.

FIG. 6 — Resource and communication telemetry:
A diagram showing per-round measurement of gradient norm statistics, clipping fraction, upstream/downstream bytes, model size, validation metrics, and training time.

## 7. DETAILED DESCRIPTION

### 7.1 Participating clinical nodes

The disclosed system includes a plurality of participating clinical nodes. Each node maintains a local dataset containing patient records and corresponding labels or outcomes. The raw records are retained locally at the node rather than being transmitted to a central repository. At each node, a local model is instantiated and trained using the node's local dataset.

The implementation uses a custom sequential federated simulator rather than a distributed runtime. In the implemented system, client objects are instantiated in a server-side simulation loop and trained serially, with each client receiving the current global model parameters and returning its local update. This research-grade simulation preserves the direct architectural concept of local computing and centralized aggregation without implementing a full production distributed network stack.

### 7.2 Local preprocessing

At each participating node, the local dataset is preprocessed and partitioned prior to training. In the current implementation, processed training and validation data are loaded from the project data/processed directory. Feature columns are isolated from the target column and used to form the input feature matrix. The training data is then partitioned into client-specific subsets using a deterministic partitioning procedure.

The implementation includes the use of a processed clinical tabular feature set for readmission prediction. The preprocessing pipeline is not treated as the primary inventive feature; rather, it provides the local feature vectors that are used by the local model and the privacy-preserving training loop.

### 7.3 Lightweight prediction model

The current implementation uses a lightweight multi-layer perceptron model, referred to in the code as LightweightMLP, for the clinical prediction task. In the disclosed system, a model of this type is used to map the local feature vectors to a binary clinical outcome. The lightweight model is used to keep the training process manageable while working with structured clinical tabular data.

The MLP is not asserted as novel by itself. It is described as a model architecture implemented within the privacy-preserving federated training workflow.

### 7.4 Record-level DP-SGD

The active privacy mechanism in the disclosed implementation is a record-level DP-SGD mechanism. The implemented logic is as follows.

1. Individual record forward/backward computation.
   During training on a local batch of size B, each record is processed individually. For each record i in the local batch, the model computes an output for the record, and the per-record loss is computed. A backward pass is executed so that a per-record gradient is produced.

2. Gradient norm calculation.
   For each processed record, the L2 norm of the gradient is determined as the square root of the sum of squared gradient entries across the model parameters. This is represented in the code as a per-example gradient norm computed from the local parameter gradients.

3. L2 clipping.
   The per-record gradient is clipped according to a clipping threshold C, using the standard scaling factor min(1, C / ||g_i||_2). The clipping is applied to bound the sensitivity of each record's contribution.

4. Clipped gradient accumulation.
   The clipped gradients from all records in the current batch are accumulated into a batch-level accumulator. The accumulation reflects the total clipped contribution of the records in the local batch.

5. Gaussian noise generation.
   After the clipped contributions have been accumulated, a Gaussian noise term is generated with standard deviation equal to noise_multiplier × clipping_norm. The implementation uses a Gaussian random variable with mean 0 and standard deviation noise_std = sigma × C.

6. Division by batch size.
   The accrued clipped gradients plus the Gaussian noise term are divided by the local batch size B before the optimizer step. In the code, the noisy gradient is set as p.grad = (acc + noise) / B.

7. Optimizer update.
   The optimizer step is then executed using the noisy, normalized gradient. This local update represents the privacy-protected update generated at the node before the model state is returned to the coordinator.

The implementation uses the per-record privacy unit of an individual record, described in the code comments as a patient encounter. The active mechanism is described as record-level differential privacy and is distinct from the diagnostic client-update Gaussian mechanism, which is specifically marked as calibration/diagnostic only.

### 7.5 Federated aggregation

The disclosed system includes a central coordinator that aggregates client updates into a global model. In the implemented custom sequential federated learning simulator, each client trains locally and returns a model update or parameter delta. The server then computes a weighted average of client updates using sample weights corresponding to the number of local training samples. The local weights are used to combine each layer's update contributions and produce an updated global model.

The implementation is consistent with a custom sequential FedAvg-style aggregation process. Each round of communication includes local training at the clients, a weighted server aggregation step, and evaluation of the global model on a validation set. This is not a claim that a real networked production deployment has been implemented; the server code explicitly distinguishes the sequential simulator from a distributed production runtime.

### 7.6 Privacy accounting

The disclosed system records a privacy-loss estimate for the training process based on the active DP-SGD configuration. The implemented privacy accounting is based on the subsampled Gaussian mechanism under a Rényi-DP formulation.

The privacy account uses the following technical parameters:
- privacy unit: individual record
- sampling rate q = batch_size / client_dataset_size
- training steps T = rounds × local_epochs × ceil(dataset_size / batch_size)
- delta: target failure probability, implemented as 1.0e-5
- noise multiplier sigma: the configured noise multiplier parameter
- clipping norm C: L2 clipping threshold for per-example gradient clipping

The accountant computes a privacy-loss estimate by evaluating a family of Rényi-DP orders, with the RDP bound computed from the subsampled Gaussian mechanism. The effective epsilon is then determined by selecting the minimum epsilon over the evaluated alpha values after conversion from RDP to an (epsilon, delta)-DP estimate.

The implementation explicitly states that the epsilon value is reported from the accountant after training and is not a pre-specified privacy cap. The current code supports a tight second-order subsampled Gaussian RDP estimate and does not claim stronger guarantees than those supported by the implemented accountant.

### 7.7 Telemetry

The disclosed system collects and stores telemetry describing the training run. The implementation records the following categories of information:

- gradient statistics:
  - mean gradient norm
  - median gradient norm
  - fraction of gradients clipped
  - clipping norm
  - noise multiplier
- validation metrics:
  - validation loss
  - AUROC
  - AUPRC
  - F1
  - precision
  - recall
  - specificity
  - balanced accuracy
  - Brier score
  - accuracy
  - threshold
- communication and resource metrics:
  - upstream bytes
  - downstream bytes
  - total bytes transmitted
  - model parameter count
  - theoretical FP32 model size
  - theoretical INT8 model size
  - serialized checkpoint size
  - training duration in seconds

The telemetry is recorded passively. It is included in the results and metadata files of the simulation; it is not used by the implementation to adaptively change the training configuration, privacy parameters, or communication strategy mid-run. Accordingly, the telemetry is disclosed as measurement and accounting information, not as a closed-loop optimization controller.

### 7.8 Resource awareness

The system includes resource-aware measurement features, including communication byte accounting and model-size estimation. The server records estimated upstream and downstream communication based on parameter count and dtype size, and the training run writes communication telemetry to results files.

The implementation also includes a benchmark script that measures model load time, memory footprint, throughput, and thread simulation. This benchmark is implemented as measurement infrastructure rather than runtime control logic. It does not automatically optimize privacy parameters, compression rate, or model configuration based on measured resource conditions.

Therefore, the system is accurately characterized as measurement-capable and configuration-driven, not adaptive-control-based.

## 8. ALTERNATIVE EMBODIMENTS

The present disclosure contemplates technically reasonable variants that are consistent with the architecture and implementation disclosed herein.

### 8.1 IMPLEMENTED EMBODIMENT

The currently implemented embodiment is a research-grade, sequential federated learning simulation for binary readmission prediction using a lightweight MLP. In this embodiment:
- multiple local clients are created from partitioned data,
- each client retains local clinical records,
- each client performs local record-level DP-SGD,
- the server performs weighted aggregation,
- privacy and communication telemetry are written to results files,
- and the final model is checkpointed and evaluated on a validation set.

### 8.2 DISCLOSED ALTERNATIVE EMBODIMENTS

Alternative embodiments include, without limitation:
- different numbers of participating nodes,
- different clinical prediction tasks within the same structured-tabular framework,
- different local model architectures of similar capacity,
- different batch sizes,
- different local epochs,
- different clipping norms,
- different noise multipliers,
- different partitioning schemes,
- different aggregation schedules,
- and different telemetry reporting formats.

These alternative embodiments are consistent with the disclosed architecture and do not imply that the alternatives were experimentally validated in the current codebase. They are disclosed as technically reasonable variations of the architecture, not as claimed embodiments experimentally completed by this project.

## 9. EXPERIMENTAL EXAMPLE

The following information is the only experimentally verified evidence included in this draft and is limited to the project evidence already accumulated: 

- 40/40 tests passed in the project environment.
- A verified sigma=0.8 DP-SGD pilot was executed and recorded.
- The recorded privacy estimate for the sigma=0.8 pilot was epsilon = 11.6717.
- The target delta was 1e-5.
- The recorded validation AUROC was 0.6340.
- The recorded validation AUPRC was 0.1876.
- Communication cost was approximately 1.90 MB.
- Training time was approximately 5085 seconds.
- Approximately 11–13% of gradients were clipped during the verified pilot.

The sigma=2.3 configuration remains pending and is not an experimentally verified evidentiary result. It should not be described as completed within this patent draft unless and until a verified final artifact exists.

## 10. TECHNICAL ADVANTAGES

The disclosed system provides the following technical advantages that are supported by the architecture and recorded experimental evidence:

- local data retention at nodes reduces the need to centralize raw patient records,
- local record-level DP-SGD introduces privacy protection at the point of local update generation,
- weighted server aggregation enables a shared model without raw-data transfer,
- privacy accounting makes the privacy-loss estimate explicit and auditable in the training run,
- communication telemetry provides transparency regarding upstream and downstream communication cost,
- model size and benchmark metrics provide a reproducible technical characterization of the run,
- and the workflow is reproducible in a structured research-grade simulation.

These advantages do not claim guaranteed clinical benefit or hospital deployment success.

## 11. LIMITATIONS

The present disclosure is limited by the following facts:

- the implementation is a research-grade simulation and not a demonstrated hospital network deployment,
- the system is not a real-world operating environment with actual physical clinical infrastructure,
- the project does not demonstrate a closed-loop adaptive controller,
- sigma=2.3 remains pending and is not treated as experimental evidence,
- no clinical validation or patient-impact outcome is claimed,
- and the present disclosure is not a legal determination of patentability.

The disclosed architecture is therefore a technically grounded research workflow and not a production clinical deployment claim.

---

# PART B — CLAIM DRAFT

## CLAIM 1 — INDEPENDENT SYSTEM CLAIM

1. A privacy-aware federated learning system for clinical prediction, comprising:
   a plurality of participating clinical nodes, each of the participating clinical nodes retaining a local clinical dataset including patient-specific records and labels;
   a preprocessing module at each participating clinical node configured to generate local training features from the local clinical dataset;
   a local model training component at each participating clinical node configured to perform training on the local training features using a shared prediction model;
   a record-level privacy-preserving training module at each participating clinical node configured to:
   compute, for each record in a local batch, a corresponding local gradient,
   determine a gradient norm for the local gradient,
   clip the local gradient to a bounded L2 norm based on a clipping threshold,
   accumulate clipped gradients for the local batch,
   generate a Gaussian noise term, and
   apply the Gaussian noise term to the accumulated clipped gradients to form a locally privacy-protected gradient update;
   a local optimizer configured to update the shared prediction model using the locally privacy-protected gradient update;
   a coordinator configured to receive one or more locally privacy-protected gradient updates from the plurality of participating clinical nodes and to aggregate the one or more locally privacy-protected gradient updates to generate an updated shared model using a weighted aggregation operation; and
   a telemetry module configured to measure at least one of gradient statistics, clipping behavior, communication volume, or privacy accounting metadata associated with the local training and aggregation process.

2. The system of claim 1, wherein the coordinator is configured to aggregate the locally privacy-protected gradient updates using a weighted averaging operation based on a number of local training samples associated with each participating clinical node.

3. The system of claim 1, wherein the telemetry module is configured to record at least one of a mean gradient norm, a median gradient norm, a clipping fraction, a communication byte count, a validation metric, or a privacy-loss estimate.

4. The system of claim 1, wherein the record-level privacy-preserving training module is configured to compute the privacy-loss estimate using a subsampled Gaussian Rényi differential privacy accountant based on a noise multiplier, a clipping norm, a batch size, and a number of training steps.

## CLAIM 2 — INDEPENDENT METHOD CLAIM

1. A computer-implemented method for privacy-aware federated learning for clinical prediction, the method comprising:
   storing, at each of a plurality of participating clinical nodes, a local clinical dataset including patient-specific records and labels;
   preprocessing, at each participating clinical node, the local clinical dataset to generate local training features;
   initializing a shared prediction model at a coordinator and distributing the shared prediction model to the plurality of participating clinical nodes;
   at each participating clinical node, for each record in a local batch, computing a local gradient using the local training features and the shared prediction model;
   at each participating clinical node, determining an L2 norm of the local gradient and clipping the local gradient to a bounded L2 norm according to a clipping threshold;
   at each participating clinical node, accumulating clipped gradients for the local batch and adding a Gaussian noise term to the accumulated clipped gradients;
   at each participating clinical node, updating the shared prediction model using the accumulated clipped gradients and the Gaussian noise term;
   transmitting a locally updated model state from each participating clinical node to the coordinator;
   aggregating, at the coordinator, the locally updated model states using a weighted averaging operation to generate an updated shared model; and
   recording at least one privacy metric and at least one communication metric associated with the local training and aggregation process.

2. The method of claim 1, wherein the local training comprises a record-level DP-SGD process and the privacy metric is computed using a Rényi differential privacy accountant based on a subsampled Gaussian mechanism.

3. The method of claim 1, wherein the communication metric comprises an estimate of upstream bytes, downstream bytes, or total bytes transmitted during the local training and aggregation process.

## DEPENDENT CLAIMS

### Claim 3 — Record-level privacy unit

1. The system of claim 1 or the method of claim 1, wherein the privacy-preserving training module is configured to treat an individual record as a privacy unit.

### Claim 4 — Per-record gradient clipping

2. The system of claim 1 or the method of claim 1, wherein the local gradient is clipped according to a scaling factor of min(1, C / ||g_i||_2), where C is a clipping threshold and g_i is a local gradient corresponding to an individual record.

### Claim 5 — Clipping threshold

3. The system of claim 1 or the method of claim 1, wherein the clipping threshold is approximately 1.0.

### Claim 6 — Gaussian noise

4. The system of claim 1 or the method of claim 1, wherein the Gaussian noise term has a standard deviation defined by a noise multiplier multiplied by the clipping threshold.

### Claim 7 — Privacy accounting and RDP

5. The system of claim 1 or the method of claim 1, wherein the privacy-loss estimate is computed using a subsampled Gaussian Rényi differential privacy accountant over a plurality of training steps and a target delta value.

### Claim 8 — Gradient telemetry

6. The system of claim 1 or the method of claim 1, wherein the telemetry module is configured to record at least one of a mean gradient norm or a median gradient norm of the local gradients.

### Claim 9 — Clipping telemetry

7. The system of claim 1 or the method of claim 1, wherein the telemetry module is configured to record a fraction of local gradients that are clipped relative to the clipping threshold.

### Claim 10 — Communication telemetry

8. The system of claim 1 or the method of claim 1, wherein the telemetry module is configured to record an estimated amount of communication volume between the participating clinical nodes and the coordinator.

### Claim 11 — Model and resource measurement

9. The system of claim 1 or the method of claim 1, wherein the telemetry module is further configured to record a model parameter count, a model size estimate, or a training duration metric.

### Claim 12 — Lightweight model implementation

10. The system of claim 1 or the method of claim 1, wherein the shared prediction model comprises a lightweight multi-layer perceptron model configured for clinical tabular prediction.

### Claim 13 — Validation metrics

11. The system of claim 1 or the method of claim 1, wherein the telemetry module is configured to record at least one validation metric selected from AUROC, AUPRC, F1, recall, precision, specificity, balanced accuracy, Brier score, or accuracy.

### Claim 14 — Simulation environment and deterministic implementation

12. The system of claim 1 or the method of claim 1, wherein the plurality of participating clinical nodes are implemented in a sequential simulator configured to model federated learning without requiring a production distributed deployment runtime.

This claim is supported technically but should be treated cautiously because it describes the current research architecture rather than a general product deployment.

## CLAIM SUPPORT MATRIX

| Claim limitation | Implementation source | Experimental support | Prior-art risk |
|---|---|---|---|
| participating clinical nodes retain local data | src/data/partition.py; src/federated/server.py | yes, architecture and partitioning implemented | moderate |
| local preprocessing | data/processed files; src/federated/server.py | yes, implemented | moderate |
| local training on client | src/federated/client.py | yes | moderate |
| record-level privacy-protecting training | src/federated/client.py; src/privacy/dp_mechanism.py | yes, active mechanism | moderate |
| per-record gradient norm and clipping | src/federated/client.py; src/privacy/dp_mechanism.py | yes, telemetry and code logic | moderate-high |
| Gaussian noise addition | src/federated/client.py | yes | moderate-high |
| optimizer update using noisy gradient | src/federated/client.py | yes | moderate |
| weighted aggregation at coordinator | src/federated/server.py | yes | high |
| privacy accounting via RDP | src/privacy/dp_mechanism.py | yes | moderate |
| communication telemetry | src/federated/server.py | yes | moderate |
| gradient telemetry | src/federated/server.py; src/federated/client.py | yes | low |
| model size/resource telemetry | src/federated/server.py; src/edge/benchmarker.py | yes | low |
| validation metric generation | src/utils/metrics.py; src/federated/server.py | yes | low |
| lightweight MLP model | src/models/mlp.py | yes | high |
| sequential simulator architecture | src/federated/server.py | yes | low |
| adaptive optimization controller | not implemented | no | not claim support |
| hospital deployment | not implemented | no | not claim support |
| sigma=2.3 completion | pending | no | not claim support |

## CLAIM-SUPPORT CHECK

For each claim element, the following check applies:

### Claim 1
1. Is it implemented? Yes.
2. Where is it implemented? src/federated/client.py, src/federated/server.py, src/privacy/dp_mechanism.py, src/data/partition.py
3. Is it experimentally verified? Yes, to the extent of the implemented scientific workflow and the verified sigma=0.8 pilot.
4. Is it merely an alternative embodiment? Not as drafted; it reflects the implemented architecture.
5. Is it potentially conventional? Some portions are conventional, particularly FedAvg, MLP, and standard Gaussian DP primitives. The interplay is the more defensible drafting point.
6. Does it depend on sigma=2.3? No.
7. Does it depend on hardware not currently demonstrated? No.

### Claim 2
1. Is it implemented? Yes.
2. Where is it implemented? src/federated/client.py and src/federated/server.py
3. Is it experimentally verified? Yes, in the verified pilot and the broader project architecture.
4. Is it merely an alternative embodiment? It matches the implemented workflow.
5. Is it potentially conventional? The larger sequence is conventional in structure but the specific record-level DP-SGD integration is the main technical point.
6. Does it depend on sigma=2.3? No.
7. Does it depend on hardware not currently demonstrated? No.

### Dependent claims
- Claim 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, and 14 are all supported by the code. Claims 8–11 are the strongest because they reflect the project telemetry and make the system technically observable.
- Claims 12 and 14 should be drafted carefully because the model type and simulation architecture are implemented but not independently novel.

---

# PART C — DISCLOSURE GAPS

## Items Requiring Human/Patent-Professional Review

The following items should be reviewed by a patent professional or a qualified human reviewer before any legal filing or claim narrowing decision:

- inventor identity
- applicant/assignee
- ownership and invention-assignment obligations
- jurisdiction
- filing type
- priority date
- any prior public disclosure
- exact legal claim scope
- patentability assessment
- claim charting against exact prior art publications
- whether a provisional or non-provisional filing is appropriate

The current project does not provide legal or ownership details, and no legal conclusion is made in this draft. The technical content is limited to the implemented architecture and verified evidence.

---

# PART D — FINAL STATUS

### Strongest independent claim

The strongest independent claim is a privacy-aware federated learning system in which each participating clinical node retains a local dataset, performs local preprocessing, uses a record-level DP-SGD mechanism that clips per-record gradients and adds Gaussian noise before the local optimizer step, transmits the resulting local update to a coordinator, and receives a weighted aggregation to generate the updated shared model while recording privacy and communication telemetry associated with the training process. This claim is supported by the implemented architecture and the verified technical evidence without overclaiming adaptive optimization or a physical deployment runtime.

### Strongest fallback claim

The strongest fallback claim is a narrower system or method claim directed to the local record-level DP-SGD training loop and the privacy-accounting/telemetry integration within a federated clinical prediction workflow. This fallback avoids depending on the broader concepts of general FL or general healthcare prediction alone and instead emphasizes the specific local privacy-preserving update generation sequence and its measurement in a federated learning pipeline.

### Biggest prior-art vulnerability

The biggest prior-art vulnerability is that the constituent elements are individually known: federated averaging, DP-SGD, Gaussian perturbation, and MLP-based prediction are all established techniques. The disclosed claim should therefore avoid broad drafting that reads on generic FL-plus-DP and instead focus on the specific technical interoperation of local record-level privacy-per-example clipping, weighted server aggregation, and privacy/communication telemetry within a clinical learning architecture.

### Evidence gap

The remaining evidence gaps are:
- no final verified sigma=2.3 result
- no demonstrated physical hospital deployment
- no adaptive optimization controller
- no final legal claim chart or examiner-facing prior-art comparison
- no inventor/ownership/legal status data
- no jurisdiction-specific filing analysis

### STEP 6

STEP 6 should produce:
PATENT FIGURES + DRAWING SPECIFICATIONS.

This should include:
- FIG. 1: overall architecture
- FIG. 2: local clinical node
- FIG. 3: record-level DP-SGD processing
- FIG. 4: federated aggregation workflow
- FIG. 5: privacy accounting block
- FIG. 6: communication and resource telemetry

Each figure should include a title, a brief technical narrative, a block-level description, and a correlation to the corresponding claim elements and implementation sources in the code.

---

PATENT WORKFLOW: 5/9

This document is a technical claim-and-specification scaffold only and should be reviewed by a patent professional before any legal or filing decision.
