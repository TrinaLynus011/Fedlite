# Invention Disclosure

## 1. Invention title
FedLite-Care: Resource-Aware Federated Learning for Privacy-Preserving Hospital Readmission Prediction

## 2. Technical field
Healthcare machine learning, privacy-preserving distributed learning, edge-aware federated learning, clinical risk prediction, secure aggregation workflows.

## 3. Background / problem
Hospital datasets are fragmented across institutions, each holding sensitive patient information. Centralizing these data is difficult because of governance constraints, bandwidth limits, privacy risk, and operational complexity. At the same time, clinical prediction systems require robust utility and measurable privacy accounting.

## 4. Existing limitations
Conventional centralized training approaches require data pooling. Standard federated learning can reduce direct data sharing but may still be heavy for resource-constrained hospital edge environments. Privacy mechanisms can degrade utility if applied without careful accounting, and many systems do not provide clear telemetry across communication, utility, and privacy budgets.

## 5. Proposed system
The disclosure concerns a federated clinical prediction system that combines local preprocessing, lightweight MLP training, per-record gradient clipping, Gaussian noise, FedAvg aggregation, and resource-aware telemetry in a single end-to-end workflow.

## 6. System architecture
- local clinical node
- local data preprocessing
- local lightweight MLP model
- per-record gradient computation
- per-record L2 clipping
- Gaussian noise addition
- FedAvg aggregation
- validation evaluation
- privacy accounting and telemetry collection

## 7. Federated workflow
Each hospital maintains local training and evaluation data. Models are trained locally, parameter updates are aggregated centrally, and global metrics are monitored. The workflow preserves the privacy boundary without requiring raw data transfer.

## 8. DP-SGD workflow
The local updates are protected by DP-SGD under a record-level privacy analysis. The mechanism includes per-record clipping followed by Gaussian noise addition to the aggregated update.

## 9. Privacy accounting
The privacy accounting relies on the tight second-order subsampled Gaussian RDP implementation and the project accountant `compute_dpsgd_epsilon()`. The project records actual privacy telemetry, including clipping statistics and gradient norm measurements.

## 10. Resource-aware design
The system explicitly monitors communication volume and model size, with the objective of reducing operational overhead while maintaining a reproducible clinical learning workflow.

## 11. Communication architecture
The design tracks:
- upstream bytes
- downstream bytes
- total communication volume
- parameter count
- model size estimates

## 12. Telemetry
The project gathers:
- per-round validation metrics
- privacy telemetry
- clipping statistics
- gradient norm statistics
- communication totals
- model size metrics

## 13. Experimental evidence
Verified experimental evidence includes:
- 40/40 project tests pass
- verified sigma=0.8 DP-SGD pilot with epsilon=11.6717 and AUROC=0.6340
- calibrated sigma=2.3 target with expected epsilon ≈ 5 remains pending final completion

## 14. Alternative embodiments
Alternative implementations may vary in hospital count, local model capacity, legal data partitions, communication schedules, or privacy configurations while preserving the same overall architecture.

## 15. Potentially novel combinations
The potentially distinctive contribution is the specific combination of:
- resource-aware federated training
- record-level DP-SGD
- local hospital data processing
- communication telemetry
- reproducible privacy accounting within a lightweight clinical risk prediction system

## 16. Known / non-novel components
The following components are conventional prior art and should not be asserted as novel by themselves:
- FedAvg
- DP-SGD
- Gaussian noise
- MLP architecture
- differential privacy generally

## 17. Advantages
- reduced need for central data pooling
- privacy-aware learning workflow
- resource-aware communication tracking
- reproducible experimental and privacy telemetry
- measurable privacy/utility tradeoff

## 18. Limitations
This disclosure does not claim broad clinical deployment, patient-level privacy, or hardware validation beyond current evidence. The system is a research-grade implementation with a simulated federated environment.

## 19. Inventor contribution record
This contribution reflects the project implementation, audit, and verification work performed for the FedLite-Care research prototype.

## 20. Figures required for patent drafting
- high-level system architecture
- hospital node data flow
- DP-SGD local update workflow
- Federated aggregation workflow
- privacy-accounting block diagram
- communication and resource telemetry panels
