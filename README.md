# FedLite-Care

## 1. Product overview
FedLite-Care is a lightweight federated learning framework for privacy-preserving hospital readmission prediction.

## 2. Problem
Healthcare data are distributed across hospitals and are difficult to centralize because of privacy, governance, and operational constraints. At the same time, edge environments require efficient training and communication workflows. Model development must preserve measurable utility while applying privacy controls that remain interpretable and reproducible.

## 3. Solution
The project combines:
- federated learning with FedAvg
- a lightweight MLP model
- DP-SGD for record-level privacy
- validation-based evaluation with locked test data
- privacy and utility telemetry

This produces a resource-aware training workflow that can be evaluated under realistic federated learning constraints without centralizing raw hospital data.

## 4. Privacy
The active mechanism is:
- mechanism = DP-SGD
- privacy unit = record
- clipping norm = 1.0
- delta = 1e-5
- accountant = tight second-order subsampled Gaussian RDP

The implementation is documented in the privacy accounting code and the experiment metadata produced during training.

## 5. Verified pilot
The following result was actually produced and verified in the project telemetry:

- sigma = 0.8
- epsilon = 11.6717
- AUROC = 0.6340
- AUPRC = 0.1876
- training time ≈ 5085 seconds
- communication ≈ 1.90 MB

This result is the verified pilot result in the current project record.

## 6. Limitations
The current project has several clear limitations:
- DP-SGD is computationally expensive compared with non-private training.
- The current experiments are run in a simulated federated environment.
- Physical edge hardware validation has not been established.
- Clinical validation has not been performed.
- External patent prior-art analysis remains required.

The project does not claim HIPAA compliance, clinical deployment, real hospital deployment, hardware validation, or guaranteed anonymity beyond the specific evidence currently generated.

## Repository status
- Verified tests: 40/40 pass in the project virtual environment.
- Verified pilot: sigma=0.8 DP-SGD run with epsilon=11.6717 and AUROC=0.6340.
- Calibrated target: sigma=2.3 is the approximate epsilon=5 target configuration.
- Status of sigma=2.3: PENDING_COMPUTE until the run produces final verified artifacts.
