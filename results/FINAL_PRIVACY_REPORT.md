# Final Privacy Report

## Mechanism
- Active mechanism: DP-SGD
- Privacy unit: record
- Federated update rule: FedAvg with DP-SGD on local client gradients
- Validation boundary: validation data only; test set locked

## Verified privacy configuration
- Clipping norm: 1.0
- Delta: 1e-5
- Accountant: tight second-order subsampled Gaussian RDP via compute_dpsgd_epsilon()
- Verified pilot: sigma=0.8 => epsilon = 11.6717
- Target configuration: sigma=2.3 => approximately epsilon=5

## Pending privacy validation
- Experiment: FL_MLP_NONIID_5_DP_sigma_2.3
- Status: PENDING_COMPUTE
- Reason: no final process exit and no final telemetry artifact set were produced at the time of verification.

## Important limitation
- This project establishes record-level DP under the active DP-SGD training mechanism.
- It does not claim patient-level DP where multiple records may belong to the same patient without a separate analysis.
