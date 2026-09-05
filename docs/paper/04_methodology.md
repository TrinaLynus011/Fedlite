# 04. Methodology

The project implements a lightweight federated learning pipeline for hospital readmission prediction. Each client trains locally on hospital-specific data; the server aggregates updates using FedAvg; and the training loop maintains privacy telemetry. The core privacy mechanism is DP-SGD with record-level clipping and Gaussian noise addition after per-record gradient clipping.

This methodology is deliberately constrained to preserve reproducibility and to avoid modifying the active mechanism in response to minor metric variability.
