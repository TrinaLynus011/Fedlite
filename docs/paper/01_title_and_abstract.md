# 01. Title and Abstract

## Title
FedLite-Care: A Lightweight Federated Learning Framework for Privacy-Preserving Hospital Readmission Prediction

## Abstract
FedLite-Care studies a lightweight federated learning workflow for hospital readmission prediction under privacy constraints. The framework combines local preprocessing, a compact MLP model, FedAvg aggregation, and DP-SGD with record-level clipping and Gaussian noise. The current implementation has been audited and validated in part, including a verified sigma=0.8 pilot run with epsilon=11.6717 and AUROC=0.6340. A sigma=2.3 calibration target for epsilon≈5 remains a pending compute item until final artifact validation is completed. The project emphasizes reproducibility, privacy accounting telemetry, and a clear distinction between verified experimental evidence and pending calibration work.
