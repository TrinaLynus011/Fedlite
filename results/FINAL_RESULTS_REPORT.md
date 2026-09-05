# Final Results Report

## Verified experiments

### Centralized baseline
- Partition: IID
- DP: No
- AUROC: 0.6247
- AUPRC: 0.1833
- F1: 0.2517
- Precision: 0.1649
- Recall: 0.5313
- Specificity: 0.6715
- Balanced accuracy: 0.6014
- Brier score: 0.2175
- Training time: 173 s
- Communication: 1.90 MB
- Status: VERIFIED

### Sigma=0.8 DP-SGD pilot
- Partition: IID
- DP: Yes
- Mechanism: DP-SGD
- Privacy unit: record
- Noise multiplier: 0.8
- Accounted epsilon: 11.6717
- AUROC: 0.6340
- AUPRC: 0.1876
- F1: 0.0957
- Precision: 0.3397
- Recall: 0.0557
- Specificity: 0.9867
- Balanced accuracy: 0.5212
- Brier score: 0.1027
- Training time: approximately 5085 s
- Communication: 1.90 MB
- Status: VERIFIED

## Calibrated but pending computation

### Sigma=2.3 target configuration
- Partition: Non-IID
- DP: Yes
- Mechanism: DP-SGD
- Privacy unit: record
- Noise multiplier: 2.3
- Target epsilon: approximately 5
- Status: PENDING_COMPUTE

The project remains in a scientifically defensible state because the sigma=2.3 run has not yet produced a verified final artifact set. No result is estimated or inferred; the experiment remains pending until the final process exits successfully and writes the required telemetry files.
