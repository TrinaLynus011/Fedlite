# Final Telemetry Report

## Verified telemetry sources
- Round metrics: [results/FL_MLP_IID_5_DP_SGD_sigma_0.8_round_metrics.csv](results/FL_MLP_IID_5_DP_SGD_sigma_0.8_round_metrics.csv)
- Round metrics JSON: [results/FL_MLP_IID_5_DP_SGD_sigma_0.8_round_metrics.json](results/FL_MLP_IID_5_DP_SGD_sigma_0.8_round_metrics.json)
- Calibration summary: [results/dp_calibration_summary.csv](results/dp_calibration_summary.csv)
- Calibration results: [results/dp_calibration_results.json](results/dp_calibration_results.json)

## Verified telemetry values for sigma=0.8
- Final AUROC: 0.6339772352589269
- Final AUPRC: 0.18763080421483927
- Mean gradient norm: 7.1435
- Median gradient norm: 0.1177
- Fraction clipped: 0.1156
- Communication total: 199,720 bytes (approximately 0.19 MB per round, 1.90 MB total across the 10-round run)
- Parameter count: 4,993
- Model size: 19.5 KB FP32 theoretical size

## Pending telemetry
- Sigma=2.3 run has not yet produced final process completion, metadata, checkpoint, or final round metrics.
- Status: PENDING_COMPUTE
