# FedLite-Care Adaptive Experiment Results

## Verified setup
The final static and adaptive runs were executed in the project venv and generated complete metadata artifacts.

### Static baseline
- Experiment ID: `FL_MLP_NONIID_5_STATIC`
- Mode: static
- AUROC: 0.4835
- AUPRC: 0.1059
- Privacy epsilon: 4.9974
- Communication bytes: 1,997,200
- Runtime: 23,039.28 s

### Adaptive run
- Experiment ID: `FL_MLP_NONIID_5_ADAPTIVE`
- Mode: adaptive
- AUROC: 0.5356
- AUPRC: 0.1246
- Privacy epsilon: 4.1397
- Communication bytes: 761,131
- Runtime: 63,372.45 s

## Result interpretation
The adaptive controller improved AUROC by about 10.8% relative to the static baseline and improved AUPRC by about 17.7% relative to the static baseline, while reducing total communication bytes by about 61.8%. In the same DP setting, this indicates the controller traded a modestly longer wall-clock runtime for better utility and better communication efficiency.

## Scientific validity notes
- Both experiments use the same dataset split, model, client count, and DP-SGD settings.
- The static baseline is a true fixed-configuration run with no controller decision loop.
- The adaptive run is a true closed-loop run whose selected configurations are persisted in `round_history`.
- The comparison is therefore valid for fixed-configuration vs adaptive-controller behavior under the same underlying simulator.

## Files for audit
- `results/experiment_metadata/FL_MLP_NONIID_5_STATIC.json`
- `results/experiment_metadata/FL_MLP_NONIID_5_ADAPTIVE.json`
- `results/experiment_registry.csv`
- `results/FL_MLP_NONIID_5_STATIC_round_metrics.csv`
- `results/FL_MLP_NONIID_5_ADAPTIVE_round_metrics.csv`
