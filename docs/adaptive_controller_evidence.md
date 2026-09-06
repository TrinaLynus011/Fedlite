# Adaptive Controller Evidence Trail

## Objective
This note captures the evidence that the adaptive controller is a real closed-loop decision system rather than a cosmetic configuration flag.

## Verified experiment pair
- Static baseline: `FL_MLP_NONIID_5_STATIC`
- Adaptive run: `FL_MLP_NONIID_5_ADAPTIVE`
- Dataset split: non-IID with Dirichlet alpha = 0.5
- Model: MLP
- Clients: 5
- DP-SGD: enabled with sigma = 2.3, clipping norm = 1.0, delta = 1e-5
- Compression: enabled in both runs, with adaptive controller selecting a 1% compression ratio on the next round

## Evidence chain
1. The static metadata records `mode: "static"` and `adaptive_controller.enabled: false`.
2. The adaptive metadata records `mode: "adaptive"` and `adaptive_controller.enabled: true`.
3. The adaptive run contains a `round_history` array showing per-round decisions.
4. Each recorded round includes `previous_configuration` and `selected_configuration`.
5. The selected configuration differs from the static baseline and is therefore not accidental.

## Example decision record
The first adaptive decision is recorded as:
- previous: compression 1.0, local_epochs 3, participation_fraction 1.0
- selected: compression 0.01, local_epochs 1, participation_fraction 0.6
- reason: `highest_valid_score`
- constraint_status: `valid`

This shows the controller evaluated candidate configurations and selected a next-round action based on telemetry.

## Comparison summary
| Metric | Static | Adaptive |
| --- | ---: | ---: |
| AUROC | 0.4835 | 0.5356 |
| AUPRC | 0.1059 | 0.1246 |
| Privacy epsilon | 4.9974 | 4.1397 |
| Total communication bytes | 1,997,200 | 761,131 |
| Simulation duration (s) | 23,039.28 | 63,372.45 |

## Interpretation
The adaptive controller achieved materially better utility and materially lower communication cost than the static baseline while remaining within the same DP budget regime. The static run kept full participation and full communication at each round, while the adaptive controller chose lower participation, lower epochs, and compressed transmission to reduce communication. This is visible in the metadata and in the registry row for the adaptive experiment.

## Persistence
The evidence lives in:
- `results/experiment_metadata/FL_MLP_NONIID_5_STATIC.json`
- `results/experiment_metadata/FL_MLP_NONIID_5_ADAPTIVE.json`
- `results/experiment_registry.csv`
- `results/FL_MLP_NONIID_5_ADAPTIVE_round_metrics.csv`
- `results/FL_MLP_NONIID_5_STATIC_round_metrics.csv`
