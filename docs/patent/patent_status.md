# FedLite-Care Patent Status

## Current invention title
FedLite-Care — “A Lightweight Federated Learning Framework for Privacy-Preserving Hospital Readmission Prediction on Resource-Constrained Edge Devices.”

## Current patent-readiness status
PRELIMINARY

The repository contains a concrete, implemented adaptive federated-learning controller and a valid static-versus-adaptive experiment, but the current evidence does not establish a definitive novelty or patentability position. The candidate inventive concept remains a constrained, closed-loop control mechanism and should be treated as a patent-preparation boundary, not as a legal conclusion.

## Verified implementation
VERIFIED

The implementation currently includes:

- Federated learning with a sequential simulator in `src/federated/server.py`
- A deterministic adaptive controller in `src/federated/adaptive_controller.py`
- Experiment orchestration in `src/federated/train_federated.py`
- Candidate configuration generation spanning:
  - compression ratio
  - local training epochs
  - client participation fraction
- Real dataset: `Dataset/diabetes_130_raw.csv`
- Target outcome: `readmitted`: `<30 = 1`, otherwise `0`
- Privacy-accountant and communication accounting paths integrated with the simulator
- Edge telemetry support in `src/edge/`
- An edge virtual-telemetry simulation and dashboard in `app/virtual_edge_dashboard.py`

## Verified experiments
VERIFIED

The strongest verified experiment in the repository is:

- `FL_MLP_NONIID_5_ADAPTIVE`
- Metadata: `results/experiment_metadata/FL_MLP_NONIID_5_ADAPTIVE.json`
- Registry: `results/experiment_registry.csv`

Static versus adaptive evidence:

| Metric | Static | Adaptive |
| --- | ---: | ---: |
| AUROC | 0.4835 | 0.5356 |
| AUPRC | 0.1059 | 0.1246 |
| Privacy epsilon | 4.9974 | 4.1397 |
| Total communication bytes | 1,997,200 | 761,131 |
| Training time (s) | 23,039.28 | 63,372.45 |

Communication reduction is approximately 61.9%.

Important interpretation:

- The adaptive run improved the reported AUROC and AUPRC and reduced communication.
- The adaptive run increased wall-clock runtime.
- The privacy accountant provides privacy information to the controller; the controller does not independently derive a new DP guarantee.
- The repository does not support a claim that the privacy objective alone caused the observed improvement.

## Failed or incomplete experiments
INCOMPLETE

The attempted privacy-weight ablation:

- `FL_MLP_NONIID_5_ADAPTIVE_PRIV_0`

was intended to compare the normal adaptive controller (`privacy = 1.0`) against a zero-weight privacy objective (`privacy = 0.0`).

However, the run did not produce a valid completed artifact or valid registry entry. The ablation therefore remains:

- INCOMPLETE
- INVALID as evidence
- NOT usable for patent support

This evidence must not be used to say that removing the privacy term improved performance or caused a specific result.

## Current novelty position
PRELIMINARY

The broad technical area is crowded. The repository is not novel in the general sense of:

- federated learning itself;
- healthcare FL itself;
- edge FL itself;
- resource-aware client selection itself;
- adaptive local epochs itself;
- compression itself;
- runtime-feedback loops itself;
- privacy accounts in FL itself.

The current candidate distinction is narrower and more technical:

> A constraint-gated closed-loop federated-learning configuration-control mechanism in which completed-round telemetry is converted into controller state; candidate configurations spanning client participation, local training effort, and update compression are generated; infeasible candidates are removed by predetermined privacy, communication, utility, and resource constraints; remaining candidates are deterministically evaluated using a multi-objective function; and the selected configuration is applied to a subsequent federated-learning round.

This should be treated as a candidate inventive concept, not as a confirmed legal novelty position.

## Strongest candidate inventive concept
VERIFIED / PRELIMINARY

The strongest candidate concept supported by the repository is the interaction:

```
completed-round telemetry
→ controller state
→ hard feasibility filtering
→ multidimensional candidate generation
→ deterministic multi-objective selection
→ next-round configuration
→ feedback loop
```

This is more specific than generic adaptive FL and is at least grounded in the implemented repository logic. However, the repository does not yet establish that this specific combination is novel over the prior art.

## Major risks
PRELIMINARY

The major risks include:

- obviousness risk from resource-aware and adaptive FL literature;
- crowded prior-art landscape across FedCS, FedDdrl, FedCG, CN115796271A, CN117252253A, FedCER, FedJoint, EcoFL, and related adaptive FL methods;
- risk that the implemented mechanism is a combination of known optimization ideas rather than a distinct and non-obvious technical contribution;
- risk that the virtual edge telemetry is supporting evidence only and should not be treated as a real hardware deployment.

## Immediate next step
Claim 1 → limitation-by-limitation prior-art attack and refinement.

This is the next step after the current documentation update. No expensive re-run of federated experiments should begin unless explicitly instructed.
