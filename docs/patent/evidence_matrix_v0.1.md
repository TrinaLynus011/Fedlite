# Evidence Matrix v0.1

## Claim-to-evidence mapping

| Claim element | Source file / function | Experiment or artifact evidence | Status |
| --- | --- | --- | --- |
| Completed-round telemetry received by controller | `src/federated/adaptive_controller.py` — `AdaptiveFederatedController.observe()` | Round telemetry recorded in experiment metadata under `adaptive_controller.round_history` | VERIFIED |
| Controller state constructed from telemetry | `src/federated/adaptive_controller.py` — `AdaptiveFederatedController.__init__()`, `observe()`, `_resolve_state()` | `results/experiment_metadata/FL_MLP_NONIID_5_ADAPTIVE.json` | VERIFIED |
| Candidate configurations generated | `src/federated/adaptive_controller.py` — `_candidate_space()` | `configs/adaptive_controller.yaml` candidate values | VERIFIED |
| Candidate dimensions include participation, epochs, compression | `src/federated/adaptive_controller.py` — `_candidate_space()` | `configs/adaptive_controller.yaml` | VERIFIED |
| Feasibility constraints applied | `src/federated/adaptive_controller.py` — `_evaluate_constraints()` | Controller logic and metadata; static/adaptive experiment metadata | VERIFIED |
| Privacy-budget constraint | `src/federated/adaptive_controller.py` — `_evaluate_constraints()` | `configs/adaptive_controller.yaml` and experiment metadata | VERIFIED |
| Communication-budget constraint | `src/federated/adaptive_controller.py` — `_evaluate_constraints()` | Controller logic and communication metrics in experiment metadata | VERIFIED |
| Utility threshold constraint | `src/federated/adaptive_controller.py` — `_evaluate_constraints()` | `utility.minimum_auroc` / `minimum_auprc` in config | VERIFIED |
| Resource/time constraint | `src/federated/adaptive_controller.py` — `_evaluate_constraints()` | `resource.maximum_round_time_sec` in config | VERIFIED |
| Deterministic multi-objective scoring | `src/federated/adaptive_controller.py` — `_score_candidate()` | Round history and selected configuration in experiment metadata | VERIFIED |
| Selected config applied to subsequent round | `src/federated/server.py` — round loop and controller usage | `results/experiment_metadata/FL_MLP_NONIID_5_ADAPTIVE.json` round history | VERIFIED |
| Feedback loop persists across rounds | `src/federated/server.py` — round loop and `AdaptiveFederatedController` integration | `adaptive_controller.round_history` in metadata | VERIFIED |
| Experiment orchestration | `src/federated/train_federated.py` — `run_experiment()`, `save_experiment_metadata()` | `results/experiment_registry.csv` | VERIFIED |
| Valid adaptive experiment artifact | `results/experiment_metadata/FL_MLP_NONIID_5_ADAPTIVE.json` | Experiment metadata file | VERIFIED |
| Registry evidence | `results/experiment_registry.csv` | Registry entry for `FL_MLP_NONIID_5_ADAPTIVE` | VERIFIED |
| Static baseline artifact | `results/experiment_metadata/FL_MLP_NONIID_5_STATIC.json` | Baseline experimental record | VERIFIED |
| Failed ablation evidence | `FL_MLP_NONIID_5_ADAPTIVE_PRIV_0` | No valid artifact produced; no valid registry entry | INCOMPLETE |
| Virtual edge telemetry integration | `src/edge/iot_telemetry.py`, `src/edge/telemetry_sources.py`, `src/edge/wokwi_bridge.py`, `app/virtual_edge_dashboard.py` | Telemetry tests and docs | VERIFIED as supporting implementation |
| Hospital readmission use case | `Dataset/diabetes_130_raw.csv`, data pre-processing, model training pipeline | Repository training pipeline and metrics artifacts | VERIFIED |
| Real hardware deployment claim | `src/edge/` virtual simulation and Wokwi support | No physical hardware requirement in repo evidence | UNVERIFIED for real deployment |
| Legal novelty claim | Patent analysis documents only | No legal opinion or legal claim filing | UNVERIFIED |

## Additional repository references

- `src/federated/adaptive_controller.py`
- `src/federated/server.py`
- `src/federated/train_federated.py`
- `configs/adaptive_controller.yaml`
- `results/experiment_metadata/`
- `results/experiment_registry.csv`
- `src/edge/`
- `tests/test_iot_telemetry.py`
- `docs/invention_mechanism_audit.md`
- `docs/iot_edge_integration.md`

## Important evidence note

The matrix above intentionally excludes the failed `PRIV_0` ablation from support for any claim. It remains `INCOMPLETE` and should not be used as a patent-supporting fact.
