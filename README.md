# FedLite-Care

FedLite-Care is a sequential federated learning simulation for privacy-preserving readmission prediction in a hospital data setting. The repository is designed as a reproducible research and engineering prototype for evaluating a static federated baseline and an adaptive controller that reacts to completed-round telemetry.

> This repository is a sequential federated learning simulation and not a production distributed hospital deployment.

## Project purpose
The project studies how a federated learning system can preserve utility while tracking privacy, communication cost, and runtime under a controlled simulator. It focuses on a readmission prediction task using a lightweight MLP model and a non-IID 5-client split.

The repository includes:
- a static FedAvg baseline
- an adaptive round-to-round controller
- DP-SGD privacy accounting
- communication compression experiments
- reproducible experiment metadata and round metrics
- validation and regression tests

## Dataset
The project uses the processed hospital readmission dataset under the repository’s data pipeline. The training/validation/test split is created from the processed files in `data/processed`.

The simulator uses:
- `train_preprocessed.csv`
- `val_preprocessed.csv`
- `test_preprocessed.csv`

The current experimental focus is a non-IID five-client split with Dirichlet alpha = 0.5.

## Architecture
The software is organized into the following high-level components:

- `src/data/`: preprocessing, split logic
- `src/federated/`: server, client, adaptive controller, experiment entry points
- `src/privacy/`: DP-SGD accounting and privacy utilities
- `src/compression/`: update sparsification/compression logic
- `src/models/`: model implementations
- `src/api/`: API endpoints for smoke validation
- `tests/`: project regression tests
- `configs/`: YAML configuration for data, privacy, compression, training, and controller behavior
- `results/`: experiment outputs and metadata
- `artifacts/checkpoints/federated/`: saved model checkpoints

## Static federated baseline
The static baseline is implemented through the `static` mode of `src/federated/train_federated.py`.

It runs a fixed-configuration federated simulator using:
- full participation fraction
- fixed local epochs
- fixed compression ratio
- fixed privacy settings

No controller decision loop is active in this mode.

## Adaptive controller
The adaptive controller is implemented in `src/federated/adaptive_controller.py` and connected in `src/federated/server.py`.

It works as a closed loop:

INPUT
  ↓
FEDERATED TRAINING
  ↓
ROUND TELEMETRY
  ↓
CONTROLLER OBSERVATION
  ↓
CONSTRAINT EVALUATION
  ↓
CANDIDATE CONFIGURATION EVALUATION
  ↓
ACTION SELECTION
  ↓
NEXT-ROUND CONFIGURATION
  ↓
RESULTING TELEMETRY

The controller explicitly uses completed-round telemetry to select the next-round configuration for:
- compression ratio
- local epochs
- client participation fraction

The selected values are applied to the next round only.

## Privacy mechanism
The active mechanism is DP-SGD, which performs per-example gradient clipping and Gaussian noise injection during local training.

Implemented components:
- `src/federated/client.py` performs per-example clipping and noise during local updates
- `src/privacy/dp_mechanism.py` implements the accountant and clipping helpers
- `src/federated/server.py` reports privacy telemetry after each round

Current active settings in the tracked experiments are:
- mechanism: `dpsgd`
- privacy unit: record
- clipping norm: 1.0
- delta: 1e-5
- noise multiplier: 2.3 for the main static/adaptive experiment pair

The accountant is a Rényi-DP / subsampled Gaussian formulation used to estimate epsilon from the simulated number of training steps.

## Compression
The communication compression path is implemented in `src/compression/sparsification.py` and applied in the client update path in `src/federated/client.py`.

The controller may select compression ratios such as:
- 1.0
- 0.5
- 0.25
- 0.1
- 0.01

The compression logic reduces the reported transmitted update size and is reflected in communication accounting.

## Experiment modes
Two validated experiment modes are supported:

1. `static`
   - runs the fixed configuration baseline
   - uses `adaptive_controller.enabled = false`

2. `adaptive`
   - runs the telemetry-driven controller loop
   - uses `adaptive_controller.enabled = true`

3. `compare`
   - prints the static-vs-adaptive comparison summary using saved artifacts

Execution examples:

```powershell
.\.venv\Scripts\python src/federated/train_federated.py static
.\.venv\Scripts\python src/federated/train_federated.py adaptive
.\.venv\Scripts\python src/federated/train_federated.py compare
```

## Installation
This project uses a local virtual environment.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

If a requirements file is not present in the repo, install the dependencies declared in the project environment used for validation, including at minimum:
- PyTorch
- NumPy
- pandas
- scikit-learn
- PyYAML
- flwr
- joblib
- fastapi
- uvicorn

## Tests
The repository contains regression tests for the adaptive controller and broader federated simulation behavior.

Run:

```powershell
.\.venv\Scripts\pytest -q tests/test_adaptive_controller.py tests/test_fedlite_care.py
```

## Results
The repository records evidence in:
- `results/experiment_metadata/`
- `results/experiment_registry.csv`
- `results/*_round_metrics.csv`
- `figures/`

The final verified comparison in this repository is:

- Static AUROC: 0.4835
- Adaptive AUROC: 0.5356
- Static AUPRC: 0.1059
- Adaptive AUPRC: 0.1246
- Static communication bytes: 1,997,200
- Adaptive communication bytes: 761,131
- Static epsilon: 4.9974
- Adaptive epsilon: 4.1397

## Known limitations
- This is a sequential federated simulation, not a production distributed hospital deployment.
- The communication model is an estimated parameter-based accounting model.
- The controller is heuristic and configuration-driven.
- Privacy accounting is based on the implemented DP-SGD accountant and should be interpreted as a simulation approximation.
- Runtime is significantly higher in the adaptive mode because it performs more varied training configurations and a controller decision loop.
- Clinical deployment, hospital governance validation, and real edge deployment are outside the scope of this repository.

## Reproducibility notes
The repository includes the produced evidence artifacts for the final static and adaptive runs. Re-running the full experiments is computationally expensive, but the workflow can be validated through the project test suite and the saved metadata artifacts.

## Repository status
The current codebase is a verified software freeze for the static baseline and adaptive controller before any later IoT or edge extension work.
