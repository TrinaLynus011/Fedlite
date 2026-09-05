# FedLite-Care System Architecture

## High-level flow
Hospital/clinical node
↓
Local diabetes readmission data
↓
Local preprocessing
↓
Local lightweight MLP
↓
Per-record gradient computation
↓
Per-record L2 clipping
↓
Gradient aggregation
↓
Gaussian noise
↓
Local optimizer update
↓
FedAvg aggregation
↓
Global model
↓
Validation evaluation
↓
Privacy + utility telemetry

## Components
### Client
- Each hospital node owns a local dataset split.
- Local preprocessing is performed before training.
- Local training uses a lightweight MLP on client-side data.
- Each example produces a local gradient.

### Server / aggregator
- Aggregates client updates using FedAvg.
- Oversees the global model state.
- Tracks communication metadata and round-level metrics.
- Applies the privacy accountant using the active DP-SGD configuration.

### Privacy boundary
- Gradient clipping is performed per record before aggregation.
- Gaussian noise is added after per-record clipping and aggregation.
- The active privacy unit is record-level, not client-level.
- The privacy boundary is inside the federated training loop before the server update is finalized.

### Communication boundary
- Clients send locally trained parameter updates and telemetry.
- Server broadcasts global parameters to clients.
- Total communication is measured in bytes and recorded per round.

### Validation boundary
- Validation data is used for model selection and threshold tuning.
- Test data remains locked and is not used for training, tuning, or privacy accounting.

### Locked test set
- The project explicitly avoids test-set access during training or threshold selection.
- This constraint is preserved to prevent leakage and preserve scientific integrity.

## Relevant implementation paths
- Client logic: src/federated/client.py
- Server simulation logic: src/federated/server.py
- Privacy accounting: src/privacy/dp_mechanism.py
- Model configuration: configs/models.yaml
- Experiment metadata: results/experiment_metadata
