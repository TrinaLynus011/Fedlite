# IoT Edge Integration

## 1. Architecture

Representative Wokwi ESP32
        ↓
Wokwi-compatible telemetry schema
        ↓
Python telemetry adapter
        ↓
Multi-node edge registry
        ↓
FL client eligibility
        ↓
Federated round execution
        ↓
Existing round_metrics_dict
        ↓
Adaptive controller
        ↓
Next-round configuration

## 2. What is physically simulated

- The ESP32 is simulated in Wokwi as a representative hospital edge-node resource monitor.
- The current Wokwi implementation is a simulation of an external MCU telemetry source and does not constitute physical-hardware validation.
- The Python project simulates federated learning clients in a local sequential runtime.
- The ESP32 does not train the ML model. It only emits resource telemetry that is translated into a client-eligibility context for the existing FL simulation.

## 3. What is integrated

The integration path is intentionally conservative and uses the existing FedLite-Care control loop rather than introducing a new controller.

The real execution path is:

1. Wokwi-compatible payload is normalized by `normalize_telemetry()`.
2. The adapter converts the raw payload into a deterministic `EdgeTelemetry` dataclass.
3. The registry manages the latest record per edge node.
4. Resource interpretation marks the node as eligible or ineligible and computes a deterministic participation weight.
5. Edge client mapping associates hospital edge nodes to simulated FL clients.
6. The server filters the client pool before participation.
7. The remaining round runs with the existing `round_metrics_dict`.
8. The existing adaptive controller still owns global configuration selection via `observe(...)` and `select_next_configuration(...)`.

## 4. External Telemetry Ingest Path

The repository now includes a minimal external ingest adapter that accepts machine-readable JSON telemetry from an external or simulated source and routes it through the existing packet-validation and registry flow.

### Implemented

- Wokwi ESP32 telemetry generation
- machine-readable JSON telemetry emission
- external ingest interface that accepts JSON records
- packet-level validation and source recognition
- sequence checking and stale detection
- normalization through `TelemetryPacket` and `TelemetryAdapter`
- registry update via `EdgeTelemetryRegistry`
- controller-facing resource interpretation via `interpret_edge_resource()`
- automated integration tests covering valid, duplicate, stale, malformed, source, and server paths

### Simulated

- ESP32 hardware behavior
- telemetry values
- Wokwi node state transitions
- network conditions unless those values are directly measured

### Not validated

- physical ESP32 hardware
- real hospital network behavior
- real external network packet timing
- real-time clinical deployment
- live Wokwi-to-Python transport

The ingest path is intentionally conservative: it accepts JSON exactly as an external source would provide it, but it does not claim physical hardware validation or a live network connection.

## 5. Transport limitation

This implementation uses a local machine-readable ingest path rather than a fake live network stack. It accepts external JSON telemetry and reuses the existing Python normalization and registry logic. This is the minimum reproducible ingest layer needed to connect Wokwi-style telemetry to the server/controller flow without claiming a real physical ESP32 deployment.

## 6. Multi-node architecture

One representative ESP32 simulation is treated as a single resource monitor. The Python registry supports multiple node IDs and keeps the latest telemetry for each node in memory. This allows a single simulation to model a scalable hospital deployment while still preserving deterministic testing and reproducible experiments.

## 6. Deterministic mapping rules

The edge interpretation is intentionally transparent:

- AVAILABLE + READY -> eligible, high suitability
- BUSY -> eligible, reduced suitability
- CONSTRAINED -> restricted suitability, but may still participate when above threshold
- OFFLINE -> ineligible
- CPU load, memory pressure, and network quality reduce suitability deterministically
- participation_status and state values are normalized to uppercase and validated explicitly

## 7. Controller integration

The existing adaptive controller remains the sole configuration-selection mechanism. The edge layer influences the eligible client pool and resource context, but the global FL decisions for participation fraction, local epochs, and compression ratio continue to be chosen by the controller from `round_metrics_dict`.
