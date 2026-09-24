# External MCU Telemetry Simulation

This directory contains a minimal Wokwi-style external telemetry source for the FedLite-Care simulation.

Important:
- The node is simulated and deliberately labeled as `mcu_simulation`.
- The payload is intended to model an external hospital edge device, not the local host computer.
- The current Wokwi implementation is a simulation of an external MCU telemetry source and does not constitute physical-hardware validation.
- CPU and memory values are MCU-local simulated measurements, not host CPU or RAM measurements.
- The firmware emits a machine-readable JSON record alongside the human-readable serial output so that the telemetry can be consumed by a local ingest adapter without pretending to be a real network connection.

The firmware emits a structured payload that is normalized by the same `EdgeTelemetry` contract used by the host-side software telemetry path.
