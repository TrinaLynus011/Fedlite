"""Edge telemetry and resource interpretation helpers for FedLite-Care."""

from .iot_telemetry import (
    EdgeTelemetry,
    EdgeTelemetryRegistry,
    build_edge_client_mapping,
    interpret_edge_resource,
    normalize_telemetry,
)
from .external_ingest import ExternalTelemetryIngest
from .telemetry_sources import (
    HostTelemetrySource,
    McuTelemetrySource,
    TelemetryAdapter,
    TelemetrySource,
)
from .wokwi_bridge import WokwiLaptopBridge

__all__ = [
    "EdgeTelemetry",
    "EdgeTelemetryRegistry",
    "build_edge_client_mapping",
    "interpret_edge_resource",
    "normalize_telemetry",
    "ExternalTelemetryIngest",
    "WokwiLaptopBridge",
    "TelemetrySource",
    "HostTelemetrySource",
    "McuTelemetrySource",
    "TelemetryAdapter",
]
