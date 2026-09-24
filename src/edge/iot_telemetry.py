from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Optional


VALID_STATES = {"AVAILABLE", "BUSY", "CONSTRAINED", "OFFLINE"}
VALID_PARTICIPATION_STATUSES = {"READY", "LIMITED", "RESTRICTED", "UNAVAILABLE"}
VALID_SOURCES = {"host_software", "mcu_simulation", "mcu_hardware", "gateway"}


@dataclass(frozen=True)
class EdgeTelemetry:
    """Normalized edge telemetry payload.

    The data model is intentionally narrow but source-aware. Existing FL logic
    only consumes the normalized resource state and participation signal. The
    additional metadata makes it possible to distinguish host-side telemetry from
    an independent MCU simulation channel.
    """

    node_id: str
    state: str
    cpu_load: float
    memory_pressure: float
    network_quality: float
    participation_status: str
    timestamp: str
    source: str = "host_software"
    sequence_id: int = 0
    generated_timestamp: Optional[str] = None
    received_timestamp: Optional[str] = None
    latency_sec: Optional[float] = None
    stale: bool = False
    valid: bool = True
    rejection_reason: Optional[str] = None


def _require_field(payload: Dict[str, Any], field: str) -> Any:
    if field not in payload:
        raise ValueError(f"Missing required field: {field}")
    return payload[field]


def _validate_percentage(name: str, value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric.") from None

    if not (0.0 <= numeric <= 100.0):
        raise ValueError(f"{name} must be in the range [0, 100].")
    return numeric


def normalize_telemetry(payload: Dict[str, Any]) -> EdgeTelemetry:
    """Validate and normalize a raw edge telemetry payload.

    Supported states: AVAILABLE, BUSY, CONSTRAINED, OFFLINE.
    Supported participation statuses: READY, LIMITED, RESTRICTED, UNAVAILABLE.
    Supported sources: host_software, mcu_simulation, mcu_hardware, gateway.
    """
    if payload is None:
        raise ValueError("Telemetry payload cannot be None.")
    if not isinstance(payload, dict):
        raise TypeError("Telemetry payload must be a dictionary.")

    node_id = _require_field(payload, "node_id")
    state = str(_require_field(payload, "state")).upper()
    participation_status = str(_require_field(payload, "participation_status")).upper()
    source = str(payload.get("source", "host_software")).lower()
    if source not in VALID_SOURCES:
        valid = ", ".join(sorted(VALID_SOURCES))
        raise ValueError(f"Invalid source '{source}'. Supported sources: {valid}.")

    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("node_id must be a non-empty string.")

    if state not in VALID_STATES:
        valid = ", ".join(sorted(VALID_STATES))
        raise ValueError(f"Invalid state '{state}'. Supported states: {valid}.")

    if participation_status not in VALID_PARTICIPATION_STATUSES:
        valid = ", ".join(sorted(VALID_PARTICIPATION_STATUSES))
        raise ValueError(
            f"Invalid participation_status '{participation_status}'. Supported statuses: {valid}."
        )

    cpu_load = _validate_percentage("cpu_load", _require_field(payload, "cpu_load"))
    memory_pressure = _validate_percentage("memory_pressure", _require_field(payload, "memory_pressure"))
    network_quality = _validate_percentage("network_quality", _require_field(payload, "network_quality"))

    timestamp = payload.get("timestamp") or payload.get("generated_timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("timestamp or generated_timestamp must be a non-empty string.")

    sequence_id = payload.get("sequence_id", 0)
    try:
        sequence_id = int(sequence_id)
    except (TypeError, ValueError):
        raise ValueError("sequence_id must be an integer.") from None

    generated_timestamp = payload.get("generated_timestamp")
    if generated_timestamp is not None and (not isinstance(generated_timestamp, str) or not generated_timestamp.strip()):
        raise ValueError("generated_timestamp must be a non-empty string when provided.")

    return EdgeTelemetry(
        node_id=node_id.strip(),
        state=state,
        cpu_load=cpu_load,
        memory_pressure=memory_pressure,
        network_quality=network_quality,
        participation_status=participation_status,
        timestamp=timestamp.strip(),
        source=source,
        sequence_id=sequence_id,
        generated_timestamp=(generated_timestamp.strip() if isinstance(generated_timestamp, str) else None),
    )


def interpret_edge_resource(payload: Dict[str, Any] | EdgeTelemetry | None) -> Dict[str, Any]:
    """Convert a normalized edge telemetry record into deterministic FL eligibility.

    Interpretation rules:
    - OFFLINE -> always ineligible, participation_weight = 0.0
    - AVAILABLE + READY -> eligible with strongest suitability
    - BUSY -> eligible but lower suitability
    - CONSTRAINED -> restricted suitability
    - CPU, memory, and network quality degrade values deterministically while still
      preserving the state-based floor for each state.
    """
    if payload is None:
        return {
            "node_id": None,
            "eligible_for_training": False,
            "resource_pressure": 1.0,
            "network_condition": "WEAK",
            "participation_weight": 0.0,
            "state": "OFFLINE",
            "participation_status": "UNAVAILABLE",
            "cpu_load": 0.0,
            "memory_pressure": 0.0,
            "network_quality": 0.0,
        }

    if isinstance(payload, EdgeTelemetry):
        telemetry = payload
    else:
        telemetry = normalize_telemetry(payload)

    state = telemetry.state
    status = telemetry.participation_status

    base_floor = {
        "AVAILABLE": 0.9,
        "BUSY": 0.7,
        "CONSTRAINED": 0.35,
        "OFFLINE": 0.0,
    }.get(state, 0.0)

    status_factor = {
        "READY": 1.0,
        "LIMITED": 0.8,
        "RESTRICTED": 0.45,
        "UNAVAILABLE": 0.0,
    }.get(status, 0.0)

    cpu_pressure = telemetry.cpu_load / 100.0
    memory_pressure = telemetry.memory_pressure / 100.0
    network_strength = telemetry.network_quality / 100.0

    resource_multiplier = 1.0 - (0.12 * cpu_pressure + 0.10 * memory_pressure + 0.12 * max(0.0, 1.0 - network_strength))
    raw_weight = base_floor * status_factor * max(0.2, resource_multiplier)
    participation_weight = max(0.0, min(1.0, raw_weight))

    if state == "OFFLINE" or status == "UNAVAILABLE":
        eligible = False
        participation_weight = 0.0
    elif state == "AVAILABLE" and status == "READY":
        eligible = True
        participation_weight = max(0.9, participation_weight)
    elif state == "BUSY":
        eligible = True
        participation_weight = max(0.7, participation_weight)
    elif state == "CONSTRAINED":
        eligible = participation_weight > 0.1
        participation_weight = max(0.35, participation_weight)
    else:
        eligible = participation_weight > 0.1

    resource_pressure = 0.0
    if state == "BUSY":
        resource_pressure = 0.55
    elif state == "CONSTRAINED":
        resource_pressure = 0.8
    elif state == "OFFLINE":
        resource_pressure = 1.0

    network_condition = "STRONG" if telemetry.network_quality >= 75 else "MODERATE" if telemetry.network_quality >= 40 else "WEAK"

    return {
        "node_id": telemetry.node_id,
        "eligible_for_training": bool(eligible),
        "resource_pressure": round(resource_pressure, 4),
        "network_condition": network_condition,
        "participation_weight": round(participation_weight, 4),
        "state": telemetry.state,
        "participation_status": telemetry.participation_status,
        "cpu_load": telemetry.cpu_load,
        "memory_pressure": telemetry.memory_pressure,
        "network_quality": telemetry.network_quality,
    }


class EdgeTelemetryRegistry:
    """Tracks the latest telemetry record for each edge node and exposes summaries."""

    def __init__(self, node_ids: Optional[Iterable[str]] = None):
        self._latest_by_node: Dict[str, EdgeTelemetry] = {}
        if node_ids is not None:
            for node_id in node_ids:
                self._latest_by_node[str(node_id)] = None  # type: ignore[assignment]

    def update(self, payload: Dict[str, Any] | EdgeTelemetry | List[Dict[str, Any]]) -> None:
        if isinstance(payload, list):
            for item in payload:
                self.update(item)
            return

        telemetry = payload if isinstance(payload, EdgeTelemetry) else normalize_telemetry(payload)
        self._latest_by_node[telemetry.node_id] = telemetry

    def get_latest(self, node_id: str) -> Optional[EdgeTelemetry]:
        return self._latest_by_node.get(node_id)

    def update_many(self, payloads: Iterable[Dict[str, Any]]) -> None:
        for payload in payloads:
            self.update(payload)

    def get_resource_summary(self) -> Dict[str, Any]:
        records = [item for item in self._latest_by_node.values() if item is not None]
        if not records:
            return {
                "edge_total_nodes": 0,
                "edge_available_nodes": 0,
                "edge_busy_nodes": 0,
                "edge_constrained_nodes": 0,
                "edge_offline_nodes": 0,
                "edge_eligible_clients": 0,
                "edge_mean_cpu_load": 0.0,
                "edge_mean_memory_pressure": 0.0,
                "edge_mean_network_quality": 0.0,
                "edge_resource_pressure": 1.0,
                "edge_nodes": [],
            }

        eligible_clients = 0
        states = {"AVAILABLE": 0, "BUSY": 0, "CONSTRAINED": 0, "OFFLINE": 0}
        resource_pressure_total = 0.0
        for record in records:
            states[record.state] = states.get(record.state, 0) + 1
            interpretation = interpret_edge_resource(record)
            if interpretation["eligible_for_training"]:
                eligible_clients += 1
            resource_pressure_total += interpretation["resource_pressure"]

        valid_nodes = []
        for record in records:
            interpretation = interpret_edge_resource(record)
            valid_nodes.append({
                "node_id": record.node_id,
                "state": record.state,
                "participation_status": record.participation_status,
                "eligible_for_training": interpretation["eligible_for_training"],
                "participation_weight": interpretation["participation_weight"],
                "network_condition": interpretation["network_condition"],
            })

        summary = {
            "edge_total_nodes": len(records),
            "edge_available_nodes": states.get("AVAILABLE", 0),
            "edge_busy_nodes": states.get("BUSY", 0),
            "edge_constrained_nodes": states.get("CONSTRAINED", 0),
            "edge_offline_nodes": states.get("OFFLINE", 0),
            "edge_eligible_clients": eligible_clients,
            "edge_mean_cpu_load": round(sum(r.cpu_load for r in records) / len(records), 4),
            "edge_mean_memory_pressure": round(sum(r.memory_pressure for r in records) / len(records), 4),
            "edge_mean_network_quality": round(sum(r.network_quality for r in records) / len(records), 4),
            "edge_resource_pressure": round(resource_pressure_total / len(records), 4),
            "edge_nodes": valid_nodes,
        }
        return summary


def build_edge_client_mapping(client_count: int, node_ids: Optional[Iterable[str]] = None) -> Dict[int, str]:
    """Create a deterministic mapping from FL client ids to edge node ids.

    The mapping is intentionally simple and explicit: the first `client_count`
    clients are assigned to the first `client_count` node identifiers if provided.
    If fewer nodes are provided than clients, the remaining clients are left mapped
    to a generated node label. This preserves compatibility with the existing
    client_count-based simulation while still allowing one representative ESP32
    hardware simulation to correspond to multiple edge-node software records.
    """
    if client_count <= 0:
        raise ValueError("client_count must be positive.")

    available_nodes = list(node_ids) if node_ids is not None else []
    mapping: Dict[int, str] = {}
    default_prefix = "Hospital-Edge"

    for client_id in range(client_count):
        if client_id < len(available_nodes):
            mapping[client_id] = str(available_nodes[client_id])
        else:
            mapping[client_id] = f"{default_prefix}-{client_id + 1:02d}"
    return mapping


__all__ = [
    "EdgeTelemetry",
    "EdgeTelemetryRegistry",
    "build_edge_client_mapping",
    "interpret_edge_resource",
    "normalize_telemetry",
    "VALID_STATES",
    "VALID_PARTICIPATION_STATUSES",
]
