import json
import os
import sys
import time

import pytest

sys.path.append(os.path.abspath("src"))

from edge.iot_telemetry import (
    EdgeTelemetry,
    EdgeTelemetryRegistry,
    build_edge_client_mapping,
    interpret_edge_resource,
    normalize_telemetry,
)
from edge.external_ingest import ExternalTelemetryIngest
from edge.telemetry_sources import McuTelemetrySource, TelemetryAdapter
from edge.wokwi_bridge import WokwiLaptopBridge


FIXTURE_DIR = os.path.join("tests", "fixtures", "edge_telemetry")


def _read_fixture(name):
    with open(os.path.join(FIXTURE_DIR, name), "r") as fh:
        return json.load(fh)


def test_valid_payload_normalizes_and_validates():
    payload = _read_fixture("available.json")
    telemetry = normalize_telemetry(payload)

    assert isinstance(telemetry, EdgeTelemetry)
    assert telemetry.state == "AVAILABLE"
    assert telemetry.participation_status == "READY"
    assert telemetry.node_id == "Hospital-Edge-01"
    assert 0.0 <= telemetry.cpu_load <= 100.0
    assert 0.0 <= telemetry.memory_pressure <= 100.0
    assert 0.0 <= telemetry.network_quality <= 100.0


def test_missing_required_field_raises():
    payload = _read_fixture("available.json")
    del payload["node_id"]
    with pytest.raises(ValueError, match="node_id"):
        normalize_telemetry(payload)


def test_invalid_state_raises():
    payload = _read_fixture("available.json")
    payload["state"] = "SLEEPING"
    with pytest.raises(ValueError, match="state"):
        normalize_telemetry(payload)


def test_invalid_participation_status_raises():
    payload = _read_fixture("available.json")
    payload["participation_status"] = "UNKNOWN"
    with pytest.raises(ValueError, match="participation_status"):
        normalize_telemetry(payload)


def test_cpu_below_zero_raises():
    payload = _read_fixture("available.json")
    payload["cpu_load"] = -1
    with pytest.raises(ValueError, match="cpu_load"):
        normalize_telemetry(payload)


def test_cpu_above_hundred_raises():
    payload = _read_fixture("available.json")
    payload["cpu_load"] = 101
    with pytest.raises(ValueError, match="cpu_load"):
        normalize_telemetry(payload)


def test_memory_outside_range_raises():
    payload = _read_fixture("available.json")
    payload["memory_pressure"] = 150
    with pytest.raises(ValueError, match="memory_pressure"):
        normalize_telemetry(payload)


def test_network_outside_range_raises():
    payload = _read_fixture("available.json")
    payload["network_quality"] = -5
    with pytest.raises(ValueError, match="network_quality"):
        normalize_telemetry(payload)


def test_resource_interpretation_for_supported_states():
    interpretable_states = [
        ("available.json", True, 0.9),
        ("busy.json", True, 0.7),
        ("constrained.json", True, 0.35),
        ("offline.json", False, 0.0),
    ]

    for fixture_name, expected_eligible, min_weight in interpretable_states:
        payload = _read_fixture(fixture_name)
        interpretation = interpret_edge_resource(payload)
        assert interpretation["eligible_for_training"] is expected_eligible
        assert interpretation["participation_weight"] >= 0.0
        if expected_eligible:
            assert interpretation["participation_weight"] >= min_weight
        else:
            assert interpretation["participation_weight"] == 0.0


def test_registry_tracks_latest_values_and_summary():
    registry = EdgeTelemetryRegistry(node_ids=["Hospital-Edge-01", "Hospital-Edge-02"])
    initial = _read_fixture("available.json")
    updated = _read_fixture("busy.json")
    registry.update(initial)
    registry.update(updated)

    latest = registry.get_latest("Hospital-Edge-01")
    assert latest.state == "AVAILABLE"
    assert registry.get_latest("Hospital-Edge-02").state == "BUSY"

    summary = registry.get_resource_summary()
    assert summary["edge_total_nodes"] == 2
    assert summary["edge_available_nodes"] == 1
    assert summary["edge_busy_nodes"] == 1
    assert summary["edge_mean_cpu_load"] >= 0.0
    assert summary["edge_eligible_clients"] >= 1


def test_client_eligibility_mapping_offline_client_is_excluded():
    registry = EdgeTelemetryRegistry()
    registry.update(_read_fixture("offline.json"))
    mapping = build_edge_client_mapping(client_count=3, node_ids=["Hospital-Edge-01", "Hospital-Edge-02", "Hospital-Edge-03"])
    eligible = [
        client_id
        for client_id, node_id in mapping.items()
        if interpret_edge_resource(registry.get_latest(node_id))["eligible_for_training"]
    ]
    assert 0 not in eligible


def test_client_eligibility_mapping_available_client_is_included():
    registry = EdgeTelemetryRegistry()
    registry.update(_read_fixture("available.json"))
    mapping = build_edge_client_mapping(client_count=3, node_ids=["Hospital-Edge-01", "Hospital-Edge-02", "Hospital-Edge-03"])
    eligible = [
        client_id
        for client_id, node_id in mapping.items()
        if interpret_edge_resource(registry.get_latest(node_id))["eligible_for_training"]
    ]
    assert 0 in eligible


def test_mcu_simulation_payload_is_explicitly_source_aware():
    source = McuTelemetrySource(node_id="Hospital-Edge-01", sequence_start=7)
    payload = source.build_payload(state="CONSTRAINED", cpu_load=48.0, memory_pressure=52.0, network_quality=70.0)

    assert payload["source"] == "mcu_simulation"
    assert payload["node_id"] == "Hospital-Edge-01"
    assert payload["sequence_id"] == 8
    assert payload["state"] == "CONSTRAINED"
    assert payload["participation_status"] in {"READY", "LIMITED", "RESTRICTED", "UNAVAILABLE"}


def test_telemetry_adapter_rejects_duplicate_and_stale_packets():
    adapter = TelemetryAdapter(freshness_timeout_sec=0.1)
    source = McuTelemetrySource(node_id="Hospital-Edge-01", sequence_start=42)
    packet = source.emit(state="AVAILABLE", cpu_load=20.0, memory_pressure=25.0, network_quality=95.0)

    valid = adapter.adapt(packet, now=packet.generated_timestamp and __import__("datetime").datetime.fromisoformat(packet.generated_timestamp.replace("Z", "+00:00")).timestamp(), seen_sequences={packet.sequence_id})
    assert valid.valid is False
    assert valid.rejection_reason == "duplicate_sequence"

    stale_packet = source.emit(state="AVAILABLE", cpu_load=22.0, memory_pressure=26.0, network_quality=96.0, sequence_id=99)
    stale_packet.generated_timestamp = "2020-01-01T00:00:00Z"
    stale_result = adapter.adapt(stale_packet, now=1700000000.0)
    assert stale_result.valid is False
    assert stale_result.rejection_reason == "telemetry_timeout"


def _external_payload(state, sequence_id, *, cpu_load=42, memory_pressure=31, network_quality=82, participation_status=None, source="mcu_simulation"):
    return {
        "telemetry_version": 1,
        "node_id": "Hospital-Edge-01",
        "state": state,
        "cpu_load": cpu_load,
        "memory_pressure": memory_pressure,
        "network_quality": network_quality,
        "participation_status": participation_status or {
            "AVAILABLE": "READY",
            "BUSY": "LIMITED",
            "CONSTRAINED": "RESTRICTED",
            "OFFLINE": "UNAVAILABLE",
        }[state],
        "sequence_id": sequence_id,
        "timestamp_ms": int(time.time() * 1000) + (sequence_id * 1000),
        "source": source,
    }


def test_valid_external_telemetry_json_is_adapted_to_registry():
    ingest = ExternalTelemetryIngest()
    payload = json.dumps(_external_payload("AVAILABLE", 1))

    result = ingest.ingest_json(payload)

    assert result is not None
    assert result.valid is True
    assert result.source == "mcu_simulation"
    assert ingest.registry.get_latest("Hospital-Edge-01").source == "mcu_simulation"


def test_available_external_telemetry_respects_policy():
    ingest = ExternalTelemetryIngest()
    result = ingest.ingest_json(json.dumps(_external_payload("AVAILABLE", 1)))

    interpretation = interpret_edge_resource(result)
    assert interpretation["eligible_for_training"] is True
    assert interpretation["state"] == "AVAILABLE"


def test_busy_external_telemetry_respects_policy():
    ingest = ExternalTelemetryIngest()
    result = ingest.ingest_json(json.dumps(_external_payload("BUSY", 1, cpu_load=70, memory_pressure=60, network_quality=70)))

    interpretation = interpret_edge_resource(result)
    assert interpretation["eligible_for_training"] is True
    assert interpretation["state"] == "BUSY"


def test_constrained_external_telemetry_respects_policy():
    ingest = ExternalTelemetryIngest()
    result = ingest.ingest_json(json.dumps(_external_payload("CONSTRAINED", 1, cpu_load=85, memory_pressure=80, network_quality=35)))

    interpretation = interpret_edge_resource(result)
    assert interpretation["eligible_for_training"] is True
    assert interpretation["state"] == "CONSTRAINED"


def test_offline_external_telemetry_yields_unavailable():
    ingest = ExternalTelemetryIngest()
    result = ingest.ingest_json(json.dumps(_external_payload("OFFLINE", 1, cpu_load=0, memory_pressure=0, network_quality=0, participation_status="UNAVAILABLE")))

    interpretation = interpret_edge_resource(result)
    assert interpretation["eligible_for_training"] is False
    assert interpretation["participation_status"] == "UNAVAILABLE"
    assert interpretation["state"] == "OFFLINE"


def test_external_ingest_rejects_duplicate_sequence_id():
    ingest = ExternalTelemetryIngest()
    first = ingest.ingest_json(json.dumps(_external_payload("AVAILABLE", 10)))
    second = ingest.ingest_json(json.dumps(_external_payload("BUSY", 10, cpu_load=63)))

    assert first.valid is True
    assert second is None or second.valid is False


def test_external_ingest_rejects_out_of_order_sequence_id():
    ingest = ExternalTelemetryIngest()
    ingest.ingest_json(json.dumps(_external_payload("AVAILABLE", 5)))
    stale = ingest.ingest_json(json.dumps(_external_payload("BUSY", 4, cpu_load=60)))

    assert stale is None or stale.valid is False


def test_external_ingest_rejects_stale_telemetry():
    ingest = ExternalTelemetryIngest(freshness_timeout_sec=0.1)
    stale_payload = _external_payload("AVAILABLE", 99)
    stale_payload["timestamp_ms"] = 0
    stale = ingest.ingest_json(json.dumps(stale_payload), now=1700000000.0)

    assert stale is None or stale.valid is False


def test_external_ingest_rejects_malformed_json():
    ingest = ExternalTelemetryIngest()
    result = ingest.ingest_json("{not valid json")
    assert result is None


def test_external_ingest_preserves_source_separation():
    ingest = ExternalTelemetryIngest()
    result = ingest.ingest_json(json.dumps(_external_payload("AVAILABLE", 7, source="mcu_simulation")))
    assert result.source == "mcu_simulation"

    host_ingest = ExternalTelemetryIngest()
    host_result = host_ingest.ingest_json(json.dumps(_external_payload("AVAILABLE", 8, source="host_software")))
    assert host_result.source == "host_software"


def test_server_integration_path_updates_registry_and_state():
    ingest = ExternalTelemetryIngest()
    accepted = ingest.ingest_json(json.dumps(_external_payload("AVAILABLE", 2)))
    registry = ingest.registry

    assert accepted is not None and accepted.valid is True
    assert registry.get_latest("Hospital-Edge-01") is not None
    assert interpret_edge_resource(registry.get_latest("Hospital-Edge-01"))["eligible_for_training"] is True


def test_wokwi_laptop_bridge_routes_payloads_into_registry_and_resource_interpreter():
    bridge = WokwiLaptopBridge()
    result = bridge.process_json(json.dumps(_external_payload("BUSY", 21, cpu_load=70, memory_pressure=62, network_quality=74)))

    assert result is not None
    assert bridge.registry.get_latest("Hospital-Edge-01") is not None
    assert bridge.current_state("Hospital-Edge-01")["state"] == "BUSY"
    assert bridge.current_state("Hospital-Edge-01")["eligible_for_training"] is True
