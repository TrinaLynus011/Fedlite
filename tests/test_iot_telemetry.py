import json
import os
import sys

import pytest

sys.path.append(os.path.abspath("src"))

from edge.iot_telemetry import (
    EdgeTelemetry,
    EdgeTelemetryRegistry,
    build_edge_client_mapping,
    interpret_edge_resource,
    normalize_telemetry,
)


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
