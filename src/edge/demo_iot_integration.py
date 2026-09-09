import json
import os
import sys

sys.path.append(os.path.abspath("src"))

from edge.iot_telemetry import EdgeTelemetryRegistry, build_edge_client_mapping, interpret_edge_resource, normalize_telemetry
from federated.adaptive_controller import AdaptiveFederatedController


FIXTURE_DIR = os.path.join("tests", "fixtures", "edge_telemetry")


def load_fixture(name):
    with open(os.path.join(FIXTURE_DIR, name), "r") as fh:
        return json.load(fh)


def main():
    print("=== IoT Edge Integration Demo ===")
    print("Scenario 1: AVAILABLE/READY")
    available = load_fixture("available.json")
    normalized = normalize_telemetry(available)
    interpretation = interpret_edge_resource(normalized)
    registry = EdgeTelemetryRegistry(node_ids=[normalized.node_id])
    registry.update(normalized)
    mapping = build_edge_client_mapping(client_count=3, node_ids=[normalized.node_id, "Hospital-Edge-02", "Hospital-Edge-03"])
    eligible = [
        client_id for client_id, node_id in mapping.items()
        if interpret_edge_resource(registry.get_latest(node_id))["eligible_for_training"]
    ]

    print("Raw telemetry:", available)
    print("Normalized telemetry:", normalized)
    print("Resource interpretation:", interpretation)
    print("Eligible client pool:", eligible)

    print("\nScenario 2: OFFLINE/UNAVAILABLE")
    offline = load_fixture("offline.json")
    registry.update(offline)
    interpretation_offline = interpret_edge_resource(offline)
    eligible_offline = [
        client_id for client_id, node_id in mapping.items()
        if interpret_edge_resource(registry.get_latest(node_id))["eligible_for_training"]
    ]
    print("Raw telemetry:", offline)
    print("Resource interpretation:", interpretation_offline)
    print("Eligible client pool:", eligible_offline)

    controller = AdaptiveFederatedController(config_path="configs/adaptive_controller.yaml")
    controller.enabled = True
    sample_state = {
        "epsilon": 1.0,
        "total_bytes_transmitted": 500000,
        "val_auroc": 0.74,
        "training_time_sec": 12.0,
        "edge_eligible_clients": len(eligible),
        "edge_resource_pressure": interpretation["resource_pressure"],
    }
    controller.observe(sample_state)
    next_config, decision = controller.select_next_configuration(state=sample_state)
    print("\nAdaptive controller decision:")
    print("Selected config:", next_config)
    print("Decision metadata:", decision)


if __name__ == "__main__":
    main()
