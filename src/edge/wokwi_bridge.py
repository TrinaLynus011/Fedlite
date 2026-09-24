from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Union

from .external_ingest import ExternalTelemetryIngest
from .iot_telemetry import EdgeTelemetry, EdgeTelemetryRegistry, interpret_edge_resource


class WokwiLaptopBridge:
    """Laptop-side bridge for the virtual Wokwi ESP32 telemetry stream.

    This component intentionally does not claim a physical network or device
    transport. It is a local, reproducible adapter that accepts JSON telemetry
    emitted by the Wokwi simulation and passes it into the existing FedLite-Care
    telemetry pipeline.
    """

    def __init__(self, registry: Optional[EdgeTelemetryRegistry] = None, freshness_timeout_sec: float = 5.0):
        self.registry = registry if registry is not None else EdgeTelemetryRegistry()
        self.ingest = ExternalTelemetryIngest(registry=self.registry, freshness_timeout_sec=freshness_timeout_sec)

    def process_json(self, raw: Union[str, dict]) -> Optional[EdgeTelemetry]:
        return self.ingest.ingest_json(raw)

    def process_jsonl(self, lines: Iterable[str]) -> List[EdgeTelemetry]:
        accepted: List[EdgeTelemetry] = []
        for line in lines:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            result = self.process_json(item)
            if result is not None:
                accepted.append(result)
        return accepted

    def load_jsonl_file(self, file_path: Union[str, Path]) -> List[EdgeTelemetry]:
        with open(file_path, "r", encoding="utf-8") as handle:
            return self.process_jsonl(handle)

    def current_state(self, node_id: str) -> dict:
        latest = self.registry.get_latest(node_id)
        if latest is None:
            return {
                "node_id": node_id,
                "eligible_for_training": False,
                "state": "OFFLINE",
                "participation_status": "UNAVAILABLE",
                "resource_pressure": 1.0,
            }
        return interpret_edge_resource(latest)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Laptop-side bridge for Wokwi ESP32 JSON telemetry")
    parser.add_argument("path", nargs="?", default=None, help="Optional JSONL telemetry file. Reads stdin if omitted.")
    args = parser.parse_args()

    bridge = WokwiLaptopBridge()
    if args.path:
        records = bridge.load_jsonl_file(args.path)
        print(f"Accepted {len(records)} telemetry records from {args.path}")
        return 0

    for line in sys.stdin:
        record = bridge.process_json(line)
        if record is not None:
            print(json.dumps({
                "node_id": record.node_id,
                "source": record.source,
                "state": record.state,
                "eligible_for_training": interpret_edge_resource(record)["eligible_for_training"],
            }))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
