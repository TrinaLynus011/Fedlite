from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from .iot_telemetry import EdgeTelemetry, EdgeTelemetryRegistry
from .telemetry_sources import TelemetryAdapter, TelemetryPacket

VALID_SOURCES = {"host_software", "mcu_simulation", "mcu_hardware", "gateway"}


class ExternalTelemetryIngest:
    """Small ingest adapter for externally supplied telemetry.

    This component deliberately does not create a fake network connection. It
    accepts JSON records exactly as a real external source would present them
    (stdin, file, or an HTTP body processed by a thin wrapper) and then routes
    them through the existing packet/adapter/registry pipeline.
    """

    def __init__(self, registry: Optional[EdgeTelemetryRegistry] = None, freshness_timeout_sec: float = 5.0):
        self.registry = registry if registry is not None else EdgeTelemetryRegistry()
        self.adapter = TelemetryAdapter(freshness_timeout_sec=freshness_timeout_sec)
        self.freshness_timeout_sec = float(freshness_timeout_sec)
        self._seen_sequences: Dict[str, set[int]] = {}
        self._expected_sequences: Dict[str, int] = {}

    def _get_seen(self, node_id: str) -> set[int]:
        key = str(node_id)
        seen = self._seen_sequences.setdefault(key, set())
        return seen

    def _coerce_payload(self, raw: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(raw, dict):
            payload = dict(raw)
        elif isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON: {exc.msg}") from exc
        else:
            raise TypeError("Telemetry payload must be a JSON string or dictionary.")

        if not isinstance(payload, dict):
            raise TypeError("Telemetry payload must decode to a dictionary.")
        return payload

    def _finalize_packet(self, packet: TelemetryPacket) -> Optional[EdgeTelemetry]:
        node_id = str(packet.node_id)
        seen = self._get_seen(node_id)
        expected = self._expected_sequences.get(node_id)

        normalized = self.adapter.adapt(
            packet,
            now=time.time(),
            expected_sequence=expected,
            seen_sequences=seen,
        )

        if not normalized.valid:
            return None

        self._seen_sequences[node_id].add(int(packet.sequence_id))
        self._expected_sequences[node_id] = int(packet.sequence_id) + 1
        self.registry.update(normalized)
        return normalized

    def ingest_json(self, raw: Union[str, Dict[str, Any]], *, now: Optional[float] = None) -> Optional[EdgeTelemetry]:
        """Parse external JSON and route it through the packet/adaptor registry pipeline."""
        try:
            payload = self._coerce_payload(raw)
        except (TypeError, ValueError):
            return None

        node_id = payload.get("node_id")
        if node_id is None or not str(node_id).strip():
            return None

        source = str(payload.get("source", "mcu_simulation")).lower()
        if source not in VALID_SOURCES:
            return None

        sequence_id = payload.get("sequence_id")
        try:
            sequence_id = int(sequence_id)
        except (TypeError, ValueError):
            return None

        if now is None:
            now_ts = time.time()
        else:
            now_ts = float(now)

        timestamp_ms = payload.get("timestamp_ms")
        if isinstance(timestamp_ms, (int, float)):
            generated_iso = datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            generated = payload.get("timestamp") or payload.get("generated_timestamp")
            if isinstance(generated, str) and generated.strip():
                generated_iso = generated.strip()
            else:
                generated_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        packet = TelemetryPacket(
            node_id=str(node_id),
            sequence_id=sequence_id,
            generated_timestamp=generated_iso,
            source=source,
            payload={
                **payload,
                "timestamp": generated_iso,
                "generated_timestamp": generated_iso,
                "source": source,
            },
        )

        if now is not None:
            return self._finalize_packet_with_time(packet, now)
        return self._finalize_packet(packet)

    def _finalize_packet_with_time(self, packet: TelemetryPacket, now: float) -> Optional[EdgeTelemetry]:
        node_id = str(packet.node_id)
        seen = self._get_seen(node_id)
        expected = self._expected_sequences.get(node_id)

        normalized = self.adapter.adapt(
            packet,
            now=now,
            expected_sequence=expected,
            seen_sequences=seen,
        )

        if not normalized.valid:
            return None

        self._seen_sequences[node_id].add(int(packet.sequence_id))
        self._expected_sequences[node_id] = int(packet.sequence_id) + 1
        self.registry.update(normalized)
        return normalized

    def ingest_file(self, path: str) -> list[EdgeTelemetry]:
        records: list[EdgeTelemetry] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = self.ingest_json(line)
                if record is not None:
                    records.append(record)
        return records


__all__ = ["ExternalTelemetryIngest"]
