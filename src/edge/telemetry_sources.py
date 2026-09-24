from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from edge.iot_telemetry import EdgeTelemetry, normalize_telemetry


VALID_SOURCES = {"host_software", "mcu_simulation", "mcu_hardware", "gateway"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class TelemetryPacket:
    node_id: str
    sequence_id: int
    generated_timestamp: str
    source: str
    payload: Dict[str, Any]
    received_timestamp: Optional[str] = None
    latency_sec: Optional[float] = None
    stale: bool = False
    valid: bool = True
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "sequence_id": self.sequence_id,
            "generated_timestamp": self.generated_timestamp,
            "source": self.source,
            "payload": self.payload,
            "received_timestamp": self.received_timestamp,
            "latency_sec": self.latency_sec,
            "stale": self.stale,
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
        }


class TelemetrySource:
    """Abstract telemetry source interface."""

    def __init__(self, node_id: str, source: str, freshness_timeout_sec: float = 5.0):
        if source not in VALID_SOURCES:
            raise ValueError(f"Unsupported telemetry source: {source}")
        self.node_id = node_id
        self.source = source
        self.freshness_timeout_sec = float(freshness_timeout_sec)
        self.sequence_id = 0

    def next_sequence_id(self) -> int:
        self.sequence_id += 1
        return self.sequence_id

    def build_payload(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def emit(self, **kwargs: Any) -> TelemetryPacket:
        raise NotImplementedError


class HostTelemetrySource(TelemetrySource):
    """Collect actual host-side software telemetry; this is host measurement, not MCU measurement."""

    def __init__(self, node_id: str = "Hospital-Edge-01", freshness_timeout_sec: float = 5.0):
        super().__init__(node_id=node_id, source="host_software", freshness_timeout_sec=freshness_timeout_sec)
        try:
            import psutil
            self.psutil = psutil
        except ImportError:  # pragma: no cover
            self.psutil = None

    def build_payload(self, **kwargs: Any) -> Dict[str, Any]:
        cpu_percent = 0.0
        memory_percent = 0.0
        if self.psutil is not None:
            cpu_percent = float(self.psutil.cpu_percent(interval=None))
            memory_percent = float(self.psutil.virtual_memory().percent)
        else:
            cpu_percent = float(kwargs.get("cpu_load", 0.0))
            memory_percent = float(kwargs.get("memory_pressure", 0.0))

        return {
            "node_id": self.node_id,
            "state": kwargs.get("state", "AVAILABLE"),
            "cpu_load": min(100.0, max(0.0, cpu_percent)),
            "memory_pressure": min(100.0, max(0.0, memory_percent)),
            "network_quality": float(kwargs.get("network_quality", 100.0)),
            "participation_status": kwargs.get("participation_status", "READY"),
            "timestamp": kwargs.get("timestamp", utc_now_iso()),
            "source": "host_software",
            "sequence_id": int(kwargs.get("sequence_id", self.next_sequence_id())),
            "generated_timestamp": kwargs.get("generated_timestamp", utc_now_iso()),
        }

    def emit(self, **kwargs: Any) -> TelemetryPacket:
        payload = self.build_payload(**kwargs)
        generated = payload.get("generated_timestamp") or payload.get("timestamp") or utc_now_iso()
        return TelemetryPacket(
            node_id=self.node_id,
            sequence_id=int(payload.get("sequence_id", self.next_sequence_id())),
            generated_timestamp=str(generated),
            source="host_software",
            payload=payload,
        )


class McuTelemetrySource(TelemetrySource):
    """Deterministic MCU/Wokwi-style telemetry source.

    Important: these values are MCU-local simulation measurements, not host runtime
    metrics. They represent a distinct source and must remain explicitly labeled.
    """

    def __init__(
        self,
        node_id: str = "Hospital-Edge-01",
        freshness_timeout_sec: float = 5.0,
        sequence_start: int = 1,
        baseline_state: str = "AVAILABLE",
    ):
        super().__init__(node_id=node_id, source="mcu_simulation", freshness_timeout_sec=freshness_timeout_sec)
        self.sequence_id = int(sequence_start)
        self.baseline_state = str(baseline_state).upper()

    def build_payload(self, **kwargs: Any) -> Dict[str, Any]:
        state = str(kwargs.get("state", self.baseline_state)).upper()
        cpu_load = float(kwargs.get("cpu_load", 28.0))
        memory_pressure = float(kwargs.get("memory_pressure", 35.0))
        network_quality = float(kwargs.get("network_quality", 90.0))
        participation_status = str(kwargs.get("participation_status", "READY")).upper()

        payload = {
            "node_id": self.node_id,
            "state": state,
            "cpu_load": min(100.0, max(0.0, cpu_load)),
            "memory_pressure": min(100.0, max(0.0, memory_pressure)),
            "network_quality": min(100.0, max(0.0, network_quality)),
            "participation_status": participation_status,
            "timestamp": kwargs.get("timestamp", utc_now_iso()),
            "source": "mcu_simulation",
            "sequence_id": int(kwargs.get("sequence_id", self.next_sequence_id())),
            "generated_timestamp": kwargs.get("generated_timestamp", utc_now_iso()),
        }
        return payload

    def emit(self, **kwargs: Any) -> TelemetryPacket:
        payload = self.build_payload(**kwargs)
        generated = payload.get("generated_timestamp") or payload.get("timestamp") or utc_now_iso()
        return TelemetryPacket(
            node_id=self.node_id,
            sequence_id=int(payload.get("sequence_id", self.next_sequence_id())),
            generated_timestamp=str(generated),
            source="mcu_simulation",
            payload=payload,
        )


class TelemetryAdapter:
    """Normalize telemetry from multiple sources into the standard EdgeTelemetry dataclass."""

    def __init__(self, freshness_timeout_sec: float = 5.0):
        self.freshness_timeout_sec = float(freshness_timeout_sec)

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[str]:
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _annotate_received(self, packet: TelemetryPacket, now: Optional[float] = None) -> TelemetryPacket:
        now_value = time.time() if now is None else float(now)
        generated_dt = datetime.fromisoformat(packet.generated_timestamp.replace("Z", "+00:00"))
        generated_ts = generated_dt.timestamp()
        packet.received_timestamp = utc_now_iso()
        packet.latency_sec = max(0.0, now_value - generated_ts)
        packet.stale = packet.latency_sec > self.freshness_timeout_sec
        packet.valid = True
        packet.rejection_reason = None
        return packet

    def _validate_packet(self, packet: TelemetryPacket) -> None:
        if packet.node_id is None or not str(packet.node_id).strip():
            packet.valid = False
            packet.rejection_reason = "missing_node_id"
            return
        if packet.source not in VALID_SOURCES:
            packet.valid = False
            packet.rejection_reason = "invalid_source"
            return
        if packet.payload is None or not isinstance(packet.payload, dict):
            packet.valid = False
            packet.rejection_reason = "invalid_payload"
            return

        try:
            normalize_telemetry(packet.payload)
        except Exception as exc:  # pragma: no cover - validation path
            packet.valid = False
            packet.rejection_reason = str(exc)

    def adapt(self, packet: TelemetryPacket, *, now: Optional[float] = None, expected_sequence: Optional[int] = None,
              seen_sequences: Optional[Iterable[int]] = None) -> EdgeTelemetry:
        self._annotate_received(packet, now=now)
        self._validate_packet(packet)

        if packet.stale:
            packet.valid = False
            packet.rejection_reason = "telemetry_timeout"

        if expected_sequence is not None and packet.sequence_id < expected_sequence:
            packet.valid = False
            packet.rejection_reason = "out_of_order"

        seen = set(seen_sequences) if seen_sequences is not None else set()
        if packet.sequence_id in seen:
            packet.valid = False
            packet.rejection_reason = "duplicate_sequence"

        if not packet.valid:
            payload = dict(packet.payload)
            payload["valid"] = False
            payload["stale"] = packet.stale
            payload["rejection_reason"] = packet.rejection_reason
            payload["source"] = packet.source
            return EdgeTelemetry(
                node_id=str(packet.node_id),
                state="OFFLINE",
                cpu_load=0.0,
                memory_pressure=0.0,
                network_quality=0.0,
                participation_status="UNAVAILABLE",
                timestamp=packet.generated_timestamp,
                source=packet.source,
                sequence_id=int(packet.sequence_id),
                generated_timestamp=packet.generated_timestamp,
                received_timestamp=packet.received_timestamp,
                latency_sec=packet.latency_sec,
                stale=packet.stale,
                valid=False,
                rejection_reason=packet.rejection_reason,
            )

        normalized = normalize_telemetry(packet.payload)
        return EdgeTelemetry(
            node_id=normalized.node_id,
            state=normalized.state,
            cpu_load=normalized.cpu_load,
            memory_pressure=normalized.memory_pressure,
            network_quality=normalized.network_quality,
            participation_status=normalized.participation_status,
            timestamp=normalized.timestamp,
            source=normalized.source,
            sequence_id=normalized.sequence_id,
            generated_timestamp=normalized.generated_timestamp,
            received_timestamp=packet.received_timestamp,
            latency_sec=packet.latency_sec,
            stale=packet.stale,
            valid=True,
            rejection_reason=None,
        )

    def from_payload(self, payload: Dict[str, Any], *, now: Optional[float] = None, expected_sequence: Optional[int] = None,
                     seen_sequences: Optional[Iterable[int]] = None) -> EdgeTelemetry:
        packet = TelemetryPacket(
            node_id=str(payload.get("node_id", "")),
            sequence_id=int(payload.get("sequence_id", 0)),
            generated_timestamp=str(payload.get("generated_timestamp") or payload.get("timestamp") or utc_now_iso()),
            source=str(payload.get("source", "host_software")).lower(),
            payload=dict(payload),
        )
        return self.adapt(packet, now=now, expected_sequence=expected_sequence, seen_sequences=seen_sequences)


__all__ = [
    "TelemetryPacket",
    "TelemetrySource",
    "HostTelemetrySource",
    "McuTelemetrySource",
    "TelemetryAdapter",
    "VALID_SOURCES",
]
