from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edge.iot_telemetry import interpret_edge_resource
from edge.wokwi_bridge import WokwiLaptopBridge

st.set_page_config(page_title="FedLite-Care Virtual ESP32 Demo", page_icon="🏥", layout="wide")

st.title("FedLite-Care Virtual ESP32 Demo")
st.caption("SIMULATED EDGE TELEMETRY — laptop-only Wokwi/ESP32 demonstration; not physical hardware validation.")

bridge = WokwiLaptopBridge()
fixture_path = Path("src/edge/wokwi/fixtures/hospital_edge_01.jsonl")

if fixture_path.exists():
    with open(fixture_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                bridge.process_json(line)

state_name = st.selectbox("SIMULATION CONTROL", ["AVAILABLE", "BUSY", "CONSTRAINED", "OFFLINE"], index=0)

if state_name == "AVAILABLE":
    telemetry = {
        "telemetry_version": 1,
        "node_id": "Hospital-Edge-01",
        "state": "AVAILABLE",
        "cpu_load": 42,
        "memory_pressure": 31,
        "network_quality": 82,
        "participation_status": "READY",
        "sequence_id": 101,
        "timestamp_ms": 3000,
        "source": "mcu_simulation",
    }
elif state_name == "BUSY":
    telemetry = {
        "telemetry_version": 1,
        "node_id": "Hospital-Edge-01",
        "state": "BUSY",
        "cpu_load": 68,
        "memory_pressure": 58,
        "network_quality": 71,
        "participation_status": "LIMITED",
        "sequence_id": 102,
        "timestamp_ms": 6000,
        "source": "mcu_simulation",
    }
elif state_name == "CONSTRAINED":
    telemetry = {
        "telemetry_version": 1,
        "node_id": "Hospital-Edge-01",
        "state": "CONSTRAINED",
        "cpu_load": 86,
        "memory_pressure": 81,
        "network_quality": 40,
        "participation_status": "RESTRICTED",
        "sequence_id": 103,
        "timestamp_ms": 9000,
        "source": "mcu_simulation",
    }
else:
    telemetry = {
        "telemetry_version": 1,
        "node_id": "Hospital-Edge-01",
        "state": "OFFLINE",
        "cpu_load": 0,
        "memory_pressure": 0,
        "network_quality": 0,
        "participation_status": "UNAVAILABLE",
        "sequence_id": 104,
        "timestamp_ms": 12000,
        "source": "mcu_simulation",
    }

if st.button("Inject simulated telemetry"):
    bridge.process_json(json.dumps(telemetry))

latest = bridge.registry.get_latest("Hospital-Edge-01")
if latest is not None:
    resource = interpret_edge_resource(latest)
else:
    resource = {
        "node_id": "Hospital-Edge-01",
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

col1, col2 = st.columns(2)
with col1:
    st.subheader("VIRTUAL ESP32")
    st.markdown(f"**Hospital-Edge-01**")
    st.write(f"STATE: {resource['state']}")
    st.write(f"CPU LOAD: {resource['cpu_load']}%")
    st.write(f"MEMORY PRESSURE: {resource['memory_pressure']}%")
    st.write(f"NETWORK QUALITY: {resource['network_quality']}%")
    st.write(f"PARTICIPATION: {resource['participation_status']}")

with col2:
    st.subheader("FEDLITE-CARE CONTROLLER")
    st.write(f"Eligible nodes: {1 if resource['eligible_for_training'] else 0}")
    st.write(f"Participation fraction: 1.0")
    st.write(f"Local epochs: 3")
    st.write(f"Compression ratio: 1.0")
    st.write(f"Resource decision: {resource['state']} / {resource['participation_status']}")

st.subheader("TELEMETRY PIPELINE")
st.markdown("WOKWI ESP32 → TELEMETRY → PYTHON BRIDGE → EDGE TELEMETRY → RESOURCE INTERPRETER → ADAPTIVE CONTROLLER")

st.subheader("FEDERATED SYSTEM")
st.write(f"Round: 1")
st.write(f"Selected clients: {1 if resource['eligible_for_training'] else 0}")
st.write(f"Excluded clients: {1 if not resource['eligible_for_training'] else 0}")
st.write("Aggregation state: local registry updated")
st.write("Global model status: simulated, no new experiment started")

st.caption("SIMULATED EDGE TELEMETRY — do not interpret these CPU/memory/network values as measured hardware results.")
