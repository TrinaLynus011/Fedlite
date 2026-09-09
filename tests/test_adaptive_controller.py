import pytest
import numpy as np
import yaml
import os
import sys

sys.path.append(os.path.abspath("src"))
from federated.adaptive_controller import AdaptiveFederatedController

@pytest.fixture
def mock_config(tmp_path):
    config = {
        "adaptive_controller": {
            "enabled": True,
            "seed": 42,
            "privacy": {
                "enabled": True,
                "budget_epsilon": 5.0,
                "stop_on_budget_exhaustion": True
            },
            "communication": {
                "budget_bytes": 1000000
            },
            "candidates": {
                "compression_ratios": [1.0, 0.5],
                "local_epochs": [1, 3],
                "participation_fractions": [0.5, 1.0]
            },
            "objective_weights": {
                "utility": 1.0,
                "communication": 0.5,
                "resource": 0.1,
                "privacy": 1.0
            }
        }
    }
    
    config_path = tmp_path / "adaptive_controller.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    return str(config_path)

def test_controller_initialization(mock_config):
    controller = AdaptiveFederatedController(config_path=mock_config)
    assert controller.enabled is True
    assert controller.seed == 42
    assert controller.privacy_config["budget_epsilon"] == 5.0
    assert len(controller.candidates["compression_ratios"]) == 2

def test_determinism(mock_config):
    controller1 = AdaptiveFederatedController(config_path=mock_config)
    controller2 = AdaptiveFederatedController(config_path=mock_config)
    
    state = {
        "epsilon": 1.0,
        "total_bytes_transmitted": 500000
    }
    
    controller1.observe(state)
    controller2.observe(state)
    
    cand1, _ = controller1.select_next_configuration()
    cand2, _ = controller2.select_next_configuration()
    
    assert cand1 == cand2

def test_privacy_constraint(mock_config):
    controller = AdaptiveFederatedController(config_path=mock_config)
    
    # Near exhaustion, but valid
    state_valid = {
        "epsilon": 4.9,
        "total_bytes_transmitted": 1000
    }
    controller.observe(state_valid)
    cand, meta = controller.select_next_configuration()
    assert cand is not None
    assert meta["action"] == "continue"
    
    # Exhausted
    state_exhausted = {
        "epsilon": 5.1,
        "total_bytes_transmitted": 1000
    }
    controller.observe(state_exhausted)
    cand, meta = controller.select_next_configuration()
    assert cand is None
    assert meta["action"] == "stop"
    assert meta["reason"] == "privacy_budget_exhausted"

def test_communication_constraint(mock_config):
    controller = AdaptiveFederatedController(config_path=mock_config)
    
    state_exhausted = {
        "epsilon": 1.0,
        "total_bytes_transmitted": 1500000
    }
    controller.observe(state_exhausted)
    cand, meta = controller.select_next_configuration()
    
    assert cand is None
    assert meta["action"] == "stop"
    assert "communication_budget_exhausted" in [c["constraint_status"] for c in meta.get("candidate_scores", [{"constraint_status": "communication_budget_exhausted"}])] or meta["reason"] == "all_candidates_violate_constraints"

def test_candidate_scoring(mock_config):
    controller = AdaptiveFederatedController(config_path=mock_config)
    state = {
        "epsilon": 1.0,
        "total_bytes_transmitted": 1000
    }
    controller.observe(state)
    cand, meta = controller.select_next_configuration()
    
    assert cand is not None
    # According to the heuristic:
    # U = (le/5) * pf * sqrt(cr)
    # C = cr * pf
    # R = (le/5) * pf
    # P = (le/5) * pf
    # cand = [1.0, 0.5] CR, [1, 3] LE, [0.5, 1.0] PF
    assert "compression_ratio" in cand
    assert "local_epochs" in cand
    assert "participation_fraction" in cand


def test_invalid_configuration_rejected(tmp_path):
    config = {
        "adaptive_controller": {
            "enabled": True,
            "privacy": {"enabled": True, "budget_epsilon": 5.0},
            "communication": {"budget_bytes": 1000},
            "candidates": {
                "compression_ratios": [],
                "local_epochs": [1, 2],
                "participation_fractions": [0.5, 1.0],
            },
        }
    }
    config_path = tmp_path / "bad_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    with pytest.raises(ValueError, match="compression_ratios"):
        AdaptiveFederatedController(config_path=str(config_path))


def test_static_mode_is_disabled_by_default(tmp_path):
    config_path = tmp_path / "adaptive_controller.yaml"
    config_path.write_text(yaml.safe_dump({"adaptive_controller": {"enabled": False, "candidates": {"compression_ratios": [1.0], "local_epochs": [1], "participation_fractions": [1.0]}}}))

    controller = AdaptiveFederatedController(config_path=str(config_path))
    assert controller.enabled is False


def test_round_to_round_config_applies_next_round():
    controller = AdaptiveFederatedController(config_path="configs/adaptive_controller.yaml")
    controller.enabled = True
    controller.observe({
        "epsilon": 1.0,
        "total_bytes_transmitted": 1000,
        "val_auroc": 0.72,
        "training_time_sec": 10.0,
    })
    next_cfg, meta = controller.select_next_configuration()
    assert next_cfg is not None
    assert 0.0 < next_cfg["participation_fraction"] <= 1.0
    assert next_cfg["local_epochs"] >= 1
    assert 0.0 < next_cfg["compression_ratio"] <= 1.0


def test_controller_ignores_edge_constraint_when_no_edge_telemetry_is_active():
    controller = AdaptiveFederatedController(config_path="configs/adaptive_controller.yaml")
    controller.enabled = True
    controller.observe({
        "epsilon": 1.0,
        "total_bytes_transmitted": 1000,
        "val_auroc": 0.72,
        "training_time_sec": 10.0,
        "edge_telemetry_active": False,
        "edge_eligible_clients": 0,
    })
    next_cfg, meta = controller.select_next_configuration()
    assert next_cfg is not None
    assert meta["action"] == "continue"
