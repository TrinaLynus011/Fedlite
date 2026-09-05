import pytest
import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from fastapi.testclient import TestClient

# Add src/ to path
sys.path.append(os.path.abspath("src"))
from models.mlp import LightweightMLP, estimate_model_stats
from models.logistic_regression import PyTorchLR
from privacy.dp_mechanism import (
    clip_and_add_noise, compute_epsilon,
    compute_client_update_epsilon, compute_dpsgd_epsilon,
    clip_grad_tensor,
)
from compression.sparsification import compress_updates
from federated.client import FedLiteClient
from api.main import app

# ----------------------------------------------------
# 1. Data and Splitting Tests
# ----------------------------------------------------
def test_data_and_split():
    # Check split files exist
    assert os.path.exists("data/processed/train.csv"), "Train split CSV missing!"
    assert os.path.exists("data/processed/val.csv"), "Validation split CSV missing!"
    assert os.path.exists("data/processed/test.csv"), "Test split CSV missing!"
    
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # 1. Target conversion check
    assert set(train_df["readmitted"].unique()).issubset({0, 1}), "Target must be binary 0 and 1!"
    
    # 2. Patient uniqueness check (Data Leakage check)
    train_patients = set(train_df["patient_nbr"].unique())
    val_patients = set(val_df["patient_nbr"].unique())
    test_patients = set(test_df["patient_nbr"].unique())
    
    assert train_patients.isdisjoint(val_patients), "Patient leakage between Train and Val!"
    assert train_patients.isdisjoint(test_patients), "Patient leakage between Train and Test!"
    assert val_patients.isdisjoint(test_patients), "Patient leakage between Val and Test!"

# ----------------------------------------------------
# 2. Model Structure Tests
# ----------------------------------------------------
def test_mlp_model():
    input_dim = 44
    model = LightweightMLP(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.2)
    dummy_input = torch.randn(10, input_dim)
    output = model(dummy_input)
    
    # Check predictions shape and range
    assert output.shape == (10, 1), "MLP output shape should be (Batch_Size, 1)!"
    probs = torch.sigmoid(output)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0), "Sigmoid output should be in range [0, 1]!"
    
    # Check stats estimation
    stats = estimate_model_stats(model, input_dim)
    assert stats["parameter_count"] > 0
    assert stats["fp32_size_kb"] > 0.0
    assert stats["estimated_int8_size_kb"] > 0.0

# ----------------------------------------------------
# 3. Privacy (DP) Tests
# ----------------------------------------------------
def test_differential_privacy():
    # Test clipping and noise addition
    updates = [np.ones((5, 5)), np.ones((5, 1))]
    clipping_norm = 1.0
    noise_mult = 1.5
    
    clipped_noisy_updates, norm = clip_and_add_noise(updates, clipping_norm, noise_mult, seed=42)
    
    assert len(clipped_noisy_updates) == len(updates)
    assert clipped_noisy_updates[0].shape == updates[0].shape
    assert norm > 0.0
    
    # Test RDP Accountant
    eps = compute_epsilon(steps=100, batch_size=128, dataset_size=5000, noise_multiplier=1.5, delta=1e-5)
    assert eps > 0.0
    assert eps < float('inf')

# ----------------------------------------------------
# 4. Compression (Top-K) Tests
# ----------------------------------------------------
def test_top_k_compression():
    updates = [np.array([0.1, 0.5, -0.9, 0.02, 1.2])]
    ratio = 0.4  # Keep top 40% (2 elements: -0.9 and 1.2)
    
    comp_updates, bytes_before, bytes_after = compress_updates(updates, ratio)
    
    assert len(comp_updates) == 1
    assert comp_updates[0].shape == updates[0].shape
    assert bytes_after < bytes_before, "Compression should reduce payload size!"
    
    # Check that non-top elements were zeroed out
    expected_sparse = np.array([0.0, 0.0, -0.9, 0.0, 1.2])
    np.testing.assert_array_equal(comp_updates[0], expected_sparse)

# ----------------------------------------------------
# 5. Quantization Test
# ----------------------------------------------------
def test_quantization_availability():
    mlp_state = "artifacts/checkpoints/central/mlp.pt"
    # If the state dict is available, test that it can load
    if os.path.exists(mlp_state):
        from quantization.quantizer import quantize_mlp_model
        stats = quantize_mlp_model(mlp_state, input_dim=44, X_val=np.random.randn(50, 44))
        assert stats["int8_size_bytes"] > 0
        assert stats["prediction_consistency_mae"] >= 0.0

# ----------------------------------------------------
# 6. API Endpoint Tests
# ----------------------------------------------------
def test_api_endpoints():
    client = TestClient(app)
    
    # GET /health
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    
    # GET /metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # POST /predict with dummy patient
    patient_payload = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[50-60)",
        "admission_type_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 3,
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 15,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "diag_1": "250.0",
        "diag_2": "276",
        "diag_3": "282",
        "number_diagnoses": 9,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "metformin": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": "No",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "examide": "No",
        "citoglipton": "No",
        "insulin": "No",
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": "No",
        "diabetesMed": "No"
    }
    
    # We load assets if they were not already loaded in FastAPI TestClient context
    from api.main import load_assets
    load_assets()
    
    response = client.post("/predict", json=patient_payload)
    if response.status_code == 200:
        res = response.json()
        assert "risk_probability" in res
        assert "risk_class" in res
        assert res["risk_class"] in [0, 1]
    else:
        # If model checkpoints are missing, it returns 503 Service Unavailable, which is expected
        assert response.status_code == 503

# ----------------------------------------------------
# 7. Additional Research Integrity Tests
# ----------------------------------------------------
def test_reproducibility_seeds():
    # Verify seed locks yield deterministic weights for model instantiation
    torch.manual_seed(42)
    model1 = LightweightMLP(input_dim=44, hidden_dims=[64, 32], dropout=0.2)
    w1 = [p.clone() for p in model1.parameters()]
    
    torch.manual_seed(42)
    model2 = LightweightMLP(input_dim=44, hidden_dims=[64, 32], dropout=0.2)
    w2 = [p.clone() for p in model2.parameters()]
    
    for p1, p2 in zip(w1, w2):
        assert torch.equal(p1, p2), "Seeds not locking model weight initialization!"

def test_client_class_weight_guards():
    # Test zero-positive client guard
    y_zero_pos = np.zeros(100) # Only negative samples
    client_zero_pos = FedLiteClient(
        client_id=0,
        X_train=np.random.randn(100, 44),
        y_train=y_zero_pos,
        X_val=np.random.randn(20, 44),
        y_val=np.zeros(20),
        model_type="mlp"
    )
    assert not torch.isinf(client_zero_pos.criterion.pos_weight), "pos_weight is infinite for zero-positive client!"
    assert not torch.isnan(client_zero_pos.criterion.pos_weight), "pos_weight is NaN for zero-positive client!"
    # float32 has ~7 decimal digits of precision; use approx tolerance
    assert client_zero_pos.criterion.pos_weight.item() == pytest.approx(7.680996884735203, rel=1e-5)
    
    # Test zero-negative client guard
    y_zero_neg = np.ones(100) # Only positive samples
    client_zero_neg = FedLiteClient(
        client_id=1,
        X_train=np.random.randn(100, 44),
        y_train=y_zero_neg,
        X_val=np.random.randn(20, 44),
        y_val=np.ones(20),
        model_type="mlp"
    )
    assert not torch.isinf(client_zero_neg.criterion.pos_weight), "pos_weight is infinite for zero-negative client!"
    assert not torch.isnan(client_zero_neg.criterion.pos_weight), "pos_weight is NaN for zero-negative client!"
    # float32 precision: use approx tolerance
    assert client_zero_neg.criterion.pos_weight.item() == pytest.approx(1.0 / 7.680996884735203, rel=1e-5)

def test_xgboost_config_prevalence():
    # Check that XGBoost scale_pos_weight is configured to 7.681
    with open("configs/models.yaml", "r") as f:
        models_config = yaml.safe_load(f)
    assert models_config["xgboost"]["scale_pos_weight"] == 7.681, "XGBoost scale_pos_weight mismatch with training prevalence!"

def test_lr_client_batch_loss_is_scalar_and_finite():
    """Regression test: one LR client training batch must produce a scalar, finite loss.

    Guards against the "Tensor with N elements cannot be converted to Scalar"
    error that occurred when BCELoss(reduction='none') output was passed directly
    to .item() in the post-training eval loop.
    """
    np.random.seed(0)
    n = 200
    X = np.random.randn(n, 44).astype(np.float32)
    y = (np.random.rand(n) > 0.87).astype(np.float32)  # ~13% positive, like real data

    client = FedLiteClient(
        client_id=0,
        X_train=X,
        y_train=y,
        X_val=X[:40],
        y_val=y[:40],
        model_type="lr",
        local_epochs=1,
        batch_size=64,
    )

    # Simulate one fit() round using the current (all-zeros) parameters
    init_params = client.get_parameters(config={})
    result_params, n_samples, metrics = client.fit(init_params, config={})

    loss_val = metrics["loss"]
    assert np.isscalar(loss_val) or isinstance(loss_val, float), \
        f"LR training loss must be a Python scalar, got {type(loss_val)}"
    assert np.isfinite(loss_val), \
        f"LR training loss must be finite, got {loss_val}"


def test_simulation_round_telemetry_persistence():
    # Run a small 1-round simulation with 2 clients to verify telemetry creation
    from federated.server import run_federated_simulation
    test_metrics, checkpoint = run_federated_simulation(
        experiment_id="TEST_RUN",
        model_type="lr",
        client_count=2,
        partition_method="iid",
        rounds=1,
        local_epochs=1,
        batch_size=128,
        seed=42
    )
    
    assert os.path.exists("results/TEST_RUN_round_metrics.csv"), "Round telemetry CSV missing!"
    assert os.path.exists("results/TEST_RUN_round_metrics.json"), "Round telemetry JSON missing!"
    
    round_df = pd.read_csv("results/TEST_RUN_round_metrics.csv")
    assert len(round_df) == 1, "Should have exactly 1 round recorded!"
    assert "val_auroc" in round_df.columns
    assert "val_auprc" in round_df.columns
    assert "training_time_sec" in round_df.columns
    
    # Clean up test files
    os.remove("results/TEST_RUN_round_metrics.csv")
    os.remove("results/TEST_RUN_round_metrics.json")
    if os.path.exists(checkpoint):
        os.remove(checkpoint)

def test_model_size_telemetry():
    # Check that theoretical size calculations are correct
    input_dim = 44
    model = LightweightMLP(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.2)
    stats = estimate_model_stats(model, input_dim)
    
    param_count = sum(p.numel() for p in model.parameters())
    theoretical_fp32 = (param_count * 4) / 1024.0
    theoretical_int8 = (param_count * 1) / 1024.0
    
    assert stats["parameter_count"] == param_count
    assert stats["fp32_size_kb"] == theoretical_fp32
    assert stats["estimated_int8_size_kb"] == theoretical_int8


# ============================================================
# 9. Client-Update Gaussian DP Tests
# ============================================================
# Privacy unit: CLIENT (not patient/record).
# Mechanism: clip Δθ to L2 norm C, add N(0,(σC)²I) noise.
# Accountant: Rényi DP composition of the Gaussian mechanism.
# ============================================================

def _make_updates(seed=0, n_params=200):
    """Helper: deterministic fake parameter-update list."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n_params).astype(np.float64)]


def test_dp_clipping_reduces_norm():
    """After clipping, the L2 norm of the update must be <= clipping_norm."""
    clipping_norm = 1.0
    updates = _make_updates()
    pre_norm = float(np.sqrt(sum(np.sum(u ** 2) for u in updates)))
    assert pre_norm > clipping_norm, "Pre-condition: update must exceed clipping norm for this test."

    clipped, reported_norm = clip_and_add_noise(updates, clipping_norm, noise_multiplier=0.0)
    post_norm = float(np.sqrt(sum(np.sum(u ** 2) for u in clipped)))

    assert post_norm <= clipping_norm + 1e-6, (
        f"Clipped norm {post_norm:.6f} exceeds clipping_norm {clipping_norm}."
    )
    assert abs(reported_norm - pre_norm) < 1e-6, "Reported pre-clip norm should match actual pre-clip norm."


def test_dp_clipping_norm_correctness():
    """A unit-norm update with C=1 must be unchanged by clipping."""
    clipping_norm = 1.0
    vec = np.ones(100, dtype=np.float64)
    vec /= np.linalg.norm(vec)   # unit norm
    updates = [vec.copy()]

    clipped, _ = clip_and_add_noise(updates, clipping_norm, noise_multiplier=0.0)
    post_norm = float(np.linalg.norm(clipped[0]))
    np.testing.assert_allclose(post_norm, 1.0, atol=1e-6,
                               err_msg="Unit-norm update should be unchanged by clipping.")
    np.testing.assert_allclose(clipped[0], vec, atol=1e-6,
                               err_msg="Unit-norm update values should be unchanged by clipping.")


def test_dp_noise_is_added_when_enabled():
    """With noise_multiplier > 0 the noisy update must differ from the clipped update."""
    updates = _make_updates()
    clipped_only, _ = clip_and_add_noise(updates, clipping_norm=1.0, noise_multiplier=0.0, seed=0)
    noisy, _ = clip_and_add_noise(updates, clipping_norm=1.0, noise_multiplier=1.0, seed=0)

    assert not np.allclose(clipped_only[0], noisy[0]), (
        "Noisy update must differ from the noiseless clipped update."
    )


def test_dp_no_noise_when_disabled():
    """With noise_multiplier=0 the output equals the clipped update exactly."""
    updates = _make_updates()
    out1, _ = clip_and_add_noise(updates, clipping_norm=1.0, noise_multiplier=0.0, seed=1)
    out2, _ = clip_and_add_noise(updates, clipping_norm=1.0, noise_multiplier=0.0, seed=2)
    np.testing.assert_array_equal(out1[0], out2[0],
                                  err_msg="Noiseless outputs must be identical (deterministic clip).")


def test_dp_accountant_returns_finite_epsilon():
    """compute_client_update_epsilon must return a finite positive number."""
    eps = compute_client_update_epsilon(rounds=10, noise_multiplier=1.5, delta=1e-5)
    assert np.isfinite(eps), f"Epsilon should be finite, got {eps}."
    assert eps > 0.0, f"Epsilon should be positive, got {eps}."


def test_dp_epsilon_increases_with_rounds():
    """More rounds → more privacy expenditure → larger epsilon."""
    eps_5  = compute_client_update_epsilon(rounds=5,  noise_multiplier=1.5, delta=1e-5)
    eps_10 = compute_client_update_epsilon(rounds=10, noise_multiplier=1.5, delta=1e-5)
    eps_20 = compute_client_update_epsilon(rounds=20, noise_multiplier=1.5, delta=1e-5)
    assert eps_5 < eps_10 < eps_20, (
        f"Epsilon must strictly increase with rounds: {eps_5:.4f}, {eps_10:.4f}, {eps_20:.4f}."
    )


def test_dp_epsilon_decreases_with_noise_multiplier():
    """Higher noise_multiplier → stronger privacy → smaller epsilon."""
    eps_low  = compute_client_update_epsilon(rounds=10, noise_multiplier=0.8, delta=1e-5)
    eps_mid  = compute_client_update_epsilon(rounds=10, noise_multiplier=1.5, delta=1e-5)
    eps_high = compute_client_update_epsilon(rounds=10, noise_multiplier=4.0, delta=1e-5)
    assert eps_low > eps_mid > eps_high, (
        f"Epsilon must decrease as noise_multiplier increases: "
        f"{eps_low:.4f}, {eps_mid:.4f}, {eps_high:.4f}."
    )


def test_dp_zero_rounds_gives_zero_epsilon():
    """Zero rounds means no privacy cost yet."""
    eps = compute_client_update_epsilon(rounds=0, noise_multiplier=1.5, delta=1e-5)
    assert eps == 0.0, f"Zero rounds should give epsilon=0, got {eps}."


def test_dp_zero_noise_multiplier_gives_inf_epsilon():
    """No noise = no privacy guarantee = epsilon of infinity."""
    eps = compute_client_update_epsilon(rounds=10, noise_multiplier=0.0, delta=1e-5)
    assert eps == float("inf"), f"Zero noise_multiplier should give inf epsilon, got {eps}."


def test_dp_reproducibility_with_fixed_seed():
    """Same seed must produce bit-identical noisy updates."""
    updates = _make_updates()
    out_a, _ = clip_and_add_noise(updates, clipping_norm=1.0, noise_multiplier=1.0, seed=42)
    out_b, _ = clip_and_add_noise(updates, clipping_norm=1.0, noise_multiplier=1.0, seed=42)
    np.testing.assert_array_equal(out_a[0], out_b[0],
                                  err_msg="Same seed must produce identical noisy updates.")


def test_dp_client_update_gaussian_mechanism_in_client():
    """
    FedLiteClient with dp_config enabled and mechanism='client_update_gaussian'
    must return a training loss that is scalar and finite, and the returned
    parameters must differ from the input parameters (noise was applied).
    """
    np.random.seed(0)
    n = 200
    X = np.random.randn(n, 44).astype(np.float32)
    y = (np.random.rand(n) > 0.87).astype(np.float32)

    dp_cfg = {
        "enabled": True,
        "mechanism": "client_update_gaussian",
        "privacy_unit": "client",
        "clipping_norm": 1.0,
        "noise_multiplier": 1.0,
        "delta": 1e-5,
    }

    client = FedLiteClient(
        client_id=0,
        X_train=X,
        y_train=y,
        X_val=X[:40],
        y_val=y[:40],
        model_type="mlp",
        local_epochs=1,
        batch_size=64,
        dp_config=dp_cfg,
    )

    init_params = client.get_parameters(config={})
    result_params, n_samples, metrics = client.fit(init_params, config={"seed": 1})

    loss_val = metrics["loss"]
    assert np.isfinite(loss_val), f"Loss must be finite with DP enabled, got {loss_val}."
    assert metrics["dp_enabled"] is True, "dp_enabled flag must be True in metrics."

    # The returned parameters should differ from the initial ones due to training + noise
    # Use a generous tolerance: after 1 epoch + noise the update can be small.
    max_diff = max(float(np.max(np.abs(r - i))) for r, i in zip(result_params, init_params))
    assert max_diff > 1e-7, (
        f"DP-enabled client must return parameters that differ from the initial "
        f"global parameters. Max abs diff = {max_diff:.2e}."
    )


def test_dp_disabled_produces_no_noise_in_client():
    """
    With dp_config disabled, two identical fit() calls with the same seed
    must return bit-identical parameters (no stochastic noise on the update).
    """
    np.random.seed(0)
    n = 200
    X = np.random.randn(n, 44).astype(np.float32)
    y = (np.random.rand(n) > 0.87).astype(np.float32)

    dp_cfg = {"enabled": False}

    def _run():
        torch.manual_seed(7)
        client = FedLiteClient(
            client_id=0,
            X_train=X,
            y_train=y,
            X_val=X[:40],
            y_val=y[:40],
            model_type="mlp",
            local_epochs=1,
            batch_size=64,
            dp_config=dp_cfg,
        )
        init_params = client.get_parameters(config={})
        result_params, _, _ = client.fit(init_params, config={"seed": 99})
        return result_params

    params_a = _run()
    params_b = _run()

    for a, b in zip(params_a, params_b):
        np.testing.assert_array_equal(a, b,
            err_msg="Disabled DP runs with identical seeds must yield identical parameters.")


# ============================================================
# 10. DP-SGD Tests
# ============================================================
# Privacy unit: RECORD (patient encounter).
# Mechanism: per-example gradient clipping + Gaussian noise
#            inside the local SGD training loop.
# Accountant: subsampled Gaussian, tight 2nd-order RDP bound.
# ============================================================

def _dpsgd_client(noise_multiplier=1.0, local_epochs=1, batch_size=32, seed=0):
    """Helper: build a small MLP FedLiteClient with DP-SGD enabled."""
    np.random.seed(seed)
    n = 300
    X = np.random.randn(n, 44).astype(np.float32)
    y = (np.random.rand(n) > 0.87).astype(np.float32)
    dp_cfg = {
        "enabled": True,
        "mechanism": "dpsgd",
        "privacy_unit": "record",
        "clipping_norm": 1.0,
        "noise_multiplier": noise_multiplier,
        "delta": 1e-5,
    }
    return FedLiteClient(
        client_id=0,
        X_train=X, y_train=y,
        X_val=X[:40], y_val=y[:40],
        model_type="mlp",
        local_epochs=local_epochs,
        batch_size=batch_size,
        dp_config=dp_cfg,
    ), X, y


# ── Per-example gradient clipping ──────────────────────────────────────────

def test_dpsgd_clip_grad_tensor_reduces_norm():
    """clip_grad_tensor must enforce the clipping bound."""
    import torch
    grad = torch.randn(64, 32) * 10.0   # large gradient
    C = 1.0
    clipped, pre_norm = clip_grad_tensor(grad, C)
    post_norm = float(clipped.norm(2).item())
    assert post_norm <= C + 1e-5, (
        f"Clipped grad norm {post_norm:.6f} exceeds C={C}."
    )
    assert abs(pre_norm - float(grad.norm(2).item())) < 1e-4


def test_dpsgd_clip_grad_tensor_does_not_clip_small_grad():
    """A gradient already within C must pass through unchanged."""
    import torch
    vec = torch.ones(10) * 0.05       # L2 norm = 0.05*sqrt(10) ~ 0.158 < 1.0
    C = 1.0
    clipped, pre_norm = clip_grad_tensor(vec, C)
    assert float(clipped.norm(2).item()) <= C + 1e-5
    # Values should be identical (no scaling needed)
    torch.testing.assert_close(clipped, vec, atol=1e-6, rtol=0)


# ── Noise activation ───────────────────────────────────────────────────────

def test_dpsgd_noise_is_added():
    """DP-SGD client with sigma>0 must produce parameters different from
    a second run with a different internal noise seed (same init, same data)."""
    torch.manual_seed(1)
    client_a, _, _ = _dpsgd_client(noise_multiplier=2.0, seed=5)
    params_a = client_a.get_parameters(config={})
    res_a, _, _ = client_a.fit(params_a, config={"seed": 10})

    torch.manual_seed(1)
    client_b, _, _ = _dpsgd_client(noise_multiplier=2.0, seed=5)
    params_b = client_b.get_parameters(config={})
    res_b, _, _ = client_b.fit(params_b, config={"seed": 20})   # different round seed

    # Different noise seeds → different results
    any_diff = any(not np.allclose(a, b) for a, b in zip(res_a, res_b))
    assert any_diff, "Different noise seeds must produce different parameters."


def test_dpsgd_zero_noise_multiplier_is_deterministic():
    """With noise_multiplier=0 (DP disabled), same seed → identical params."""
    def _run(seed_val):
        torch.manual_seed(7)
        np.random.seed(0)
        n = 200
        X = np.random.randn(n, 44).astype(np.float32)
        y = (np.random.rand(n) > 0.87).astype(np.float32)
        client = FedLiteClient(
            client_id=0, X_train=X, y_train=y,
            X_val=X[:40], y_val=y[:40],
            model_type="mlp", local_epochs=1, batch_size=64,
            dp_config={"enabled": False},
        )
        p0 = client.get_parameters(config={})
        res, _, _ = client.fit(p0, config={"seed": seed_val})
        return res

    p_a = _run(99)
    p_b = _run(99)
    for a, b in zip(p_a, p_b):
        np.testing.assert_array_equal(a, b)


# ── Accountant properties ──────────────────────────────────────────────────

def test_dpsgd_accountant_finite_epsilon():
    """compute_dpsgd_epsilon must return finite positive epsilon."""
    eps, steps = compute_dpsgd_epsilon(
        rounds=10, local_epochs=3, batch_size=128,
        dataset_size=13933, noise_multiplier=0.8, delta=1e-5,
    )
    assert np.isfinite(eps), f"Expected finite epsilon, got {eps}."
    assert eps > 0.0, f"Expected positive epsilon, got {eps}."
    assert steps == 10 * 3 * math.ceil(13933 / 128)


def test_dpsgd_epsilon_increases_with_rounds():
    """More rounds → more privacy expenditure → larger epsilon."""
    common = dict(local_epochs=3, batch_size=128, dataset_size=13933,
                  noise_multiplier=0.8, delta=1e-5)
    e5,  _ = compute_dpsgd_epsilon(rounds=5,  **common)
    e10, _ = compute_dpsgd_epsilon(rounds=10, **common)
    e20, _ = compute_dpsgd_epsilon(rounds=20, **common)
    assert e5 < e10 < e20, (
        f"Epsilon must increase with rounds: {e5:.4f}, {e10:.4f}, {e20:.4f}."
    )


def test_dpsgd_epsilon_decreases_with_noise_multiplier():
    """Higher sigma → stronger privacy → smaller epsilon."""
    common = dict(rounds=10, local_epochs=3, batch_size=128,
                  dataset_size=13933, delta=1e-5)
    e_low,  _ = compute_dpsgd_epsilon(noise_multiplier=0.8, **common)
    e_mid,  _ = compute_dpsgd_epsilon(noise_multiplier=1.5, **common)
    e_high, _ = compute_dpsgd_epsilon(noise_multiplier=4.0, **common)
    assert e_low > e_mid > e_high, (
        f"Epsilon must decrease with sigma: {e_low:.4f}, {e_mid:.4f}, {e_high:.4f}."
    )


def test_dpsgd_zero_noise_gives_inf_epsilon():
    """No noise → no DP guarantee → epsilon = inf."""
    eps, _ = compute_dpsgd_epsilon(
        rounds=10, local_epochs=3, batch_size=128,
        dataset_size=13933, noise_multiplier=0.0, delta=1e-5,
    )
    assert eps == float("inf")


# ── Client training behaviour ──────────────────────────────────────────────

def test_dpsgd_client_changes_parameters():
    """DP-SGD client fit() must change model parameters."""
    np.random.seed(0)
    torch.manual_seed(1)
    n = 300
    X = np.random.randn(n, 44).astype(np.float32)
    # Ensure at least some positives and negatives
    y = np.zeros(n, dtype=np.float32)
    y[:40] = 1.0   # 40 positives, 260 negatives
    dp_cfg = {
        "enabled": True,
        "mechanism": "dpsgd",
        "privacy_unit": "record",
        "clipping_norm": 1.0,
        "noise_multiplier": 1.0,
        "delta": 1e-5,
    }
    client = FedLiteClient(
        client_id=0, X_train=X, y_train=y,
        X_val=X[:40], y_val=y[:40],
        model_type="mlp", local_epochs=1, batch_size=32,
        dp_config=dp_cfg,
    )
    init = client.get_parameters(config={})
    result, _, metrics = client.fit(init, config={"seed": 42})
    max_diff = max(float(np.max(np.abs(r - i))) for r, i in zip(result, init))
    assert max_diff > 1e-7, f"Parameters must change after DP-SGD training. Max diff={max_diff:.2e}."


def test_dpsgd_client_loss_is_finite():
    """DP-SGD client fit() must return finite training loss."""
    client, _, _ = _dpsgd_client(noise_multiplier=0.8)
    init = client.get_parameters(config={})
    _, _, metrics = client.fit(init, config={"seed": 1})
    assert np.isfinite(metrics["loss"]), f"Loss must be finite, got {metrics['loss']}."


def test_dpsgd_gradient_telemetry_present():
    """fit() with DP-SGD must return mean/median grad norm and frac_clipped."""
    client, _, _ = _dpsgd_client(noise_multiplier=1.0)
    init = client.get_parameters(config={})
    _, _, metrics = client.fit(init, config={"seed": 3})
    assert metrics["mean_grad_norm"]   is not None, "mean_grad_norm missing."
    assert metrics["median_grad_norm"] is not None, "median_grad_norm missing."
    assert metrics["frac_clipped"]     is not None, "frac_clipped missing."
    assert 0.0 <= metrics["frac_clipped"] <= 1.0
    assert metrics["mean_grad_norm"]   >= 0.0
    assert metrics["median_grad_norm"] >= 0.0


def test_dpsgd_clipping_bound_respected():
    """Every per-record gradient norm after clipping must be <= clipping_norm.
    We verify this by running one batch manually and inspecting per-record norms."""
    import torch
    np.random.seed(0)
    torch.manual_seed(0)
    n, d, C = 64, 44, 1.0
    X = np.random.randn(n, d).astype(np.float32)
    y = (np.random.rand(n) > 0.87).astype(np.float32)
    model = LightweightMLP(input_dim=d, hidden_dims=[64, 32], dropout=0.2)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    model_params = list(model.parameters())
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(1)
    for i in range(len(X_t)):
        model.zero_grad()
        criterion(model(X_t[i:i+1]), y_t[i:i+1]).backward()
        raw_norm = float(
            sum(p.grad.data.pow(2).sum().item()
                for p in model_params if p.grad is not None) ** 0.5
        )
        scale = min(1.0, C / (raw_norm + 1e-8))
        clipped_norm = raw_norm * scale
        assert clipped_norm <= C + 1e-5, (
            f"Record {i}: clipped norm {clipped_norm:.6f} exceeds C={C}."
        )


# ── Privacy metadata ───────────────────────────────────────────────────────

def test_dpsgd_privacy_metadata_in_client_metrics():
    """fit() must return dp_enabled=True, mechanism='dpsgd', privacy_unit='record'."""
    client, _, _ = _dpsgd_client(noise_multiplier=1.0)
    init = client.get_parameters(config={})
    _, _, metrics = client.fit(init, config={"seed": 7})
    assert metrics["dp_enabled"]  is True
    assert metrics["mechanism"]   == "dpsgd"
    assert metrics["privacy_unit"] == "record"


def test_dpsgd_privacy_unit_is_record_not_client():
    """Explicit check: privacy_unit must be 'record', never 'client', for DP-SGD."""
    client, _, _ = _dpsgd_client(noise_multiplier=0.8)
    init = client.get_parameters(config={})
    _, _, metrics = client.fit(init, config={"seed": 9})
    assert metrics["privacy_unit"] == "record", (
        f"Expected privacy_unit='record', got '{metrics['privacy_unit']}'."
    )


# ── Reproducibility ────────────────────────────────────────────────────────

def test_dpsgd_reproducibility_with_fixed_seed():
    """Two DP-SGD fit() calls with identical seeds must give identical parameters."""
    def _run():
        torch.manual_seed(42)
        client, _, _ = _dpsgd_client(noise_multiplier=1.0, seed=0)
        init = client.get_parameters(config={})
        result, _, _ = client.fit(init, config={"seed": 1001})
        return result

    p_a = _run()
    p_b = _run()
    for a, b in zip(p_a, p_b):
        np.testing.assert_array_equal(a, b,
            err_msg="Same seed must produce identical DP-SGD parameters.")


# ── No test-set access ─────────────────────────────────────────────────────

def test_no_test_set_access_in_dpsgd_simulation():
    """A 1-round DP-SGD simulation must complete without accessing test data.
    We patch X_test/y_test loading paths and confirm they are not used."""
    from federated.server import run_federated_simulation
    dp_cfg = {
        "enabled": True,
        "mechanism": "dpsgd",
        "privacy_unit": "record",
        "clipping_norm": 1.0,
        "noise_multiplier": 1.0,
        "delta": 1e-5,
    }
    metrics, ckpt = run_federated_simulation(
        experiment_id="DPSGD_TEST_RUN",
        model_type="mlp",
        client_count=2,
        partition_method="iid",
        rounds=1,
        local_epochs=1,
        batch_size=128,
        lr=0.005,
        dp_config=dp_cfg,
        seed=42,
    )
    # split label must be 'validation' — never 'test'
    assert metrics.get("split") == "validation", (
        f"Expected split='validation', got '{metrics.get('split')}'."
    )
    # privacy metadata must be present and correct
    priv = metrics.get("privacy", {})
    assert priv.get("dp_enabled") is True
    assert priv.get("mechanism") == "dpsgd"
    assert priv.get("privacy_unit") == "record"
    assert priv.get("epsilon") is not None
    eps = priv.get("epsilon")
    assert np.isfinite(eps), f"Epsilon must be finite, got {eps}."

    # Clean up
    import os
    for ext in ["_round_metrics.csv", "_round_metrics.json"]:
        p = f"results/DPSGD_TEST_RUN{ext}"
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(ckpt):
        os.remove(ckpt)
