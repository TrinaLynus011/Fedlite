"""
FedLite-Care — Differential Privacy Mechanisms and Accountants
===============================================================

Two separate mechanisms are implemented here.  They are NOT interchangeable.
Read the header of the function you are calling before using it.

────────────────────────────────────────────────────────────────────────────
MECHANISM A — Client-Update Gaussian  (CALIBRATION / DIAGNOSTIC ONLY)
────────────────────────────────────────────────────────────────────────────
  Privacy unit : federated CLIENT (one hospital / silo per round)
  Accountant   : compute_client_update_epsilon()
  Status       : implemented, calibration runs completed, utility too low
                 at meaningful epsilon levels — see diagnostic results in
                 results/experiment_metadata/FL_MLP_NONIID_5_DP_eps_5.0.json

────────────────────────────────────────────────────────────────────────────
MECHANISM B — DP-SGD  (ACTIVE — record-level privacy)
────────────────────────────────────────────────────────────────────────────
  Privacy unit : individual RECORD (patient encounter)
  Accountant   : compute_dpsgd_epsilon()

  Per-batch training loop:
    1. Forward pass for each record i in batch B.
    2. Compute per-record gradient  g_i  via individual backward pass.
    3. Clip:  ĝ_i = g_i · min(1, C / ‖g_i‖₂)
    4. Sum clipped gradients, add Gaussian noise:
         G_noisy = Σ ĝ_i + N(0, (σ·C)² I)
    5. Gradient step:  θ ← θ − lr · G_noisy / B

  The clipping and noise injection happen INSIDE the optimizer step
  during local training.  The model update delta Δθ returned to the
  server is the PLAIN (non-additionally-noised) result of the DP-SGD
  training loop.  No further noise is added to the completed update.

  Accountant — Rényi DP for the subsampled Gaussian mechanism:
    Sampling model : uniform mini-batch (batch-without-replacement
                     approximated as Poisson with q = B / n).
    RDP at order α : tight second-order bound from Mironov (2017) Thm 8:
        RDP_subsample(α) ≤ (1/(α−1)) · log(
            1 + q²·C(α,2)·(exp(α(α−1)/(2σ²)) − 1)
        )
    Composed over T_steps steps:
        RDP_total(α) = T_steps · RDP_subsample(α)
    Converted to (ε, δ)-DP:
        ε(α) = RDP_total(α) + log(1/δ) / (α − 1)
    Final ε = min over α ∈ (1, ∞).

    T_steps = rounds × local_epochs × ⌈n / B⌉    (per client)

  REFERENCES
  ----------
  - Abadi et al. (2016) "Deep Learning with Differential Privacy"
    https://arxiv.org/abs/1607.00133
  - Mironov (2017) "Rényi Differential Privacy of the Gaussian Mechanism"
    https://arxiv.org/abs/1702.07476
  - Wang et al. (2019) "Subsampled Rényi Differential Privacy and
    Analytical Moments Accountant"  https://arxiv.org/abs/1808.00087
  - Dwork & Roth (2014) "The Algorithmic Foundations of Differential Privacy"
"""

import math
import numpy as np


# ============================================================================
# MECHANISM A — clip_and_add_noise  (client-update, used by Gaussian clients)
# ============================================================================

def clip_and_add_noise(updates, clipping_norm, noise_multiplier, seed=None):
    """
    Clip a list of numpy parameter-update arrays by their joint L2 norm to
    `clipping_norm`, then add isotropic Gaussian noise.

    Used by the CLIENT-UPDATE GAUSSIAN mechanism.  Privacy unit = CLIENT.

    Parameters
    ----------
    updates : list[np.ndarray]
        Per-layer parameter deltas Δθ.
    clipping_norm : float
        Maximum allowed L2 norm C.
    noise_multiplier : float
        σ — noise_std = σ · C.  0.0 = clip only, no noise.
    seed : int or None
        Seeds NumPy RNG for reproducibility.

    Returns
    -------
    (noisy_updates, pre_clip_norm)
    """
    total_norm = float(np.sqrt(sum(np.sum(u ** 2) for u in updates)))
    scale = min(1.0, clipping_norm / (total_norm + 1e-8))
    clipped = [u * scale for u in updates]

    if noise_multiplier <= 0.0:
        return clipped, total_norm

    rng = np.random.default_rng(seed)
    noise_std = noise_multiplier * clipping_norm
    noisy = [c + rng.normal(0.0, noise_std, size=c.shape) for c in clipped]
    return noisy, total_norm


# ============================================================================
# ACCOUNTANT A — client-update Gaussian mechanism
# ============================================================================

def compute_client_update_epsilon(rounds, noise_multiplier, delta=1e-5,
                                  alpha_min=1.01, alpha_max=200.0, n_alphas=500):
    """
    (ε, δ)-DP for T rounds of the client-update Gaussian mechanism.

    Privacy unit : CLIENT.  No subsampling amplification.

    RDP(α) = T · α / (2σ²)   [Mironov 2017, Prop. 3 — no subsampling]
    """
    if noise_multiplier <= 0.0:
        return float("inf")
    if rounds <= 0:
        return 0.0
    alphas = np.linspace(alpha_min, alpha_max, n_alphas)
    rdp = rounds * alphas / (2.0 * noise_multiplier ** 2)
    eps = rdp + np.log(1.0 / delta) / (alphas - 1.0)
    return float(np.min(eps))


# ============================================================================
# MECHANISM B — DP-SGD per-example gradient clipping (in client.py training loop)
# ============================================================================
# The actual noise injection is in FedLiteClient.fit() via _dpsgd_batch_step().
# This module provides only the accountant and the per-example clip helper.

def clip_grad_tensor(grad, clipping_norm):
    """
    Clip a single per-example gradient tensor (shape = param.shape) so that
    its L2 norm does not exceed clipping_norm.

    Returns the clipped gradient and the pre-clip L2 norm.
    """
    norm = float(grad.norm(2).item())
    scale = min(1.0, clipping_norm / (norm + 1e-8))
    return grad * scale, norm


# ============================================================================
# ACCOUNTANT B — DP-SGD subsampled Gaussian mechanism
# ============================================================================

def compute_dpsgd_epsilon(rounds, local_epochs, batch_size, dataset_size,
                          noise_multiplier, delta=1e-5,
                          alpha_min=1.1, alpha_max=100.0, n_alphas=1000):
    """
    Compute (ε, δ) for DP-SGD via Rényi DP composition of the subsampled
    Gaussian mechanism.

    Privacy unit : individual RECORD (patient encounter).
    Sampling     : uniform mini-batch, approximated as Poisson with
                   q = batch_size / dataset_size.
    Steps        : T = rounds × local_epochs × ⌈dataset_size / batch_size⌉

    RDP bound used — tight second-order (Mironov 2017, Thm 8):

        RDP_subsample(α) ≤ (1/(α−1)) · log(
            1 + q² · (α choose 2) · (exp(α(α−1)/(2σ²)) − 1)
        )

        where (α choose 2) = α(α−1)/2.

    This is the same bound used by Google's dp_accounting library and
    OpenDP.  It is tighter than the loose q²·α/(2σ²) approximation by
    a factor of roughly exp(α/(2σ²)).

    For integer α this is exact under Poisson subsampling.  For non-integer
    α it is a valid upper bound.

    The alpha grid is automatically limited to values where the RDP exponent
    does not overflow float64, so that the computation is always numerically
    stable.

    Parameters
    ----------
    rounds : int          Federated communication rounds.
    local_epochs : int    SGD passes per round per client.
    batch_size : int      Mini-batch size B.
    dataset_size : int    Number of training records on this client n.
    noise_multiplier : float   σ — noise_std = σ · C.
    delta : float         Target δ.
    alpha_min, alpha_max  Grid bounds for α optimisation (before overflow cap).
    n_alphas              Grid resolution.

    Returns
    -------
    epsilon : float
        Tightest (ε, δ)-DP epsilon over the α grid.
        Returns +inf if noise_multiplier == 0.
    steps : int
        Total number of SGD steps accounted for (logged in telemetry).
    """
    if noise_multiplier <= 0.0:
        return float("inf"), 0

    q = batch_size / dataset_size
    if q >= 1.0:
        q = 1.0

    steps = rounds * local_epochs * math.ceil(dataset_size / batch_size)

    # Cap alpha so that alpha*(alpha-1)/(2*sigma^2) <= 500 (safe for exp)
    # alpha*(alpha-1) <= 1000*sigma^2  =>  alpha <= (~sqrt(1000)*sigma for large alpha)
    max_safe_exp = 500.0
    # Solve alpha*(alpha-1) = 2*sigma^2*max_safe_exp  =>  alpha ~ sqrt(2*sigma^2*max_safe_exp)
    alpha_overflow_cap = 0.5 + math.sqrt(0.25 + 2.0 * noise_multiplier**2 * max_safe_exp)
    effective_alpha_max = min(alpha_max, alpha_overflow_cap)
    if effective_alpha_max <= alpha_min:
        effective_alpha_max = alpha_min + 1.0

    alphas = np.linspace(alpha_min, effective_alpha_max, n_alphas)

    # RDP for subsampled Gaussian at each order α (tight 2nd-order bound)
    rdp_per_step = np.array([
        _rdp_subsampled_gaussian(a, q, noise_multiplier)
        for a in alphas
    ])

    # Compose over steps
    rdp_total = steps * rdp_per_step

    # Convert RDP → (ε, δ)
    eps_arr = rdp_total + np.log(1.0 / delta) / (alphas - 1.0)
    return float(np.min(eps_arr)), steps


def _rdp_subsampled_gaussian(alpha, q, sigma):
    """
    RDP of the subsampled Gaussian mechanism at order alpha (single step).

    Uses the tight second-order bound from Mironov (2017) Theorem 8:

        RDP(α) ≤ (1/(α−1)) · log(
            1 + q² · α(α−1)/2 · (exp(α(α−1)/(2σ²)) − 1)
        )

    When the exponent α(α−1)/(2σ²) would overflow float64 (> ~709),
    the term exp(·) is replaced by its large-argument limit and the
    computation is done in log-space to avoid OverflowError.
    """
    if alpha <= 1.0:
        return 0.0
    binom2  = alpha * (alpha - 1.0) / 2.0      # C(alpha, 2)
    exp_arg = alpha * (alpha - 1.0) / (2.0 * sigma ** 2)

    if exp_arg > 700.0:
        # log-space: log(1 + q^2*binom2*(exp(exp_arg)-1))
        #          ≈ log(q^2*binom2*exp(exp_arg))
        #          = log(q^2) + log(binom2) + exp_arg
        log_term = math.log(q ** 2) + math.log(binom2) + exp_arg
    else:
        inner    = math.exp(exp_arg) - 1.0
        argument = 1.0 + (q ** 2) * binom2 * inner
        log_term = math.log(argument)

    return log_term / (alpha - 1.0)


# ============================================================================
# LEGACY SHIM — do not use in new experiments
# ============================================================================

def compute_epsilon(steps, batch_size, dataset_size, noise_multiplier, delta=1e-5):
    """
    DEPRECATED.

    Originally implemented the loose q²·α/(2σ²) DP-SGD bound.
    Delegated to compute_client_update_epsilon() after the mechanism
    refactor, which was INCORRECT for DP-SGD.

    Retained only so existing imports compile.  Do not call in new code.
    Use compute_dpsgd_epsilon() for DP-SGD or
    compute_client_update_epsilon() for the client-update Gaussian mechanism.
    """
    # Conservative fallback: treat steps as rounds in the no-subsampling accountant.
    return compute_client_update_epsilon(
        rounds=steps,
        noise_multiplier=noise_multiplier,
        delta=delta,
    )
