"""
Constants for Reality Engine v2.

Reference values for validation — NOT inputs to dynamics.
Derivations traced in PACSeries Papers 1-5.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Golden ratio and derived
# ---------------------------------------------------------------------------
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0        # 1.6180339887...
PHI_INV: float = PHI - 1.0                         # 0.6180339887... = 1/φ

# ---------------------------------------------------------------------------
# Ξ reference values (for validation, NOT injection)
# ---------------------------------------------------------------------------
# PACSeries Paper 2: Ξ_analytic = γ + ln(φ) ≈ 1.0584
# PACSeries Paper 1/5: Ξ_PAC ≈ 1.0571 (recursive saturation via F₁₀ = 55)
# These bracket the expected emergent value.
XI_REFERENCE: float = 1.0571                        # Primary target
XI_ANALYTIC: float = 0.5772156649 + math.log(PHI)   # γ + ln(φ) ≈ 1.0584
ALPHA_REFERENCE: float = 0.964                       # Memory coeff (from Ξ)

# ---------------------------------------------------------------------------
# Möbius topology
# ---------------------------------------------------------------------------
TWIST_ANGLE: float = math.pi                        # Half-twist

# ---------------------------------------------------------------------------
# SEC default parameters (robust to ±20% per GAIA validation)
# ---------------------------------------------------------------------------
SEC_DEFAULTS: dict = {
    "kappa": 0.1,        # Diffusion coefficient
    "gamma": 1.0,        # Collapse coupling
    "beta_0": 1.0,       # Base collapse rate
    "sigma_0": 0.1,      # Source strength (balances collapse at S_ss ≈ 0.11)
    "dt": 0.01,          # Integration timestep
    "xi_gain": 2.0,      # Bidirectional Ξ modulation sensitivity
    "rho": 1.0,          # Topological reinforcement strength
    "phi_source": PHI_INV,  # Antiperiodic source ratio (1/φ ≈ 0.618)
    # --- RBF (Recursive Balance Field) parameters ---
    # Memory-dampened balance with Fibonacci breathing.
    # See vcpu_unified.py for reference: B = λ·[(E-I)/(1+α|M|)]·Φ
    "alpha_rbf": 5.0,       # Memory dampening coefficient
    "rbf_decay": 0.995,     # Memory exponential decay per step
    "ki_rbf": 1.0,          # Integral gain (eliminates proportional droop)
    "integral_clamp": 1.0,  # Anti-windup clamp for integral term
    "rbf_omega": 0.2,       # Fibonacci harmonic base frequency
    "low_k_mix": 0.382,     # Low-k anti mode mix (≈ φ⁻², topology-natural)
}

# ---------------------------------------------------------------------------
# Actualization defaults
# ---------------------------------------------------------------------------
ACT_DEFAULTS: dict = {
    "rate": 0.1,         # Crystallisation/dissolution rate
    "scale": 1.0,        # Laplacian → coherence scale
    "memory": 0.9,       # EMA smoothing for A
}

# ---------------------------------------------------------------------------
# PAC
# ---------------------------------------------------------------------------
PAC_TOLERANCE: float = 1e-12                        # Machine-precision target
