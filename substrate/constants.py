"""
Universal Constants for Reality Engine v2

These constants emerge from Möbius geometry and validated experiments.
DO NOT tune these - they are geometric/experimental facts!
"""

import numpy as np

# =============================================================================
# CORE MATHEMATICAL CONSTANTS (algebraically exact)
# =============================================================================

# Golden ratio - unique positive solution to r² = r + 1
PHI = (1 + np.sqrt(5)) / 2  # 1.6180339887498949...
PHI_INV = 1 / PHI           # 0.6180339887498949... = φ - 1

# Geometric Constants (from Möbius topology)
XI = 1.0571  # Ξ - Universal balance constant (Möbius holonomy correction)
TWIST_ANGLE = np.pi  # π - Möbius half-twist

# =============================================================================
# FEIGENBAUM UNIVERSALITY (validated to 13+ digits)
# =============================================================================

# Feigenbaum constants (period-doubling universality)
DELTA_FEIGENBAUM = 4.669201609102990671853203820466  # Bifurcation ratio
ALPHA_FEIGENBAUM = 2.502907875095892822283902873218  # Scaling factor

# Critical parameter values
R_INF_LOGISTIC = 3.5699456718709449  # Onset of chaos in logistic map
R_INF_SINE = 0.8924864179677363      # Onset of chaos in sine map (= R_inf_logistic / 4)

# Universal offset from Möbius fixed point
UNIVERSAL_DELTA_Z = 5.382561e-04  # M₁₀(-1/φ + Δz) = r_inf/π

# Fibonacci Möbius transformation M₁₀(z) = (89z + 55)/(55z + 34)
M10_A = 89  # F₁₁
M10_B = 55  # F₁₀
M10_C = 55  # F₁₀
M10_D = 34  # F₉

# Eigenvalue at unstable fixed point -1/φ
M10_EIGENVALUE = PHI ** 20  # ≈ 15126.99993...

# Key algebraic identity: 89 - 55φ = 1/φ¹⁰
EIGENVALUE_IDENTITY = 89 - 55 * PHI  # ≈ 0.008130618755783

# Structural constants (4-5 pattern)
STRUCT_39 = 39    # (5⁴-1)/4² = 624/16
STRUCT_160 = 160  # 4² × 2 × 5
STRUCT_1371 = 1371  # F₁₀ × 5² - 4 = 55 × 25 - 4
STRUCT_1857 = 1857  # F₁₀ × F₉ - F₇ = 55 × 34 - 13

# =============================================================================
# DYNAMICAL CONSTANTS (from legacy experiments)
# =============================================================================

LAMBDA = 0.020  # λ - Universal frequency (Hz)
ALPHA_SEC = 1.0  # α - SEC local attraction strength
BETA_MED = 0.6   # β - MED global smoothing strength

# =============================================================================
# CONSERVATION TOLERANCE
# =============================================================================

PAC_TOLERANCE = 1e-12  # Machine precision for conservation

# =============================================================================
# VALIDATION TARGETS (from experiments)
# =============================================================================

# These should be DISCOVERED by the law detector, not hardcoded
# Included here only as validation targets
EXPECTED_FREQUENCY = 0.020  # Hz - Should emerge naturally
EXPECTED_XI = 1.0571        # Should emerge from geometry
EXPECTED_MAX_DEPTH = 2      # Structures should be depth ≤ 2
EXPECTED_MODE_TYPE = "half_integer"  # Möbius signature
