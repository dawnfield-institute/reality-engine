"""Derived constants for Reality Engine v3.

All from Dawn Field Theory — geometric/experimental facts, not tuneable.
Pure Python math module — no numpy, no torch overhead for scalars.
"""

import math

# Core mathematical constants
PHI: float = (1 + math.sqrt(5)) / 2          # φ = 1.6180339887498949
PHI_INV: float = 1 / PHI                      # 1/φ = 0.6180339887498949

# Möbius geometry
XI: float = 1 + math.pi / 55                  # Ξ = 1.0571072... (universal balance)
TWIST_ANGLE: float = math.pi                   # π — Möbius half-twist
ALPHA_PAC: float = 0.964                       # Memory coefficient (derived from Ξ)

# Mass parameter
M_SQUARED: float = (XI - 1) / XI              # m² ≈ 0.054 (Klein-Gordon mass)

# Dynamical constants
LAMBDA: float = 0.020                          # λ — universal frequency (Hz)
ALPHA_SEC: float = 1.0                         # SEC local attraction
BETA_MED: float = 0.6                          # MED global smoothing

# Conservation
PAC_TOLERANCE: float = 1e-12                   # Machine precision for PAC
