"""
Universal Constants for Reality Engine v2

These constants emerge from Möbius geometry and validated experiments.
DO NOT tune these - they are geometric/experimental facts!
"""

# Geometric Constants (from Möbius topology)
XI = 1.0571  # Ξ - Universal balance constant (Möbius holonomy correction)
TWIST_ANGLE = 3.14159265359  # π - Möbius half-twist

# Dynamical Constants (from legacy experiments)
LAMBDA = 0.020  # λ - Universal frequency (Hz)
ALPHA_SEC = 1.0  # α - SEC local attraction strength
BETA_MED = 0.6   # β - MED global smoothing strength

# Conservation Tolerance
PAC_TOLERANCE = 1e-12  # Machine precision for conservation

# Physical Constants (emergent, not imposed!)
# These should be DISCOVERED by the law detector, not hardcoded
# Included here only as validation targets

# Validation Targets (from legacy experiments)
EXPECTED_FREQUENCY = 0.020  # Hz - Should emerge naturally
EXPECTED_XI = 1.0571        # Should emerge from geometry
EXPECTED_MAX_DEPTH = 2      # Structures should be depth ≤ 2
EXPECTED_MODE_TYPE = "half_integer"  # Möbius signature
