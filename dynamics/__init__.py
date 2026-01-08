"""
Dynamics Layer

Physical evolution operators with thermodynamic coupling and time emergence.

December 2025 upgrade: Added Klein-Gordon with PAC-derived mass.
January 2026 upgrade: Added Feigenbaum bifurcation detection.

Modules:
- time_emergence: Time from disequilibrium pressure, relativistic effects
- confluence: Möbius inversion for geometric time stepping
- klein_gordon: Klein-Gordon evolution with m² = (Ξ-1)/Ξ
- feigenbaum_detector: Period-doubling detection with universal constants
- resonance_detector: Oscillation frequency detection
"""

from .time_emergence import TimeEmergence, TimeMetrics
from .confluence import MobiusConfluence, create_confluence_operator
from .klein_gordon import KleinGordonEvolution, KleinGordonMetrics, create_initial_perturbation
from .feigenbaum_detector import (
    FeigenbaumDetector, BifurcationEvent,
    detect_period, compute_delta_from_ratios, verify_universality
)
from .resonance_detector import ResonanceDetector

__all__ = [
    'TimeEmergence',
    'TimeMetrics',
    'MobiusConfluence',
    'create_confluence_operator',
    'KleinGordonEvolution',
    'KleinGordonMetrics',
    'create_initial_perturbation',
    # Feigenbaum universality (v3.0)
    'FeigenbaumDetector',
    'BifurcationEvent',
    'detect_period',
    'compute_delta_from_ratios',
    'verify_universality',
    # Resonance detection
    'ResonanceDetector',
]
