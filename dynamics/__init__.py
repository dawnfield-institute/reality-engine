"""
Dynamics Layer

Physical evolution operators with thermodynamic coupling and time emergence.

December 2025 upgrade: Added Klein-Gordon with PAC-derived mass.

Modules:
- time_emergence: Time from disequilibrium pressure, relativistic effects
- confluence: Möbius inversion for geometric time stepping
- klein_gordon: Klein-Gordon evolution with m² = (Ξ-1)/Ξ
"""

from .time_emergence import TimeEmergence, TimeMetrics
from .confluence import MobiusConfluence, create_confluence_operator
from .klein_gordon import KleinGordonEvolution, KleinGordonMetrics, create_initial_perturbation

__all__ = [
    'TimeEmergence',
    'TimeMetrics',
    'MobiusConfluence',
    'create_confluence_operator',
    'KleinGordonEvolution',
    'KleinGordonMetrics',
    'create_initial_perturbation',
]
