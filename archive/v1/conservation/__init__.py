"""
Conservation Layer

Enforces PAC conservation with thermodynamic-information duality.

December 2025 upgrade: Added PAC recursion enforcement.

Modules:
- thermodynamic_pac: PAC enforcement with Landauer costs, heat flow, 2nd law
- sec_operator: Symbolic Entropy Collapse operator for field evolution
- pac_recursion: PAC recursion Ψ(k) = Ψ(k+1) + Ψ(k+2) enforcement
"""

from .thermodynamic_pac import ThermodynamicPAC, ThermodynamicMetrics
from .sec_operator import SymbolicEntropyCollapse, create_sec_operator
from .pac_recursion import PACRecursion, PACMetrics, PHI, XI

__all__ = [
    'ThermodynamicPAC',
    'ThermodynamicMetrics',
    'SymbolicEntropyCollapse',
    'create_sec_operator',
    'PACRecursion',
    'PACMetrics',
    'PHI',
    'XI',
]
