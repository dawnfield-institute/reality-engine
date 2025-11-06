"""
Conservation Layer

Enforces PAC conservation with thermodynamic-information duality.

Modules:
- thermodynamic_pac: PAC enforcement with Landauer costs, heat flow, 2nd law
- sec_operator: Symbolic Entropy Collapse operator for field evolution
"""

from .thermodynamic_pac import ThermodynamicPAC, ThermodynamicMetrics
from .sec_operator import SymbolicEntropyCollapse, create_sec_operator

__all__ = [
    'ThermodynamicPAC',
    'ThermodynamicMetrics',
    'SymbolicEntropyCollapse',
    'create_sec_operator',
]
