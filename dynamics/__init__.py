"""
Dynamics Layer

Physical evolution operators with thermodynamic coupling and time emergence.

Modules:
- time_emergence: Time from disequilibrium pressure, relativistic effects
- confluence: Möbius inversion for geometric time stepping
"""

from .time_emergence import TimeEmergence, TimeMetrics
from .confluence import MobiusConfluence, create_confluence_operator

__all__ = [
    'TimeEmergence',
    'TimeMetrics',
    'MobiusConfluence',
    'create_confluence_operator',
]
