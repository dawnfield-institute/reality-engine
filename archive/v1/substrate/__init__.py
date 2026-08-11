"""
Reality Engine v2 - Substrate Layer

Geometric foundation using Möbius topology with anti-periodic boundaries.
Provides the computational substrate where physics emerges.
"""

from .mobius_manifold import MobiusManifold
from .field_types import FieldState
from .constants import *

__all__ = [
    'MobiusManifold',
    'FieldState',
]
