"""Substrate layer: Möbius manifold, field state, constants."""

from .mobius import MobiusManifold
from .state import FieldState
from .constants import (
    XI_REFERENCE, ALPHA_REFERENCE, PHI, PHI_INV,
    SEC_DEFAULTS,
)

__all__ = [
    "MobiusManifold",
    "FieldState",
    "XI_REFERENCE",
    "ALPHA_REFERENCE",
    "PHI",
    "PHI_INV",
    "SEC_DEFAULTS",
]
