"""
Scales Layer

December 2025 addition: φ-based scale hierarchy.

The scale hierarchy where Ψ(k) = φ^(-k) defines amplitude at each scale,
and adjacent scales have ratio φ. This explains why structure forms
at particular scales and why particle masses have specific ratios.

Modules:
- scale_hierarchy: Multi-scale field with PAC recursion
"""

from .scale_hierarchy import ScaleHierarchy, ScaleLevel, MultiScaleField

__all__ = [
    'ScaleHierarchy',
    'ScaleLevel',
    'MultiScaleField',
]
