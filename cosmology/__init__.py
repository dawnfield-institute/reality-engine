"""
Cosmology Layer

December 2025 addition: JWST anomaly predictions.

Computes cosmological observables for comparison with JWST/Hubble data.
Key predictions:
- Hubble tension as scale-dependent H(k)
- Early SMBH formation via herniation
- 0.02 Hz signature in gravitational wave background
- Dark energy/matter fractions from PAC equilibrium

Modules:
- observables: Cosmological predictions and JWST comparisons
"""

from .observables import (
    CosmologicalObservables,
    CosmologicalPrediction,
    GalaxyPrediction,
)

__all__ = [
    'CosmologicalObservables',
    'CosmologicalPrediction',
    'GalaxyPrediction',
]
