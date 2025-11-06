"""
Reality Engine - Pure Field Evolution Creating All of Physics

Based on Dawn Field Theory:
- No imposed physics laws
- Everything emerges from recursive balance
- Collapse-regeneration as the engine of reality

Usage:
    from reality_engine import RealityEngine
    
    # Create universe
    engine = RealityEngine(shape=(128, 128, 128))
    
    # Big Bang
    engine.big_bang()
    
    # Evolve and watch physics emerge
    report = engine.evolve(steps=10000)
"""

from .engine import RealityEngine
from .core import DawnField, BigBangEvent
from .emergence import QuantumEmergence, ParticleEmergence
from .utils import EmergenceMetrics

__version__ = "1.0.0"
__all__ = [
    'RealityEngine',
    'DawnField', 
    'BigBangEvent',
    'QuantumEmergence',
    'ParticleEmergence',
    'EmergenceMetrics'
]
