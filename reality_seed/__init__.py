"""
Reality Seed - Generative Reality Engine

A reality that grows itself from pure PAC dynamics.
No pre-defined physics. No assumed structure.
Just events, conservation, and observation.

Based on PAC-Lazy tensor architecture.
"""

from .genesis import GenesisSeed, GenesisObserver
from .pac_substrate import PACNode, PACSubstrate
from .visualizer import GenesisVisualizer, run_genesis
from .patterns import (
    PatternDetector, 
    PatternCodeGenerator, 
    EmergenceAnalyzer,
    DetectedPattern,
)

__all__ = [
    'GenesisSeed',
    'GenesisObserver', 
    'PACNode',
    'PACSubstrate',
    'GenesisVisualizer',
    'run_genesis',
    'PatternDetector',
    'PatternCodeGenerator',
    'EmergenceAnalyzer',
    'DetectedPattern',
]
