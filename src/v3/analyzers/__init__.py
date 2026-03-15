"""Analyzers — read-only observers that detect emergent structures."""

from .base import Analyzer, Detection
from .conservation import ConservationAnalyzer
from .gravity import GravityAnalyzer
from .atom import AtomDetector
from .star import StarDetector
from .quantum import QuantumDetector
from .galaxy import GalaxyAnalyzer

__all__ = [
    "Analyzer", "Detection",
    "ConservationAnalyzer", "GravityAnalyzer", "AtomDetector",
    "StarDetector", "QuantumDetector", "GalaxyAnalyzer",
]
