"""Analyzer protocol — read-only observers that emit events.

Analyzers inspect FieldState and emit detection events via the EventBus.
They never modify state. They run after all operators in the tick cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus


@dataclass
class Detection:
    """A detected structure or phenomenon."""
    kind: str           # e.g. "gravity_well", "atom", "star"
    position: tuple     # (u, v) grid indices
    properties: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for read-only field analyzers.

    Analyzers receive prior_detections — the detections from all analyzers
    that ran earlier in the chain. This enables causal awareness: a star
    detector can require a gravity well, an atom detector can require
    nonzero metallicity (from fusion), etc.
    """

    @property
    def name(self) -> str: ...

    def analyze(
        self,
        state: FieldState,
        bus: EventBus,
        prior_detections: Optional[List[Detection]] = None,
    ) -> List[Detection]: ...


def detections_near(
    detections: List[Detection],
    kind: str,
    position: tuple,
    radius: int = 5,
) -> List[Detection]:
    """Find detections of a given kind within radius of a position."""
    u, v = position
    return [
        d for d in detections
        if d.kind == kind and abs(d.position[0] - u) + abs(d.position[1] - v) <= radius
    ]
