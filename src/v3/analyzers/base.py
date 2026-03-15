"""Analyzer protocol — read-only observers that emit events.

Analyzers inspect FieldState and emit detection events via the EventBus.
They never modify state. They run after all operators in the tick cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable

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
    """Protocol for read-only field analyzers."""

    @property
    def name(self) -> str: ...

    def analyze(self, state: FieldState, bus: EventBus) -> List[Detection]: ...
