"""ConservationAnalyzer — tracks PAC conservation, entropy, and total mass.

Emits 'conservation_violated' when drift exceeds threshold.
Emits 'conservation_measured' every analysis.
"""

from __future__ import annotations

from typing import List, Optional

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Analyzer, Detection
from src.v3.substrate.constants import PAC_TOLERANCE


class ConservationAnalyzer:
    """Track PAC conservation quality over time."""

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance
        self._prev_pac: Optional[float] = None

    @property
    def name(self) -> str:
        return "conservation"

    def analyze(self, state: FieldState, bus: EventBus, prior_detections=None) -> List[Detection]:
        pac = state.pac_total
        detections: List[Detection] = []

        if self._prev_pac is not None:
            drift = abs(pac - self._prev_pac) / (abs(self._prev_pac) + 1e-10)
            bus.emit("conservation_measured", {
                "pac_total": pac,
                "drift": drift,
                "total_energy": state.total_energy,
                "mass_total": state.M.sum().item(),
            })
            if drift > self.tolerance:
                bus.emit("conservation_violated", {"drift": drift, "pac": pac})
                detections.append(Detection(
                    kind="conservation_violation",
                    position=(0, 0),
                    properties={"drift": drift, "pac": pac},
                ))

        self._prev_pac = pac
        return detections
