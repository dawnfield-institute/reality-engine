"""StarDetector — finds high-mass collapsed regions.

A "star" is a large mass concentration with high temperature —
gravitational collapse generating heat (like real stellar formation).
"""

from __future__ import annotations

from typing import List

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Detection


class StarDetector:
    """Detect star-like high-mass, high-temperature structures."""

    def __init__(self, mass_threshold: float = 3.0, temp_threshold: float = 2.0) -> None:
        self.mass_threshold = mass_threshold
        self.temp_threshold = temp_threshold

    @property
    def name(self) -> str:
        return "star"

    def analyze(self, state: FieldState, bus: EventBus) -> List[Detection]:
        star_mask = (state.M > self.mass_threshold) & (state.T > self.temp_threshold)

        detections: List[Detection] = []
        if star_mask.any():
            positions = torch.nonzero(star_mask, as_tuple=False)
            for pos in positions[:5]:
                u, v = pos[0].item(), pos[1].item()
                detections.append(Detection(
                    kind="star",
                    position=(u, v),
                    properties={
                        "mass": state.M[u, v].item(),
                        "temperature": state.T[u, v].item(),
                    },
                ))
            bus.emit("star_detected", {"count": len(detections)})

        return detections
