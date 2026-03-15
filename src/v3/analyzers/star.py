"""StarDetector — finds stellar ignition sites.

A "star" requires:
1. High mass concentration (fuel)
2. High temperature (ignition)
3. A gravity well at the same location (gravitational collapse caused this)

Without a gravity well, high M + high T is just a hot gas cloud, not a star.
This enforces the causal chain: gravity → stellar collapse → star.
"""

from __future__ import annotations

from typing import List

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Detection, detections_near


class StarDetector:
    """Detect star-like structures — requires gravity well prerequisite."""

    def __init__(self, mass_threshold: float = 3.0, temp_threshold: float = 2.0) -> None:
        self.mass_threshold = mass_threshold
        self.temp_threshold = temp_threshold

    @property
    def name(self) -> str:
        return "star"

    def analyze(self, state: FieldState, bus: EventBus, prior_detections=None) -> List[Detection]:
        star_mask = (state.M > self.mass_threshold) & (state.T > self.temp_threshold)

        detections: List[Detection] = []
        if star_mask.any():
            positions = torch.nonzero(star_mask, as_tuple=False)
            for pos in positions[:10]:
                u, v = pos[0].item(), pos[1].item()

                # Causal gate: must be inside a gravity well
                if prior_detections is not None:
                    nearby_wells = detections_near(
                        prior_detections, "gravity_well", (u, v), radius=3
                    )
                    if not nearby_wells:
                        continue  # hot dense region but no gravity well → not a star

                detections.append(Detection(
                    kind="star",
                    position=(u, v),
                    properties={
                        "mass": state.M[u, v].item(),
                        "temperature": state.T[u, v].item(),
                    },
                ))
            if detections:
                bus.emit("star_detected", {"count": len(detections)})

        return detections
