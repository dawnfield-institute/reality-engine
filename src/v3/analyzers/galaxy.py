"""GalaxyAnalyzer — detects large-scale structure in the mass field.

A "galaxy" is a connected region of significant mass spanning more than
a minimum fraction of the field — cosmic web / large-scale structure.
"""

from __future__ import annotations

from typing import List

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Detection


class GalaxyAnalyzer:
    """Detect large-scale mass structures (galaxies / cosmic web)."""

    def __init__(self, mass_threshold: float = 0.3, min_region_fraction: float = 0.05) -> None:
        self.mass_threshold = mass_threshold
        self.min_fraction = min_region_fraction

    @property
    def name(self) -> str:
        return "galaxy"

    def analyze(self, state: FieldState, bus: EventBus) -> List[Detection]:
        M = state.M
        total_cells = M.numel()

        # Significant mass regions
        sig_mask = M > self.mass_threshold
        sig_count = sig_mask.sum().item()
        sig_fraction = sig_count / total_cells

        detections: List[Detection] = []
        if sig_fraction > self.min_fraction:
            # Find centroid of mass distribution
            positions = torch.nonzero(sig_mask, as_tuple=False).float()
            if len(positions) > 0:
                centroid = positions.mean(dim=0)
                mass_total = M[sig_mask].sum().item()
                detections.append(Detection(
                    kind="galaxy",
                    position=(int(centroid[0].item()), int(centroid[1].item())),
                    properties={
                        "mass_total": mass_total,
                        "region_fraction": sig_fraction,
                        "region_size": int(sig_count),
                    },
                ))
                bus.emit("galaxy_detected", {
                    "mass_total": mass_total,
                    "region_fraction": sig_fraction,
                })

        return detections
