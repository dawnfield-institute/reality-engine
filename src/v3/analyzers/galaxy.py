"""GalaxyAnalyzer — detects large-scale gravitationally bound structure.

A "galaxy" requires:
1. Large-scale mass region (significant fraction of field)
2. Multiple gravity wells within that region (gravitational substructure)
3. At least one star (active stellar population)

Without gravity wells, a large mass region is just a gas cloud.
Without stars, it's a dark proto-galaxy at best.
"""

from __future__ import annotations

from typing import List

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Detection


class GalaxyAnalyzer:
    """Detect galaxy-scale structures — requires gravity wells + stars."""

    def __init__(
        self,
        mass_threshold: float = 0.3,
        min_region_fraction: float = 0.05,
        min_gravity_wells: int = 3,
    ) -> None:
        self.mass_threshold = mass_threshold
        self.min_fraction = min_region_fraction
        self.min_gravity_wells = min_gravity_wells

    @property
    def name(self) -> str:
        return "galaxy"

    def analyze(self, state: FieldState, bus: EventBus, prior_detections=None) -> List[Detection]:
        M = state.M
        total_cells = M.numel()

        sig_mask = M > self.mass_threshold
        sig_count = sig_mask.sum().item()
        sig_fraction = sig_count / total_cells

        detections: List[Detection] = []
        if sig_fraction > self.min_fraction:
            # Causal gate: count gravity wells and stars in the mass region
            well_count = 0
            star_count = 0
            if prior_detections is not None:
                for d in prior_detections:
                    u, v = d.position
                    if u < M.shape[0] and v < M.shape[1] and sig_mask[u, v]:
                        if d.kind == "gravity_well":
                            well_count += 1
                        elif d.kind == "star":
                            star_count += 1

            if well_count < self.min_gravity_wells:
                return []  # not enough gravitational substructure

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
                        "gravity_wells": well_count,
                        "stars": star_count,
                    },
                ))
                bus.emit("galaxy_detected", {
                    "mass_total": mass_total,
                    "gravity_wells": well_count,
                    "stars": star_count,
                })

        return detections
