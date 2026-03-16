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

            # Causal gate: filter to only positions near a gravity well
            if prior_detections is not None:
                well_positions = [
                    d.position for d in prior_detections if d.kind == "gravity_well"
                ]
                if well_positions:
                    # Vectorized proximity check
                    wells_t = torch.tensor(well_positions, device=positions.device, dtype=positions.dtype)
                    # For each candidate star, check min distance to any well
                    # positions: (N, 2), wells_t: (W, 2)
                    diffs = positions.unsqueeze(1).float() - wells_t.unsqueeze(0).float()  # (N, W, 2)
                    dists = diffs.norm(dim=2)  # (N, W)
                    min_dists = dists.min(dim=1).values  # (N,)
                    near_well = min_dists <= 3
                    positions = positions[near_well]
                else:
                    positions = positions[:0]  # no wells → no stars

            total_stars = positions.shape[0]
            if total_stars > 0:
                masses = state.M[positions[:, 0], positions[:, 1]]
                temps = state.T[positions[:, 0], positions[:, 1]]

                # Report top-20 by mass
                n_report = min(20, total_stars)
                _, top_idx = masses.topk(n_report)

                for i in range(n_report):
                    idx = top_idx[i]
                    pos = positions[idx]
                    detections.append(Detection(
                        kind="star",
                        position=(pos[0].item(), pos[1].item()),
                        properties={
                            "mass": masses[idx].item(),
                            "temperature": temps[idx].item(),
                        },
                    ))
                bus.emit("star_detected", {"count": total_stars})

        return detections
