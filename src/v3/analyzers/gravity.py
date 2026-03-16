"""GravityAnalyzer — detects gravity wells from memory field curvature.

A gravity well is a local maximum in M (mass concentration) with
negative Laplacian (concave — matter has pooled).
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Detection
from src.v3.substrate.manifold import MobiusManifold


class GravityAnalyzer:
    """Detect gravity wells from memory field curvature."""

    def __init__(self, mass_threshold: float = 0.5, min_curvature: float = 0.1) -> None:
        self.mass_threshold = mass_threshold
        self.min_curvature = min_curvature
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "gravity"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    def analyze(self, state: FieldState, bus: EventBus, prior_detections=None) -> List[Detection]:
        m = self._get_manifold(state)
        M = state.M

        # Gravity wells: high mass + negative Laplacian (concave)
        lap_M = m.laplacian(M)
        wells_mask = (M > self.mass_threshold) & (lap_M < -self.min_curvature)

        detections: List[Detection] = []
        if wells_mask.any():
            positions = torch.nonzero(wells_mask, as_tuple=False)
            total_wells = positions.shape[0]

            # Vectorized extraction — no Python loop
            masses = M[wells_mask]
            curvatures = lap_M[wells_mask]

            # Report top-20 strongest wells by mass (avoid thousands of Detection objects)
            n_report = min(20, total_wells)
            _, top_idx = masses.topk(n_report)

            for i in range(n_report):
                idx = top_idx[i]
                pos = positions[idx]
                detections.append(Detection(
                    kind="gravity_well",
                    position=(pos[0].item(), pos[1].item()),
                    properties={
                        "mass": masses[idx].item(),
                        "curvature": curvatures[idx].item(),
                    },
                ))
            bus.emit("gravity_well_detected", {"count": total_wells})

        return detections
