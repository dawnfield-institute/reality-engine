"""HerniationDetector — detects Möbius topology herniations.

A herniation is a point where the antiperiodic constraint is strongly violated:
|f(u+π, 1-v) + f(u,v)| > threshold

These represent topological stress points where new structure can form.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Detection
from src.v3.substrate.manifold import MobiusManifold


class HerniationDetector:
    """Detect Möbius topology herniations."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "herniation"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    def analyze(self, state: FieldState, bus: EventBus, prior_detections=None) -> List[Detection]:
        m = self._get_manifold(state)

        # Antiperiodic violation: |f_twisted + f|
        violation_E = (m.twist(state.E) + state.E).abs()
        violation_I = (m.twist(state.I) + state.I).abs()
        violation = violation_E + violation_I

        hern_mask = violation > self.threshold

        detections: List[Detection] = []
        if hern_mask.any():
            positions = torch.nonzero(hern_mask, as_tuple=False)
            total_herniations = positions.shape[0]
            intensity = violation[hern_mask].mean().item()
            # Report top-10 strongest for the detection list (can be hundreds)
            violations_at_pos = violation[hern_mask]
            _, top_idx = violations_at_pos.topk(min(10, total_herniations))
            for idx in top_idx:
                pos = positions[idx]
                u, v = pos[0].item(), pos[1].item()
                detections.append(Detection(
                    kind="herniation",
                    position=(u, v),
                    properties={
                        "intensity": violation[u, v].item(),
                    },
                ))
            bus.emit("herniation_detected", {
                "count": total_herniations,
                "mean_intensity": intensity,
            })

        return detections
