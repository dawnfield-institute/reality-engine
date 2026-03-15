"""AtomDetector — finds bound energy-memory structures.

An "atom" is a localised mass concentration surrounded by an energy shell:
- High M at centre (nucleus)
- High |E| gradient around it (electron cloud analog)
- Stable over multiple ticks (tracked by persistence)
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Detection
from src.v3.substrate.manifold import MobiusManifold


class AtomDetector:
    """Detect atom-like bound structures."""

    def __init__(self, mass_threshold: float = 1.0, gradient_threshold: float = 0.5) -> None:
        self.mass_threshold = mass_threshold
        self.gradient_threshold = gradient_threshold
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "atom"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    def analyze(self, state: FieldState, bus: EventBus) -> List[Detection]:
        m = self._get_manifold(state)

        # Nuclei: high mass concentrations
        nuclei_mask = state.M > self.mass_threshold

        # Energy shell: high gradient magnitude around nucleus
        grad_E = m.gradient_magnitude(state.E)
        shell_mask = grad_E > self.gradient_threshold

        # Atom = nucleus with nearby energy shell
        # Dilate nuclei by 1 pixel to check adjacency
        nuclei_float = nuclei_mask.float()
        dilated = (
            torch.roll(nuclei_float, 1, 0) + torch.roll(nuclei_float, -1, 0) +
            torch.roll(nuclei_float, 1, 1) + torch.roll(nuclei_float, -1, 1) +
            nuclei_float
        ) > 0

        atom_mask = nuclei_mask & (dilated.bool() & shell_mask)

        detections: List[Detection] = []
        if atom_mask.any():
            positions = torch.nonzero(atom_mask, as_tuple=False)
            for pos in positions[:10]:
                u, v = pos[0].item(), pos[1].item()
                detections.append(Detection(
                    kind="atom",
                    position=(u, v),
                    properties={
                        "mass": state.M[u, v].item(),
                        "energy_gradient": grad_E[u, v].item(),
                    },
                ))
            bus.emit("atom_detected", {"count": len(detections)})

        return detections
