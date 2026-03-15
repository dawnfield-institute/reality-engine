"""AtomDetector — finds bound energy-memory structures.

An "atom" requires:
1. High M at centre (nucleus)
2. High |E| gradient around it (electron cloud analog)
3. Nonzero Z (metallicity) nearby — fusion products needed for anything
   heavier than hydrogen. Without a star having fused material first,
   you can't have complex atoms.

This enforces: gravity → star → fusion → metals → atoms.
Hydrogen-like structures (M peak, no Z) are reported as "hydrogen" kind.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Detection
from src.v3.substrate.manifold import MobiusManifold


class AtomDetector:
    """Detect atom-like bound structures with causal metallicity gate."""

    def __init__(
        self,
        mass_threshold: float = 1.0,
        gradient_threshold: float = 0.5,
        metallicity_threshold: float = 0.01,
    ) -> None:
        self.mass_threshold = mass_threshold
        self.gradient_threshold = gradient_threshold
        self.metallicity_threshold = metallicity_threshold
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "atom"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    def analyze(self, state: FieldState, bus: EventBus, prior_detections=None) -> List[Detection]:
        m = self._get_manifold(state)

        # Nuclei: high mass concentrations
        nuclei_mask = state.M > self.mass_threshold

        # Energy shell: high gradient magnitude around nucleus
        grad_E = m.gradient_magnitude(state.E)
        shell_mask = grad_E > self.gradient_threshold

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
            for pos in positions[:15]:
                u, v = pos[0].item(), pos[1].item()

                # Check local metallicity to determine element type
                u_lo = max(0, u - 1)
                u_hi = min(state.shape[0], u + 2)
                v_lo = max(0, v - 1)
                v_hi = min(state.shape[1], v + 2)
                local_Z = state.Z[u_lo:u_hi, v_lo:v_hi].mean().item()

                if local_Z > self.metallicity_threshold:
                    kind = "atom"  # heavy element — fusion products present
                else:
                    kind = "hydrogen"  # mass peak with no metals

                detections.append(Detection(
                    kind=kind,
                    position=(u, v),
                    properties={
                        "mass": state.M[u, v].item(),
                        "energy_gradient": grad_E[u, v].item(),
                        "metallicity": local_Z,
                    },
                ))
            if detections:
                atom_count = sum(1 for d in detections if d.kind == "atom")
                h_count = sum(1 for d in detections if d.kind == "hydrogen")
                bus.emit("atom_detected", {"atoms": atom_count, "hydrogen": h_count})

        return detections
