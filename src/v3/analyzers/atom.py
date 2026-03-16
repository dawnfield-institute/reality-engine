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
            total_atoms = positions.shape[0]

            # Vectorized local metallicity via avg-pooling (3x3 neighborhood)
            Z_padded = torch.nn.functional.pad(state.Z.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            local_Z_map = torch.nn.functional.avg_pool2d(Z_padded, kernel_size=3, stride=1, padding=0).squeeze()

            masses = state.M[atom_mask]
            grads = grad_E[atom_mask]
            local_Zs = local_Z_map[atom_mask]
            is_heavy = local_Zs > self.metallicity_threshold

            total_heavy = int(is_heavy.sum().item())
            total_hydrogen = total_atoms - total_heavy

            # Report top-20 by mass
            n_report = min(20, total_atoms)
            _, top_idx = masses.topk(n_report)

            for i in range(n_report):
                idx = top_idx[i]
                pos = positions[idx]
                kind = "atom" if is_heavy[idx] else "hydrogen"
                detections.append(Detection(
                    kind=kind,
                    position=(pos[0].item(), pos[1].item()),
                    properties={
                        "mass": masses[idx].item(),
                        "energy_gradient": grads[idx].item(),
                        "metallicity": local_Zs[idx].item(),
                    },
                ))
            if total_atoms > 0:
                bus.emit("atom_detected", {"atoms": total_heavy, "hydrogen": total_hydrogen})

        return detections
