"""QuantumDetector — finds quantum coherence patterns.

Quantum coherence: regions where E and I fields oscillate in near-lockstep
(high correlation) — analogous to entanglement / coherent superposition.
"""

from __future__ import annotations

from typing import List

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.event_bus import EventBus
from src.v3.analyzers.base import Detection


class QuantumDetector:
    """Detect quantum-coherent field regions."""

    def __init__(self, coherence_threshold: float = 0.9) -> None:
        self.coherence_threshold = coherence_threshold

    @property
    def name(self) -> str:
        return "quantum"

    def analyze(self, state: FieldState, bus: EventBus) -> List[Detection]:
        E, I = state.E, state.I

        # Local coherence: normalised correlation |E·I| / (|E|·|I| + ε)
        EI = E * I
        coherence = EI.abs() / (E.abs() * I.abs() + 1e-10)

        coh_mask = coherence > self.coherence_threshold

        detections: List[Detection] = []
        if coh_mask.any():
            # Report region centroid rather than every pixel
            positions = torch.nonzero(coh_mask, as_tuple=False).float()
            if len(positions) > 0:
                centroid = positions.mean(dim=0)
                detections.append(Detection(
                    kind="quantum_coherence",
                    position=(int(centroid[0].item()), int(centroid[1].item())),
                    properties={
                        "coherent_fraction": coh_mask.float().mean().item(),
                        "mean_coherence": coherence[coh_mask].mean().item(),
                        "region_size": int(coh_mask.sum().item()),
                    },
                ))
                bus.emit("quantum_coherence_detected", {
                    "coherent_fraction": detections[0].properties["coherent_fraction"],
                })

        return detections
