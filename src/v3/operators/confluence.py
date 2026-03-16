"""ConfluenceOperator — Möbius antiperiodic geometric transformation.

The confluence operation transforms Potential → Actual through the
Möbius manifold's non-orientable topology:

    I_actualized = twist(E)  →  (f(u+π, 1-v))
    I_new = (1-w)·I + w·project_antiperiodic(I_actualized)

This creates temporal flow without assuming time exists a priori.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


class ConfluenceOperator:
    """Apply Möbius antiperiodic projection to information field."""

    def __init__(self) -> None:
        self._manifold: Optional[MobiusManifold] = None
        self._tick_counter: int = 0

    @property
    def name(self) -> str:
        return "confluence"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu or self._manifold.nv != nv:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        if not config.enable_confluence:
            return state

        self._tick_counter += 1
        if self._tick_counter % config.confluence_every != 0:
            return state

        m = self._get_manifold(state)

        # Twist E to get actualized I: I_act = E(u+π, 1-v)
        I_actualized = m.twist(state.E)

        # Project onto antiperiodic subspace
        I_actualized = m.project_antiperiodic(I_actualized)

        # Blend with current I field
        w = config.confluence_weight
        I_new = (1.0 - w) * state.I + w * I_actualized

        # PAC conservation: confluence changes sum(I) without changing sum(E) or sum(M).
        # Scale I_new to preserve sum(I), maintaining antiperiodic structure.
        I_old_sum = state.I.sum()
        I_new_sum = I_new.sum()
        if I_new_sum.abs() > 1e-10 and I_old_sum.abs() > 1e-10:
            I_new = I_new * (I_old_sum / I_new_sum)
        elif I_new_sum.abs() > 1e-10:
            # Old sum was ~0 (antiperiodic), new sum drifted — subtract mean
            I_new = I_new - I_new.mean()

        if bus is not None:
            magnitude = (I_new - state.I).abs().mean().item()
            bus.emit("confluence_applied", {"magnitude": magnitude})

        return state.replace(I=I_new)
