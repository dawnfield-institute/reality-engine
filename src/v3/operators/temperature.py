"""TemperatureOperator — local temperature from energy gradients.

    T = β · ||∇E||²

Blended with previous temperature for smoothness, clamped to [T_min, T_max].
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


class TemperatureOperator:
    """Compute temperature from energy gradients."""

    def __init__(self, beta: float = 1.0, blend: float = 0.3) -> None:
        self.beta = beta
        self.blend = blend  # fraction of previous T to keep
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "temperature"

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
        m = self._get_manifold(state)

        # T from disequilibrium (primary source)
        T_base = state.disequilibrium

        # Blend with previous
        T_new = (1.0 - self.blend) * T_base + self.blend * state.T

        # Clamp
        T_new = torch.clamp(T_new, min=config.t_min, max=config.t_max)

        return state.replace(T=T_new)
