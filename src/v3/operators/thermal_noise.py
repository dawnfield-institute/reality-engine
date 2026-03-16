"""ThermalNoiseOperator — Langevin dynamics (PAC-conserving).

Adds anticorrelated thermal fluctuations: E += η, I -= η.
This drives disequilibrium (E-I) without changing the total E+I,
preserving PAC conservation while preventing over-stabilisation.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus


class ThermalNoiseOperator:
    """Inject thermal fluctuations into E and I fields."""

    @property
    def name(self) -> str:
        return "thermal_noise"

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        if not config.enable_thermal_noise:
            return state

        T_mean = state.T.mean()
        amplitude = torch.sqrt(2.0 * T_mean * config.dt + 1e-10)
        scale = config.noise_scale

        # Single noise field: E += η, I -= η  →  E+I unchanged (PAC-conserving)
        noise = scale * amplitude * torch.randn_like(state.E)

        return state.replace(E=state.E + noise, I=state.I - noise)
