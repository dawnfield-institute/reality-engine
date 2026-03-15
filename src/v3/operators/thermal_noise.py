"""ThermalNoiseOperator — Langevin dynamics.

Adds thermal fluctuations to E and I fields:
    noise = scale · √(2T·dt) · N(0,1)

Without noise the system over-stabilises and freezes.
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

        noise_E = scale * amplitude * torch.randn_like(state.E)
        noise_I = scale * amplitude * torch.randn_like(state.I)

        return state.replace(E=state.E + noise_E, I=state.I + noise_I)
