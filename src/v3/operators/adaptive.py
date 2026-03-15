"""AdaptiveOperator — self-tuning physics parameters.

Adjusts coupling constants based on field state feedback:
- γ (damping) adapts to maintain stability
- dt can shrink when fields are volatile

Uses logarithmic adaptation (slow, stable) to avoid oscillation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus


class AdaptiveOperator:
    """Self-tuning parameter controller."""

    def __init__(
        self,
        gamma_init: float = 0.005,
        gamma_min: float = 0.001,
        gamma_max: float = 0.1,
        dt_min: float = 1e-5,
        dt_max: float = 0.01,
        adaptation_rate: float = 0.01,
    ) -> None:
        self.gamma = gamma_init
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.rate = adaptation_rate
        self._prev_energy: Optional[float] = None

    @property
    def name(self) -> str:
        return "adaptive"

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        if not config.enable_adaptive:
            return state

        energy = state.total_energy

        if self._prev_energy is not None:
            ratio = energy / (self._prev_energy + 1e-10)

            # Energy growing fast → increase damping, decrease dt
            if ratio > 1.1:
                self.gamma = min(self.gamma * (1 + self.rate), self.gamma_max)
                config.dt = max(config.dt * (1 - self.rate), self.dt_min)
            # Energy stable or shrinking → relax damping slightly
            elif ratio < 1.01:
                self.gamma = max(self.gamma * (1 - self.rate * 0.1), self.gamma_min)
                config.dt = min(config.dt * (1 + self.rate * 0.1), self.dt_max)

        self._prev_energy = energy

        metrics = dict(state.metrics)
        metrics["adaptive_gamma"] = self.gamma
        metrics["adaptive_dt"] = config.dt

        if bus is not None:
            bus.emit("parameters_adapted", {
                "gamma": self.gamma,
                "dt": config.dt,
                "energy": energy,
            })

        return state.replace(metrics=metrics)
