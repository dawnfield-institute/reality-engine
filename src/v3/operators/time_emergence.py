"""TimeEmergenceOperator — emergent time step from disequilibrium.

Time doesn't exist a priori. It emerges from the physics:
    dt_effective = dt_base / (1 + κ · max(|E - I|))

Regions of high activity experience faster evolution (more happens per tick).
This operator modifies config.dt for the *next* tick.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus


class TimeEmergenceOperator:
    """Compute emergent dt from field disequilibrium."""

    def __init__(self, kappa: float = 0.1, dt_base: Optional[float] = None) -> None:
        self.kappa = kappa
        self._dt_base: Optional[float] = dt_base

    @property
    def name(self) -> str:
        return "time_emergence"

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        if self._dt_base is None:
            self._dt_base = config.dt

        max_diseq = state.disequilibrium.max().item()
        dt_new = self._dt_base / (1.0 + self.kappa * max_diseq)

        # Clamp to reasonable range
        dt_new = max(1e-6, min(dt_new, self._dt_base * 2))
        config.dt = dt_new

        metrics = dict(state.metrics)
        metrics["emergent_dt"] = dt_new
        metrics["max_disequilibrium"] = max_diseq

        if bus is not None:
            bus.emit("time_emerged", {"dt": dt_new, "max_diseq": max_diseq})

        return state.replace(metrics=metrics)
