"""EulerIntegrator — applies time derivatives to fields.

Reads dE/dt and dI/dt from state.metrics and does forward Euler:
    E_new = E + dt * dE/dt
    I_new = I + dt * dI/dt

Must run after RBFOperator and QBEOperator.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus


class EulerIntegrator:
    """Forward Euler integration of field derivatives."""

    @property
    def name(self) -> str:
        return "euler"

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        dt = config.dt
        dE_dt = state.metrics.get("dE_dt")
        dI_dt = state.metrics.get("dI_dt")

        E_new = state.E
        I_new = state.I

        if dE_dt is not None:
            E_new = E_new + dt * dE_dt
        if dI_dt is not None:
            I_new = I_new + dt * dI_dt

        return state.replace(E=E_new, I=I_new)
