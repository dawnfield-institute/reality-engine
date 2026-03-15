"""QBEOperator — Quantum Balance Enforcement.

Regulates E↔I dynamics: dI/dt + dE/dt ≈ 0 (with optional QPL modulation).
Prevents runaway by distributing the balance field between E and I.

Reads dE/dt from state.metrics (set by RBFOperator) and computes dI/dt.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus


class QBEOperator:
    """Enforce quantum balance: dI/dt = -dE/dt + λ·QPL(t).

    Must run after RBFOperator (needs dE/dt in metrics).
    """

    def __init__(self, qpl_omega: float = 0.020) -> None:
        self.qpl_omega = qpl_omega

    @property
    def name(self) -> str:
        return "qbe"

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        dE_dt = state.metrics.get("dE_dt")
        if dE_dt is None:
            return state  # No RBF output yet — passthrough

        # QPL modulation: small oscillatory source term
        qpl = math.sin(self.qpl_omega * 2 * math.pi * state.time)
        dI_dt = -dE_dt + config.lambda_freq * qpl

        metrics = dict(state.metrics)
        metrics["dI_dt"] = dI_dt

        return state.replace(metrics=metrics)
