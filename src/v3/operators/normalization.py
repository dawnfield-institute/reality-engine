"""NormalizationOperator — prevent field blow-up.

Soft-clamps E and I with tanh, and soft-caps M to preserve structure.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus


class NormalizationOperator:
    """Soft-clamp fields to prevent divergence."""

    @property
    def name(self) -> str:
        return "normalization"

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        if not config.enable_normalization:
            return state

        s = config.field_scale

        # E, I: tanh soft clamp
        E_new = s * torch.tanh(state.E / s)
        I_new = s * torch.tanh(state.I / s)

        # M: non-negative, soft cap above M_scale
        M_scale = s / 5.0  # 10.0 when field_scale=50
        M_new = torch.clamp(state.M, min=0.0)
        M_new = torch.where(
            M_new > M_scale,
            M_scale + M_scale * torch.tanh((M_new - M_scale) / M_scale),
            M_new,
        )

        return state.replace(E=E_new, I=I_new, M=M_new)
