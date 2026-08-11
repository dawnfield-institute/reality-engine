"""
FieldState — complete simulation state at one timestep.

All field tensors are shape (n_u, n_v) and live on the same device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FieldState:
    """Complete state at one timestep.  All tensors shape ``(n_u, n_v)``."""

    P: torch.Tensor          # Potential field
    A: torch.Tensor          # Actualization field
    M: torch.Tensor          # Memory / momentum field
    t: int = 0               # Discrete timestep (confluence cycle count)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        devices = {self.P.device, self.A.device, self.M.device}
        if len(devices) != 1:
            raise ValueError(
                f"Device mismatch: P={self.P.device}, "
                f"A={self.A.device}, M={self.M.device}"
            )
        shapes = {self.P.shape, self.A.shape, self.M.shape}
        if len(shapes) != 1:
            raise ValueError(
                f"Shape mismatch: P={self.P.shape}, "
                f"A={self.A.shape}, M={self.M.shape}"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.P.device

    @property
    def shape(self) -> torch.Size:
        return self.P.shape

    @property
    def pac_total(self) -> float:
        """Additive PAC = P + A + M.  Should be constant.  No Ξ coefficient."""
        return (self.P + self.A + self.M).sum().item()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def to(self, device: str | torch.device) -> "FieldState":
        """Move all tensors to *device*, returning a new ``FieldState``."""
        return FieldState(
            P=self.P.to(device),
            A=self.A.to(device),
            M=self.M.to(device),
            t=self.t,
        )

    def clone(self) -> "FieldState":
        """Deep-copy every tensor."""
        return FieldState(
            P=self.P.clone(),
            A=self.A.clone(),
            M=self.M.clone(),
            t=self.t,
        )
