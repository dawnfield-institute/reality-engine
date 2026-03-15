"""FieldState — immutable container for all simulation fields.

Every operator receives a FieldState and returns a new one.
Fields are torch tensors on the same device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class FieldState:
    """Immutable snapshot of all simulation fields.

    Attributes:
        E: Energy field (actualization) — shape (nu, nv)
        I: Information field (potential) — shape (nu, nv)
        M: Memory field (mass/persistence) — shape (nu, nv)
        T: Temperature field — shape (nu, nv)
        Z: Metallicity field (fusion products) — shape (nu, nv)
        tick: Current simulation tick
        dt: Time step used to reach this state
        time: Cumulative simulation time
        metrics: Operator-populated metrics dict (frozen via tuple trick)
    """

    E: torch.Tensor
    I: torch.Tensor
    M: torch.Tensor
    T: torch.Tensor
    Z: Optional[torch.Tensor] = None  # metallicity — only written by FusionOperator
    tick: int = 0
    dt: float = 0.001
    time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-create Z as zeros if not provided
        if self.Z is None:
            object.__setattr__(self, "Z", torch.zeros_like(self.E))

    # --- helpers ---------------------------------------------------------

    def replace(self, **kwargs: Any) -> FieldState:
        """Return a new FieldState with specified fields replaced."""
        current = {
            "E": self.E,
            "I": self.I,
            "M": self.M,
            "T": self.T,
            "Z": self.Z,
            "tick": self.tick,
            "dt": self.dt,
            "time": self.time,
            "metrics": self.metrics,
        }
        current.update(kwargs)
        return FieldState(**current)

    @property
    def device(self) -> torch.device:
        return self.E.device

    @property
    def shape(self) -> torch.Size:
        return self.E.shape

    @property
    def disequilibrium(self) -> torch.Tensor:
        """Point-wise |E - I|."""
        return (self.E - self.I).abs()

    @property
    def total_energy(self) -> float:
        """Scalar total energy E² + I² + M²."""
        return (self.E.pow(2) + self.I.pow(2) + self.M.pow(2)).sum().item()

    @property
    def pac_total(self) -> float:
        """PAC conservation quantity: sum(E + I + α·M)."""
        alpha_pac = 0.964
        return (self.E + self.I + alpha_pac * self.M).sum().item()

    @staticmethod
    def zeros(
        nu: int = 128,
        nv: int = 32,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> FieldState:
        """Create a zero-initialised FieldState."""
        device = device or torch.device("cpu")
        z = torch.zeros(nu, nv, dtype=dtype, device=device)
        return FieldState(E=z.clone(), I=z.clone(), M=z.clone(), T=z.clone(), Z=z.clone())

    @staticmethod
    def big_bang(
        nu: int = 128,
        nv: int = 32,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        temperature: float = 5.0,
    ) -> FieldState:
        """Hot dense initial state — the starting point for a universe."""
        device = device or torch.device("cpu")
        E = torch.randn(nu, nv, dtype=dtype, device=device) * temperature
        I = torch.randn(nu, nv, dtype=dtype, device=device) * temperature
        M = torch.zeros(nu, nv, dtype=dtype, device=device)
        T = torch.full((nu, nv), temperature, dtype=dtype, device=device)
        Z = torch.zeros(nu, nv, dtype=dtype, device=device)
        return FieldState(E=E, I=I, M=M, T=T, Z=Z)
