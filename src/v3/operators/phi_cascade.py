"""PhiCascadeOperator — Fibonacci two-step memory for phi-spaced mass levels.

DFT proves that phi-scaling in mass spectra requires two-step memory feedback:
    P(n+1) = w1 * P(n) + w2 * P(n-1)

This is the Fibonacci recurrence. The weights come from DFT:
    w1 = alpha_PAC = 0.964 (actualization coupling)
    w2 = XI - 1 = 0.0571 (topology feedback coefficient)

The operator creates discrete mass "shelves" at M_cap * phi^(-k) by modulating
the mass generation rate with a Fibonacci-weighted cascade memory. This produces
energy levels at phi^(-k) intervals instead of approximately linear spacing.

The cascade depth k = -log(M / M_cap) / log(phi) is exported as a metric for
downstream operators (Phase 4 pi/2-harmonic collapse modulation uses it).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.constants import PHI, XI, ALPHA_PAC


_EPS = 1e-12
_LN_PHI = math.log(PHI)


class PhiCascadeOperator:
    """Fibonacci two-step memory creating phi-spaced mass levels.

    Maintains per-cell history of mass generation rates and applies a
    Fibonacci-weighted correction that creates discrete mass shelves
    at M_cap * phi^(-k) for integer k.

    Exports cascade_depth metric for downstream operators.
    """

    def __init__(self) -> None:
        self._rate_prev1: Optional[torch.Tensor] = None  # previous tick rate
        self._rate_prev2: Optional[torch.Tensor] = None  # two ticks ago rate

    @property
    def name(self) -> str:
        return "phi_cascade"

    def _init_history(self, shape: tuple, device: torch.device) -> None:
        """Initialize rate history tensors on first call or shape change."""
        if (
            self._rate_prev1 is None
            or self._rate_prev1.shape != shape
            or self._rate_prev1.device != device
        ):
            self._rate_prev1 = torch.zeros(shape, dtype=torch.float64, device=device)
            self._rate_prev2 = torch.zeros(shape, dtype=torch.float64, device=device)

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        E, I, M = state.E, state.I, state.M
        dt = config.dt
        M_cap = config.field_scale / 5.0

        self._init_history(M.shape, M.device)

        # --- Current mass generation rate (from MemoryOperator upstream) ---
        # Use the actual mass_generation_rate metric if available,
        # otherwise compute disequilibrium-based proxy
        current_rate = state.metrics.get("mass_generation_rate", None)
        if current_rate is None:
            diseq2 = (E - I).pow(2)
            total_field2 = E.pow(2) + I.pow(2) + _EPS
            gamma_local = diseq2 / total_field2
            current_rate_tensor = gamma_local * diseq2
        else:
            # mass_generation_rate is a scalar mean — use disequilibrium as spatial proxy
            diseq2 = (E - I).pow(2)
            total_field2 = E.pow(2) + I.pow(2) + _EPS
            gamma_local = diseq2 / total_field2
            current_rate_tensor = gamma_local * diseq2

        # --- Fibonacci two-step cascade rate ---
        # P(n+1) = w1 * P(n) + w2 * P(n-1)
        # w1 = alpha_PAC (0.964), w2 = XI - 1 (0.0571)
        w1 = ALPHA_PAC
        w2 = XI - 1.0  # 0.0571 — topology feedback
        cascade_rate = w1 * self._rate_prev1 + w2 * self._rate_prev2

        # --- Cascade depth per cell ---
        # k = -log(M / M_cap) / log(phi) — how many phi-levels below cap
        M_safe = torch.clamp(M, min=_EPS)
        cascade_depth = -torch.log(M_safe / M_cap) / _LN_PHI
        cascade_depth = torch.clamp(cascade_depth, min=0.0, max=12.0)

        # --- Phi-level proximity ---
        # How close each cell is to a phi^(-k) shelf
        # At exact shelf: frac(depth) = 0 or 1 → proximity = 1
        # Between shelves: frac(depth) = 0.5 → proximity = 0
        depth_frac = cascade_depth - cascade_depth.floor()
        phi_proximity = torch.cos(math.pi * depth_frac).pow(2)

        # --- Cascade modulation of mass ---
        # Cells near phi-shelves get enhanced mass retention (stabilized)
        # Cells between shelves get enhanced diffusion (pushed toward nearest shelf)
        # The cascade_rate amplifies this effect based on Fibonacci memory

        # Stabilization force: positive at shelves, negative between
        shelf_force = (phi_proximity - 0.5) * 2.0  # [-1, +1]

        # Scale by cascade rate magnitude and dt
        cascade_strength = cascade_rate.abs().clamp(max=1.0)
        dM_cascade = shelf_force * cascade_strength * dt * 0.1  # gentle modulation

        # Apply saturation (same as memory operator)
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        degen = state.metrics.get("degeneracy_level", None)
        sat_degen = (1.0 - degen) if degen is not None else 1.0
        saturation = sat_cap * sat_degen

        dM_cascade = dM_cascade * saturation

        # Update mass
        M_candidate = M + dM_cascade
        M_new = torch.clamp(M_candidate, min=0.0)

        # PAC conservation: drain from E+I
        dM_actual = M_new - M
        pac_drain = dM_actual * 0.5
        E_new = E - pac_drain
        I_new = I - pac_drain

        # --- Update history (shift) ---
        self._rate_prev2 = self._rate_prev1.clone()
        self._rate_prev1 = current_rate_tensor.detach().clone()

        # --- Metrics ---
        metrics = dict(state.metrics)
        metrics["cascade_depth"] = cascade_depth  # tensor for downstream
        metrics["cascade_depth_mean"] = cascade_depth.mean().item()
        metrics["cascade_depth_std"] = cascade_depth.std().item()
        metrics["phi_proximity_mean"] = phi_proximity.mean().item()
        metrics["cascade_rate_mean"] = cascade_rate.abs().mean().item()
        metrics["cascade_dM"] = dM_actual.abs().mean().item()

        if bus is not None:
            bus.emit("phi_cascade_applied", {
                "cascade_depth_mean": metrics["cascade_depth_mean"],
                "phi_proximity_mean": metrics["phi_proximity_mean"],
                "cascade_dM": metrics["cascade_dM"],
            })

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)
