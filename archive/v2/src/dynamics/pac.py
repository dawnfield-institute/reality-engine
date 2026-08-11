"""
PACTracker — enforce and measure additive conservation.

    PAC  =  P + A + M  =  const        (ADDITIVE, no Ξ coefficient)

Ξ appears ONLY in spectral measurement — it is an *observable* of the
system, not a parameter.  This avoids circularity.

Two modes:
    ENFORCE  — project back onto the conservation surface after each step
    MEASURE  — diagnostic only, report drift
"""

from __future__ import annotations

from typing import Any

import torch

from ..substrate.state import FieldState


class PACTracker:
    """Track and (optionally) enforce additive PAC conservation."""

    def __init__(self, initial_state: FieldState) -> None:
        self.C_target: float = (
            initial_state.P.sum()
            + initial_state.A.sum()
            + initial_state.M.sum()
        ).item()
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    def measure(self, state: FieldState) -> dict[str, Any]:
        """Compute conservation diagnostics *without* modifying the state."""
        C_current = (
            state.P.sum() + state.A.sum() + state.M.sum()
        ).item()
        residual = abs(C_current - self.C_target) / max(abs(self.C_target), 1e-14)

        result: dict[str, Any] = {
            "t": state.t,
            "C": C_current,
            "C_target": self.C_target,
            "residual": residual,
            "P_total": state.P.sum().item(),
            "A_total": state.A.sum().item(),
            "M_total": state.M.sum().item(),
        }
        self.history.append(result)
        return result

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------
    def enforce(self, state: FieldState) -> FieldState:
        """
        Project *state* back onto ``P + A + M = C_target``.

        Distributes the residual uniformly to each grid point,
        weighted by field magnitudes (minimum-norm correction in ℓ∞).
        Mutates in-place and returns the same ``FieldState``.

        NOTE: the uniform correction adds to E_sym (zero mode) without
        affecting E_anti.  This keeps P_mean near the collapse saturation
        point (S ≈ 1/β₀) where the nonlinear collapse naturally produces
        ξ ≈ 1.0.  The topology-modulated SEC then shifts ξ to the target.
        """
        C_current = (
            state.P.sum() + state.A.sum() + state.M.sum()
        ).item()
        delta = self.C_target - C_current

        if abs(delta) < 1e-14:
            return state

        totals = (
            state.P.abs().sum()
            + state.A.abs().sum()
            + state.M.abs().sum()
        )
        if totals < 1e-14:
            return state

        n = state.P.numel()  # grid cells
        w_P = state.P.abs().sum() / totals
        w_A = state.A.abs().sum() / totals
        w_M = state.M.abs().sum() / totals

        state.P += delta * w_P / n
        state.A += delta * w_A / n
        state.M += delta * w_M / n

        return state
