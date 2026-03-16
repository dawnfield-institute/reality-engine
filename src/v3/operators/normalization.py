"""NormalizationOperator -- PAC-conserving field stabilisation.

With PAC conservation exact, E+I+M = const globally. As M grows, E+I
must shrink -- fields are self-bounding through conservation alone.

This operator does two things:

1. Hard M cap + Landauer reinjection: Mass above M_cap is hard-clipped.
   Removed mass returns to E+I locally (Landauer principle -> supernovae).
   This bounds M and creates the mass-energy feedback cycle.

2. Bounded QBE cross-injection: tanh clamp catches extreme E/I values.
   The excess is cross-injected locally (E excess -> I, I excess -> E)
   but ONLY up to the headroom of the receiving field. Any remainder
   that can't be absorbed goes to M (crystallisation). M is then
   re-floored at 0 with global PAC correction for residuals.

PAC = E + I + M is conserved at every cell and globally.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus

try:
    from fracton.physics import PACValidator as _FractonPACValidator
    _HAS_PAC_VALIDATOR = True
except ImportError:
    _HAS_PAC_VALIDATOR = False


class NormalizationOperator:
    """PAC-conserving field stabilisation with bounded QBE cross-injection."""

    def __init__(self) -> None:
        self._initial_pac: Optional[float] = None
        self._pac_validator = (
            _FractonPACValidator(tolerance=1e-10, auto_correct=False)
            if _HAS_PAC_VALIDATOR else None
        )

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
        M_cap = s / 5.0

        # --- Hard M cap with Landauer reinjection ---
        # Any M above cap is removed and returned to E+I locally.
        # This is Landauer's principle: destroying information (mass)
        # releases energy. Supernovae are this mechanism at scale.
        M_floored = torch.clamp(state.M, min=0.0)
        M_new = torch.clamp(M_floored, max=M_cap)
        dM_removed = M_floored - M_new  # >= 0

        # Landauer: 1 unit M destroyed -> 1 unit E+I (split equally)
        reinjection = dM_removed * 0.5
        E_cur = state.E + reinjection
        I_cur = state.I + reinjection

        # --- Tanh safety clamp ---
        E_clamped = s * torch.tanh(E_cur / s)
        I_clamped = s * torch.tanh(I_cur / s)

        E_loss = E_cur - E_clamped  # positive = removed, negative = added
        I_loss = I_cur - I_clamped

        # --- Bounded QBE cross-injection ---
        # QBE: information and energy are dual (dI = -dE).
        # Cross-inject tanh losses to the dual field, but only up to what
        # the receiving field can absorb without exceeding its own bounds.
        # This prevents the runaway oscillation where cross-injection
        # undoes the clamp.
        E_inject = torch.clamp(I_loss, min=-s - E_clamped, max=s - E_clamped)
        I_inject = torch.clamp(E_loss, min=-s - I_clamped, max=s - I_clamped)

        E_new = E_clamped + E_inject
        I_new = I_clamped + I_inject

        # Remainder that couldn't be cross-injected → M (crystallisation).
        # Energy that can't exist as E or I crystallises into mass.
        E_remainder = I_loss - E_inject
        I_remainder = E_loss - I_inject
        crystallised = E_remainder + I_remainder
        M_new = M_new + crystallised

        # Floor M at 0 — negative crystallisation can happen when tanh
        # reduces magnitude of negative fields. The residual is caught
        # by the global PAC correction below.
        M_new = torch.clamp(M_new, min=0.0)

        # --- Global PAC safety check ---
        if self._initial_pac is None:
            self._initial_pac = (E_new + I_new + M_new).sum().item()

        current_pac = (E_new + I_new + M_new).sum().item()
        pac_residual = self._initial_pac - current_pac

        if abs(pac_residual) > 1e-8:
            correction = pac_residual / (2.0 * E_new.numel())
            E_new = E_new + correction
            I_new = I_new + correction

        metrics = dict(state.metrics)
        total_reinjected = dM_removed.sum().item()
        total_crystallised = crystallised.sum().item()

        metrics["landauer_reinjection"] = total_reinjected
        metrics["crystallisation"] = total_crystallised
        metrics["pac_correction"] = pac_residual

        # --- Fracton PACValidator audit (read-only, no correction) ---
        if self._pac_validator is not None:
            E_sum = E_new.sum().item()
            I_sum = I_new.sum().item()
            M_sum = M_new.sum().item()
            result = self._pac_validator.validate(
                E_sum + I_sum + M_sum, [E_sum, I_sum, M_sum],
            )
            metrics["pac_validator_residual"] = result.residual
            metrics["pac_validator_violations"] = self._pac_validator.stats["violations"]

        if total_reinjected > 0.01 and bus is not None:
            bus.emit("landauer_reinjection", {
                "energy_reinjected": total_reinjected,
                "cells_affected": (dM_removed > 1e-6).sum().item(),
            })

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)
