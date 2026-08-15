"""TimeEmergenceOperator — emergent time step from disequilibrium.

Time doesn't exist a priori. It emerges from the physics:
    dt_effective = dt_base / (1 + κ · max(|E - I|))

This operator modifies config.dt for the *next* tick.

KNOWN DEFECT — the intent and the behaviour differ, and the intent is the correct one.

    Intended (and previously documented here): "Regions of high activity experience faster
    evolution." That describes a dt FIELD — each region carrying its own clock.

    Actual: `state.disequilibrium.max()` is a single scalar over the entire manifold, and
    `config.dt` is a single float applied to every cell. There are no regions. Every cell's
    clock is set by the one most extreme cell in the field.

Left as a defect rather than reconciled by rewriting the docstring, because the docstring was
right about the physics and the code is what is wrong. The corpus is explicit on this:
`asymmetric_conservation/core/async_pac.py` is built on "event-indexed execution, not global
timesteps", with reconciliation firing per node at |Delta| > Xi, and `archive/era1-symbolic/
legacy/brain.py` carries `time[x, y, z]` as a genuine per-cell counter.

Why it matters beyond bookkeeping: a shared clock is one of the terms that makes this engine's
relation graph effectively complete. Under identity-as-complement (M13), a complete graph has
exactly one identity — remove any vertex of K_n and the remainder is K_{n-1} for every vertex —
and one identity clumps instead of forming structure. A global clock will also damp whatever
per-cell history a refractory accumulates, so it is a registered confound for
dawn-field-theory milestone16 exp_01.

Fixing it means making `dt` a tensor, which touches every operator's integration. Scoped to its
own round; see `experiments/milestones/milestone16/journals/2026-08-14_exp01_prereg_v2.md`.
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
