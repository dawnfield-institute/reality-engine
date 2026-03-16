"""MemoryOperator — mass generation, quantum pressure, and diffusion.

Memory field M accumulates where information collapses (E-I disequilibrium).
Quantum pressure prevents uniform collapse. Diffusion redistributes mass.

    dM/dt = mass_generation + quantum_pressure + diffusion
    mass_generation = γ_local · ((E-I)² + |∇(E-I)|²) · saturation
    quantum_pressure = -β · ∇²(M²) / M
    diffusion = D · ∇²M

The mass generation coefficient γ_local is NOT hardcoded — it EMERGES per cell:
    γ_local = (E-I)² / (E² + I² + ε)
This is the fraction of total field energy in disequilibrium form — high where
fields are imbalanced (ready to crystallize), low where fields are balanced.

Two mass generation channels:
1. Bulk: γ_local · (E-I)² — mass forms where disequilibrium is large
2. Boundary: γ_local · |∇(E-I)|² · 1/(1+M) — mass nucleates at boundaries
   between regions with different disequilibrium. This fills voids between
   dense structures, creating web-like filaments instead of isolated clumps.
   The 1/(1+M) adaptive suppression ensures seeding targets empty cells
   and diminishes where mass already exists, preventing late-time overshoot.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


_EPS = 1e-12


class MemoryOperator:
    """Evolve the memory (mass) field from collapse dynamics.

    γ_local = (E-I)² / (E² + I² + ε) — emergent per cell.
    Mass generates faster where disequilibrium dominates total field energy.
    """

    def __init__(self) -> None:
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "memory"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu or self._manifold.nv != nv:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        m = self._get_manifold(state)
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        disequilibrium = E - I

        # --- Emergent mass generation coefficient per cell ---
        # γ_local = (E-I)² / (E² + I² + ε) — disequilibrium fraction of total
        # High where fields are imbalanced (ready to crystallize), low when balanced.
        diseq2 = disequilibrium.pow(2)
        total_field2 = E.pow(2) + I.pow(2) + _EPS
        gamma_local = diseq2 / total_field2

        # Bulk mass generation: gamma_local * (E-I)^2
        mass_gen_bulk = gamma_local * diseq2

        # Boundary mass generation: gamma_local * |∇(E-I)|² * 1/(1+M)
        # Mass nucleates at boundaries between regions with different
        # disequilibrium — fills voids, creates web filaments.
        # Adaptive suppression: 1/(1+M) targets empty cells, diminishes
        # where mass already exists (prevents late-time overshoot).
        grad_u = (torch.roll(disequilibrium, -1, 0) - torch.roll(disequilibrium, 1, 0)) / 2.0
        grad_v = (torch.roll(disequilibrium, -1, 1) - torch.roll(disequilibrium, 1, 1)) / 2.0
        grad_diseq2 = grad_u.pow(2) + grad_v.pow(2)
        mass_gen_boundary = gamma_local * grad_diseq2 / (1.0 + M)

        mass_gen = mass_gen_bulk + mass_gen_boundary
        # --- Two-factor saturation ---
        # Factor 1: Hard cap safety (numerical stability floor)
        #   Linear dropoff to zero at M_cap. Prevents overflow.
        M_cap = config.field_scale / 5.0
        sat_cap = torch.clamp(1.0 - M / M_cap, min=0.0)
        # Factor 2: Degeneracy feedback (emergent mass limits)
        #   If SpinStatisticsOperator ran upstream, it exported degeneracy_level
        #   [0,1] per cell. High degeneracy = half-integer spin, similar neighbors,
        #   mass-dominant -> mass generation throttled.
        #   When spin_statistics is not in the pipeline, falls back to 1.0.
        degen = state.metrics.get("degeneracy_level", None)
        if degen is not None:
            sat_degen = 1.0 - degen
        else:
            sat_degen = 1.0
        saturation = sat_cap * sat_degen
        mass_gen = mass_gen * saturation

        # Quantum pressure: -β · ∇²(M²) / M_safe
        M_safe = M + 1e-6
        lap_M2 = m.laplacian(M * M)
        quantum_pressure = -config.quantum_pressure_coeff * lap_M2 / M_safe

        # Diffusion: D · ∇²M
        lap_M = m.laplacian(M)
        diffusion = config.mass_diffusion_coeff * lap_M

        # Combined
        dM_dt = mass_gen + quantum_pressure + diffusion
        M_candidate = M + dt * dM_dt

        # PAC-conserving clamp: track mass created by min=0 floor
        M_new = torch.clamp(M_candidate, min=0.0)

        # PAC conservation: E + I + M = const.
        # The ACTUAL net M change (including clamp) must be drained from E+I.
        # Note: quantum pressure (∇²(M²)/M) is NOT zero-sum because the
        # division by M breaks the Laplacian's conservation property.
        # So we drain based on the actual M change, not just mass_gen.
        dM_actual = M_new - M  # actual per-cell M change (>= -M)
        net_pac_drain = dM_actual * 0.5
        E_new = E - net_pac_drain
        I_new = I - net_pac_drain

        # Track emergent mass generation coefficient
        gamma_mean = gamma_local.mean().item()
        gamma_std = gamma_local.std().item()

        metrics = dict(state.metrics)
        metrics["mass_generation_rate"] = mass_gen.mean().item()
        metrics["gamma_local_mean"] = gamma_mean
        metrics["gamma_local_std"] = gamma_std

        if bus is not None:
            bus.emit("memory_evolved", {
                "mass_total": M_new.sum().item(),
                "gamma_local_mean": gamma_mean,
            })

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)
