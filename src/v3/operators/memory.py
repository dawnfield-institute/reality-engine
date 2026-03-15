"""MemoryOperator — mass generation, quantum pressure, and diffusion.

Memory field M accumulates where information collapses (E-I disequilibrium).
Quantum pressure prevents uniform collapse. Diffusion redistributes mass.

    dM/dt = mass_generation + quantum_pressure + diffusion
    mass_generation = γ · α · (E-I)² · saturation
    quantum_pressure = -β · ∇²(M²) / M
    diffusion = D · ∇²M
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


class MemoryOperator:
    """Evolve the memory (mass) field from collapse dynamics."""

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

        # Mass generation: γ · α · (E-I)² · saturation
        mass_gen = config.mass_gen_coeff * config.alpha_pac * disequilibrium.pow(2)
        M_mean = M.mean() + 1e-10
        saturation = 1.0 / (1.0 + M / M_mean)
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
        M_new = M + dt * dM_dt
        M_new = torch.clamp(M_new, min=0.0)  # Mass is non-negative

        metrics = dict(state.metrics)
        metrics["mass_generation_rate"] = mass_gen.mean().item()

        if bus is not None:
            bus.emit("memory_evolved", {"mass_total": M_new.sum().item()})

        return state.replace(M=M_new, metrics=metrics)
