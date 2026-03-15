"""GravitationalCollapseOperator — mass attracts mass.

Without self-gravity, mass just diffuses. This operator solves for the
gravitational potential Φ where ∇²Φ = M, then moves mass down the
potential gradient:

    dM_grav = G · ∇·(M · ∇Φ)

This is the Poisson-advection equation — mass flows toward mass
concentrations, forming gravity wells that can collapse into stars.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


class GravitationalCollapseOperator:
    """Self-gravity: mass attracts mass via Poisson equation."""

    def __init__(self, G: float = 0.05, iterations: int = 20) -> None:
        self.G = G
        self.iterations = iterations  # Jacobi iterations for Poisson solve
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "gravity"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu or self._manifold.nv != nv:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    def _solve_poisson(self, source: torch.Tensor) -> torch.Tensor:
        """Approximate Poisson solve ∇²Φ = source via Jacobi iteration.

        Uses the 5-point stencil: Φ_new = (neighbors + source) / 4
        """
        phi = torch.zeros_like(source)
        for _ in range(self.iterations):
            neighbors = (
                torch.roll(phi, 1, 0) + torch.roll(phi, -1, 0) +
                torch.roll(phi, 1, 1) + torch.roll(phi, -1, 1)
            )
            phi = (neighbors + source) / 4.0
        return phi

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        M = state.M
        dt = config.dt

        # Solve ∇²Φ = M for gravitational potential
        phi = self._solve_poisson(M)

        # Gradient of potential: ∇Φ (force direction)
        grad_phi_u = (torch.roll(phi, -1, 0) - torch.roll(phi, 1, 0)) / 2.0
        grad_phi_v = (torch.roll(phi, -1, 1) - torch.roll(phi, 1, 1)) / 2.0

        # Mass flux: J = M · ∇Φ
        flux_u = M * grad_phi_u
        flux_v = M * grad_phi_v

        # Divergence of flux: ∇·J = dJu/du + dJv/dv
        div_flux = (
            (torch.roll(flux_u, -1, 0) - torch.roll(flux_u, 1, 0)) / 2.0 +
            (torch.roll(flux_v, -1, 1) - torch.roll(flux_v, 1, 1)) / 2.0
        )

        # dM_grav = G · ∇·(M · ∇Φ)
        dM_grav = self.G * div_flux * dt
        M_new = torch.clamp(M + dM_grav, min=0.0)

        metrics = dict(state.metrics)
        metrics["gravitational_potential_max"] = phi.max().item()

        if bus is not None:
            bus.emit("gravity_evolved", {"potential_max": phi.max().item()})

        return state.replace(M=M_new, metrics=metrics)
