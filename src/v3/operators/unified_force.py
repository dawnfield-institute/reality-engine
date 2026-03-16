"""UnifiedForceOperator — gravity + EM from single pre-field projection.

DFT proves that gravity and EM are the SAME pre-field projected differently:
- Symmetric projection (amplitude) → gravitational potential
- Antisymmetric projection (phase) → electromagnetic field

This replaces separate GravitationalCollapseOperator + ChargeDynamicsOperator
with a single operator that derives both forces from the pre-field
torch.stack([E, I, M], dim=0).

The emergent gravity/EM energy ratio is tracked as a metric — DFT predicts
it should converge to a constant related to the fine structure constant.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.projections import (
    project_symmetric_2d,
    project_antisymmetric_2d,
    gradient_2d,
    curl_2d,
    divergence_2d,
)


_EPS = 1e-12


class UnifiedForceOperator:
    """Unified gravity + EM from pre-field symmetric/antisymmetric decomposition.

    Constructs pre-field from (E, I, M), projects to gravity potential and EM
    field, then routes forces through the same mechanisms as the separate
    operators: gravity → mass redistribution, EM → dE_dt contribution.

    Tracks emergent coupling ratio R = grav_energy / em_energy.
    """

    def __init__(self) -> None:
        self._inv_lap: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "unified_force"

    def _solve_poisson(self, source: torch.Tensor) -> torch.Tensor:
        """Spectral Poisson solver: nabla^2 Phi = source via FFT."""
        nu, nv = source.shape

        if self._inv_lap is None or self._inv_lap.shape != (nu, nv):
            ku = torch.arange(nu, device=source.device, dtype=torch.float64)
            kv = torch.arange(nv, device=source.device, dtype=torch.float64)
            ku_grid, kv_grid = torch.meshgrid(ku, kv, indexing='ij')
            eigenvalues = (
                2.0 * (torch.cos(2.0 * torch.pi * ku_grid / nu) - 1.0) +
                2.0 * (torch.cos(2.0 * torch.pi * kv_grid / nv) - 1.0)
            )
            eigenvalues[0, 0] = 1.0
            inv_lap = 1.0 / eigenvalues
            inv_lap[0, 0] = 0.0
            self._inv_lap = inv_lap

        source_hat = torch.fft.fft2(source)
        phi_hat = source_hat * self._inv_lap
        return torch.fft.ifft2(phi_hat).real

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        # ================================================================
        # Pre-field construction and projection
        # ================================================================
        prefield = torch.stack([E, I, M], dim=0)  # (3, nu, nv)

        # Symmetric → gravity (amplitude envelope)
        grav_potential_raw = project_symmetric_2d(prefield)  # (nu, nv)

        # Antisymmetric → EM (phase structure)
        em_u, em_v = project_antisymmetric_2d(prefield)  # each (nu, nv)

        # ================================================================
        # GRAVITY: potential → mass redistribution
        # ================================================================
        # Solve Poisson on the symmetric projection to get smooth potential
        grav_potential = self._solve_poisson(grav_potential_raw)

        # Emergent G_local (same as GravitationalCollapseOperator)
        M2 = M.pow(2)
        diseq2 = (E - I).pow(2)
        G_local = M2 / (M2 + diseq2 + _EPS)

        # Mass flux: J = M * grad(Phi_grav)
        grad_grav_u, grad_grav_v = gradient_2d(grav_potential)
        flux_u = M * grad_grav_u
        flux_v = M * grad_grav_v

        # Divergence of flux
        div_flux = divergence_2d(flux_u, flux_v)

        # dM_grav with emergent coupling
        dM_grav = G_local * div_flux * dt

        # Apply with floor clamp
        M_candidate = M + dM_grav
        M_new = torch.clamp(M_candidate, min=0.0)

        # PAC correction for clamp
        mass_created = M_new - M_candidate
        pac_leak = mass_created * 0.5
        E_new = E - pac_leak
        I_new = I - pac_leak

        # ================================================================
        # EM: antisymmetric field → charge force → dE_dt
        # ================================================================
        # Solve Poisson on EM magnitude to get charge potential
        em_magnitude = (em_u.pow(2) + em_v.pow(2)).sqrt()
        phi_em = self._solve_poisson(em_magnitude)

        # EM force: curl of the EM field (circulation)
        em_curl = curl_2d(em_u, em_v)

        # Emergent charge coupling (analogous to e_local in ChargeDynamics)
        em_energy = em_magnitude.pow(2)
        grav_energy = grav_potential_raw.pow(2)
        e_local = em_energy / (em_energy + grav_energy + _EPS)

        # Force on energy field via curl-driven circulation
        force_u = e_local * em_curl * em_u
        force_v = e_local * em_curl * em_v

        # Divergence of force → dE_dt contribution
        div_em_force = divergence_2d(force_u, force_v)

        # Route through dE_dt (actualization gate limits perturbation)
        dE_dt_existing = state.metrics.get("dE_dt", torch.zeros_like(E))
        dE_dt_new = dE_dt_existing + div_em_force

        # ================================================================
        # Metrics
        # ================================================================
        grav_total = grav_energy.sum().item()
        em_total = em_energy.sum().item()
        coupling_ratio = grav_total / (em_total + _EPS)

        metrics = dict(state.metrics)
        metrics["dE_dt"] = dE_dt_new
        metrics["gravitational_potential_max"] = grav_potential.abs().max().item()
        metrics["G_local_mean"] = G_local.mean().item()
        metrics["G_local_std"] = G_local.std().item()
        metrics["em_energy_total"] = em_total
        metrics["grav_energy_total"] = grav_total
        metrics["grav_em_ratio"] = coupling_ratio
        metrics["e_local_mean"] = e_local.mean().item()
        metrics["charge_force_mean"] = div_em_force.abs().mean().item()

        if bus is not None:
            bus.emit("unified_force_applied", {
                "G_local_mean": metrics["G_local_mean"],
                "grav_em_ratio": coupling_ratio,
                "em_energy": em_total,
                "grav_energy": grav_total,
                "charge_force": metrics["charge_force_mean"],
            })

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)
