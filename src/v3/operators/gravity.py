"""GravitationalCollapseOperator — mass attracts mass.

Without self-gravity, mass just diffuses. This operator solves for the
gravitational potential Φ where ∇²Φ = √M (amplitude coupling), then
moves mass down the potential gradient:

    dM_grav = G_local · ∇·(M · ∇Φ)

Three emergent modulations prevent runaway clumping:

1. Mass-dominance coupling: G_mass = M² / (M² + (E-I)² + ε)
2. Entropy-coherence balance: xi_mod = √(xi_s^(1/φ) / (xi_s^(1/φ) + 1))
   where xi_s = I²/E² (DFT infodynamic framework, exp_29/exp_36)
   The φ-scaling flattens the response around xi_s=1, preventing
   late-time overshoot as the system thermalizes.
3. Cascade-depth spectral tiling (exp_36): the Poisson solver is
   filtered in spectral domain so gravity is LOCAL-DOMINANT.
   Each mode k has a tiling suppression (ln²(2))^(Ξ·n(k)) where
   n(k) = log_φ(|k|_max/|k|) is the cascade depth. This is the
   SEC cost of tiling local PAC patches globally — the same
   mechanism that produces the cosmological constant.

G_local = G_mass * xi_mod — strongest where mass dominates AND coherence
dominates entropy. The sqrt Poisson source compresses dynamic range.
The spectral tiling filter makes gravity predominantly local, creating
web-like structure (filaments between voids) instead of global clumping.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


_EPS = 1e-12
_PHI = (1 + math.sqrt(5)) / 2
_LN_PHI = math.log(_PHI)
_LN2_SQ = math.log(2) ** 2                    # 0.4805 — tiling residual per level (exp_28)
_GAMMA_EM = 0.5772156649015329
_XI = _GAMMA_EM + _LN_PHI                     # 1.05843 — global frame attractor (exp_29)


class GravitationalCollapseOperator:
    """Self-gravity: mass attracts mass via Poisson equation.

    G_local = M² / (M² + (E-I)² + ε) — emergent per cell.
    Gravity is strong where mass dominates, weak where disequilibrium dominates.
    """

    def __init__(self, iterations: int = 0) -> None:
        self._iterations = iterations  # 0 = auto-scale with grid size
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "gravity"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu or self._manifold.nv != nv:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    def _build_inv_laplacian(self, nu: int, nv: int, device: torch.device) -> torch.Tensor:
        """Precompute -1/|k|² for the spectral Poisson solver.

        For the 5-point discrete Laplacian on a periodic grid, the eigenvalues are:
            λ(k,l) = 2(cos(2πk/nu) - 1) + 2(cos(2πl/nv) - 1)
        So Φ̂ = f̂ / λ  (avoiding division by zero at k=l=0).
        """
        ku = torch.arange(nu, device=device, dtype=torch.float64)
        kv = torch.arange(nv, device=device, dtype=torch.float64)
        ku_grid, kv_grid = torch.meshgrid(ku, kv, indexing='ij')

        eigenvalues = (
            2.0 * (torch.cos(2.0 * torch.pi * ku_grid / nu) - 1.0) +
            2.0 * (torch.cos(2.0 * torch.pi * kv_grid / nv) - 1.0)
        )
        # Avoid div by zero at (0,0) — set DC component to 1 (Φ̂(0,0) = 0)
        eigenvalues[0, 0] = 1.0
        inv_lap = 1.0 / eigenvalues
        inv_lap[0, 0] = 0.0  # Zero mean potential
        return inv_lap

    def _build_tiling_filter(self, nu: int, nv: int, device: torch.device) -> torch.Tensor:
        """Cascade-depth spectral tiling filter (exp_36).

        Each mode k has cascade depth n(k) = log_phi(|k|_max / |k|).
        Tiling suppression: (ln^2(2))^(Xi * n(k)).

        High |k| (local): filter ~ 1 (full gravity)
        Low |k| (global): filter ~ 0 (gravity suppressed by SEC tiling cost)
        """
        ku = torch.arange(nu, device=device, dtype=torch.float64)
        kv = torch.arange(nv, device=device, dtype=torch.float64)
        ku = torch.where(ku > nu // 2, ku - nu, ku)
        kv = torch.where(kv > nv // 2, kv - nv, kv)
        ku_grid, kv_grid = torch.meshgrid(ku, kv, indexing='ij')

        k_mag = torch.sqrt(ku_grid**2 + kv_grid**2)
        k_max = math.sqrt((nu // 2)**2 + (nv // 2)**2)

        k_safe = torch.clamp(k_mag, min=1.0)
        cascade_depth = torch.log(k_max / k_safe) / _LN_PHI

        log_suppression = _XI * cascade_depth * math.log(_LN2_SQ)
        tiling_filter = torch.exp(log_suppression)
        tiling_filter[0, 0] = 0.0
        return tiling_filter

    def _solve_poisson(self, source: torch.Tensor) -> torch.Tensor:
        """Spectral Poisson solver with cascade-depth tiling filter.

        Exact solution for ∇²Φ = source on a periodic domain, filtered
        so that long-range (low-k) modes are suppressed by the SEC
        tiling cost. O(N log N) — no iterations needed.
        """
        nu, nv = source.shape

        # Lazy-init the inverse Laplacian kernel (cached for reuse)
        if not hasattr(self, '_inv_lap') or self._inv_lap.shape != (nu, nv):
            self._inv_lap = self._build_inv_laplacian(nu, nv, source.device)
        if not hasattr(self, '_tiling_filter') or self._tiling_filter.shape != (nu, nv):
            self._tiling_filter = self._build_tiling_filter(nu, nv, source.device)

        # FFT → divide by eigenvalues → apply tiling filter → IFFT
        source_hat = torch.fft.fft2(source)
        phi_hat = source_hat * self._inv_lap * self._tiling_filter
        phi = torch.fft.ifft2(phi_hat).real
        return phi

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        E, I, M = state.E, state.I, state.M
        dt = config.dt

        # --- Emergent gravitational coupling per cell ---
        # Two factors:
        # 1. Mass dominance: G_mass = M² / (M² + (E-I)² + ε)
        # 2. Entropy-coherence balance: xi_s = I² / E² (DFT infodynamic framework)
        #    Gravity is strong where coherence (I) dominates entropy (E),
        #    weakened where entropy dominates (thermalized dense regions).
        M2 = M.pow(2)
        diseq2 = (E - I).pow(2)
        G_mass = M2 / (M2 + diseq2 + _EPS)

        # Entropy-coherence modulation (exp_29 global-local duality, exp_36 tiling)
        E2 = E.pow(2)
        I2 = I.pow(2)
        xi_s = I2 / (E2 + _EPS)
        # phi-scaled sigmoid: xi_s^(1/phi) flattens response around xi_s=1,
        # preventing late-time overshoot as system thermalizes (exp_36 spike)
        xi_s_phi = xi_s.pow(1.0 / _PHI)
        xi_mod = torch.sqrt(xi_s_phi / (xi_s_phi + 1.0))

        G_local = G_mass * xi_mod

        # Solve ∇²Φ = √M — amplitude coupling.
        # Gravity couples to field amplitude, not intensity: compresses
        # dynamic range so dense regions don't dominate the potential.
        phi = self._solve_poisson(torch.sqrt(M + _EPS))

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

        # dM_grav = G_local · ∇·(M · ∇Φ) — per-cell emergent coupling
        dM_grav = G_local * div_flux * dt

        # PAC-conserving clamp: if gravity would push M below zero,
        # limit the change to -M (can't extract more mass than exists).
        M_candidate = M + dM_grav
        M_new = torch.clamp(M_candidate, min=0.0)

        # Any mass created by the clamp is a PAC violation — drain from E+I.
        mass_created = (M_new - M_candidate)  # positive where clamp activated
        pac_leak = mass_created * 0.5  # split equally (QBE: E and I dual)
        E_new = state.E - pac_leak
        I_new = state.I - pac_leak

        # Track emergent G
        G_mean = G_local.mean().item()
        G_std = G_local.std().item()

        metrics = dict(state.metrics)
        metrics["gravitational_potential_max"] = phi.max().item()
        metrics["G_local_mean"] = G_mean
        metrics["G_local_std"] = G_std
        metrics["xi_s_mean"] = xi_s.mean().item()
        metrics["xi_s_std"] = xi_s.std().item()
        metrics["xi_mod_mean"] = xi_mod.mean().item()

        if bus is not None:
            bus.emit("gravity_evolved", {
                "potential_max": phi.max().item(),
                "G_local_mean": G_mean,
                "G_local_std": G_std,
            })

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)
