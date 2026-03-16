"""SpinStatisticsOperator — emergent Pauli exclusion from information dynamics.

THE INSIGHT: Pauli exclusion is an information-theoretic constraint. Two identical
fermions sharing a state means zero information — the system can't distinguish them.
In DFT, actualizing zero-information states costs infinite PAC potential per bit
(Landauer bound). So identical half-integer spin states *resist* co-location.

MECHANISM (no hardcoded exclusion):
1. Spin emerges from the curl of disequilibrium: S = curl(E-I).
   On a 2D Möbius manifold, this is a pseudoscalar.
2. Half-integer spin: cells where |S| is near (n+1/2) for integer n.
3. State similarity: multi-scale — neighbors at distances 1,2,4,8 with Fibonacci
   decay weights (PAC tree style). Extends exclusion correlation length.
4. Degeneracy pressure: where state similarity is high AND spin is half-integer,
   the MEMORY FIELD experiences enhanced diffusion (mass spreads out from peaks).

The degeneracy level (sigma * f_half * rho_sim) is exported via metrics so the
memory operator can throttle mass generation at degenerate sites — creating
genuinely emergent mass limits without the hard cap.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


_EPS = 1e-12
_PHI = 1.618033988749895


def _half_integer_proximity(S: torch.Tensor) -> torch.Tensor:
    """How close |S| is to a half-integer value (0.5, 1.5, 2.5, ...).

    Returns 1.0 when S is exactly half-integer, 0.0 when integer.
    Uses cos^2(pi*S) which peaks at half-integers and zeros at integers.
    """
    return torch.cos(math.pi * S).pow(2)


def _state_similarity_density(
    M: torch.Tensor,
    Q: torch.Tensor,
    S: torch.Tensor,
    scales: int = 4,
) -> torch.Tensor:
    """Multi-scale density of similar quantum states — PAC tree style.

    Checks neighbors at increasing distances (1, 2, 4, 8 cells) with
    Fibonacci decay weights (1, 1/phi, 1/phi^2, 1/phi^3). This extends the
    correlation length of the exclusion pressure beyond nearest neighbors.

    In real QM, Pauli exclusion acts over the entire wavefunction extent.
    Multi-scale sampling approximates this: O(N*scales) not O(N^2).

    Returns a value in [0, 1] where 1 = all neighbors identical at all scales.
    """
    total_sim = torch.zeros_like(M)
    total_weight = 0.0

    norm = M.pow(2) + _EPS  # relative similarity normalization

    for k in range(scales):
        dist = 1 << k  # 1, 2, 4, 8
        weight = 1.0 / (_PHI ** k)  # Fibonacci decay
        total_weight += weight * 4  # 4 neighbors per scale

        for shift, dim in [(-dist, 0), (dist, 0), (-dist, 1), (dist, 1)]:
            M_n = torch.roll(M, shift, dim)
            Q_n = torch.roll(Q, shift, dim)
            S_n = torch.roll(S, shift, dim)

            dM = (M - M_n).pow(2)
            dQ = (Q - Q_n).pow(2)
            dS = (S - S_n).pow(2)

            dist2 = (dM + dQ + dS) / norm
            total_sim += weight * torch.exp(-dist2)

    return total_sim / total_weight


class SpinStatisticsOperator:
    """Emergent Pauli exclusion: half-integer spin states resist co-location.

    Creates degeneracy pressure that stabilizes intermediate mass structures,
    preventing universal gravitational collapse to the mass cap.

    Physics emerges from information cost — NOT hardcoded exclusion rules.

    Exports ``degeneracy_level`` tensor via metrics for downstream operators
    (memory operator uses it to throttle mass generation at degenerate sites).
    """

    def __init__(self) -> None:
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "spin_statistics"

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

        # --- Emergent spin from curl of disequilibrium ---
        diseq = E - I
        d_du = (torch.roll(diseq, -1, 0) - torch.roll(diseq, 1, 0)) / 2.0
        d_dv = (torch.roll(diseq, -1, 1) - torch.roll(diseq, 1, 1)) / 2.0
        S = d_du - d_dv  # curl in 2D -> pseudoscalar spin

        # --- Emergent charge from antisymmetric gradient ---
        dE_du = (torch.roll(E, -1, 0) - torch.roll(E, 1, 0)) / 2.0
        dI_dv = (torch.roll(I, -1, 1) - torch.roll(I, 1, 1)) / 2.0
        Q = dE_du - dI_dv

        # --- Half-integer proximity ---
        f_half = _half_integer_proximity(S)

        # --- Multi-scale state similarity density ---
        rho_sim = _state_similarity_density(M, Q, S)

        # --- Emergent degeneracy pressure ---
        # sigma_local = M^2 / (E^2 + I^2 + M^2 + eps) -- mass dominance
        total_field2 = E.pow(2) + I.pow(2) + M.pow(2) + _EPS
        sigma_local = M.pow(2) / total_field2

        # Degeneracy level = composite exclusion strength [0,1]
        degeneracy_level = sigma_local * f_half * rho_sim

        # Enhanced diffusion: Laplacian pushes mass OUT of peaks at degenerate sites
        lap_M = m.laplacian(M)
        degen_force = degeneracy_level * lap_M

        # Apply
        dM_degen = degen_force * dt
        M_new = M + dM_degen
        M_new = torch.clamp(M_new, min=0.0)
        dM_actual = M_new - M

        # PAC conservation: drain from E and I equally
        pac_drain = dM_actual * 0.5
        E_new = E - pac_drain
        I_new = I - pac_drain

        # --- Metrics ---
        metrics = dict(state.metrics)
        metrics["spin_field_mean"] = S.mean().item()
        metrics["spin_field_std"] = S.std().item()
        metrics["spin_half_integer_fraction"] = (f_half > 0.5).float().mean().item()
        metrics["charge_field_mean"] = Q.mean().item()
        metrics["charge_field_std"] = Q.std().item()
        metrics["state_similarity_mean"] = rho_sim.mean().item()
        metrics["degeneracy_pressure_mean"] = degen_force.abs().mean().item()
        metrics["degeneracy_level_mean"] = degeneracy_level.mean().item()
        metrics["sigma_local_mean"] = sigma_local.mean().item()
        # Export the full degeneracy_level tensor for downstream operators
        metrics["degeneracy_level"] = degeneracy_level

        if bus is not None:
            bus.emit("spin_statistics_applied", {
                "half_integer_fraction": metrics["spin_half_integer_fraction"],
                "degeneracy_pressure": metrics["degeneracy_pressure_mean"],
                "degeneracy_level": metrics["degeneracy_level_mean"],
                "state_similarity": metrics["state_similarity_mean"],
            })

        return state.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)
