"""ChargeDynamicsOperator — emergent electromagnetic-like forces.

Charge is NOT a new field. It emerges from the existing E and I fields:
    Q = ∂E/∂u - ∂I/∂v   (antisymmetric cross-gradient)

This is the natural "handedness" of the energy-information field — how E and I
break symmetry in orthogonal directions. It's conserved by the RBF equation
because the Laplacian preserves curl structure.

MECHANISM (no hardcoded Coulomb's law):
1. Charge Q is computed from the E, I gradients at each cell.
2. Q creates a POTENTIAL via Poisson: ∇²Φ_Q = Q (like the gravity potential).
3. The gradient ∇Φ_Q is the "electric field" — the force on charges.
4. The force on each cell is: F = Q_local · ∇Φ_Q
   - Like charges (same sign Q near same sign Φ_Q gradient) → repulsion
   - Unlike charges → attraction
5. This force acts on the ENERGY FIELD, not mass directly.
   Charge forces move energy, which then creates disequilibrium,
   which then affects mass generation. The causal chain is:
   charge → energy redistribution → disequilibrium → mass dynamics.

The coupling strength is NOT hardcoded. It emerges as:
    e_local = |Q| / (|Q| + |S| + ε)
where S is spin. This is the charge-dominance ratio — how much of the
cell's angular structure is charge vs spin. It's analogous to α_local
(field dominance) and G_local (mass dominance) in other operators.

Expected emergent behavior:
- Atoms: bound E-I structures with charge balance (neutral)
- Molecules: charge-mediated binding between structures
- Plasma: hot regions where charge separates (pre-recombination)
- 1/r² force law emerges from the Laplacian Green's function
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


_EPS = 1e-12


class ChargeDynamicsOperator:
    """Emergent EM-like forces from charge field Q = ∂E/∂u - ∂I/∂v.

    Like charges repel, unlike attract — through Poisson potential,
    not hardcoded Coulomb. Coupling strength emerges per cell.
    """

    def __init__(self) -> None:
        self._manifold: Optional[MobiusManifold] = None
        self._inv_lap: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "charge_dynamics"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu or self._manifold.nv != nv:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    def _solve_poisson(self, source: torch.Tensor) -> torch.Tensor:
        """Spectral Poisson solver: ∇²Φ = source → Φ via FFT."""
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

        # --- Emergent charge field ---
        # Q = ∂E/∂u - ∂I/∂v (cross-gradient: E-I handedness)
        dE_du = (torch.roll(E, -1, 0) - torch.roll(E, 1, 0)) / 2.0
        dI_dv = (torch.roll(I, -1, 1) - torch.roll(I, 1, 1)) / 2.0
        Q = dE_du - dI_dv

        # --- Spin field (for coupling ratio) ---
        diseq = E - I
        d_du = (torch.roll(diseq, -1, 0) - torch.roll(diseq, 1, 0)) / 2.0
        d_dv = (torch.roll(diseq, -1, 1) - torch.roll(diseq, 1, 1)) / 2.0
        S = d_du - d_dv

        # --- Emergent coupling: charge dominance ---
        # e_local = Q² / (Q² + S² + ε) — how much angular structure is charge
        Q2 = Q.pow(2)
        S2 = S.pow(2)
        e_local = Q2 / (Q2 + S2 + _EPS)

        # --- Solve for charge potential ---
        # ∇²Φ_Q = Q → Φ_Q encodes the "electric potential"
        phi_Q = self._solve_poisson(Q)

        # --- Gradient of charge potential: the "electric field" ---
        E_field_u = (torch.roll(phi_Q, -1, 0) - torch.roll(phi_Q, 1, 0)) / 2.0
        E_field_v = (torch.roll(phi_Q, -1, 1) - torch.roll(phi_Q, 1, 1)) / 2.0

        # --- Force on energy field ---
        # F = e_local · Q · ∇Φ_Q → acts on E field
        # Like charges: Q and ∇Φ_Q have same sign → positive force → repel
        # Unlike charges: opposite signs → negative force → attract
        # The force creates disequilibrium which drives further evolution.
        force_u = e_local * Q * E_field_u
        force_v = e_local * Q * E_field_v

        # Divergence of force: how force accumulates/depletes energy locally
        div_force = (
            (torch.roll(force_u, -1, 0) - torch.roll(force_u, 1, 0)) / 2.0 +
            (torch.roll(force_v, -1, 1) - torch.roll(force_v, 1, 1)) / 2.0
        )

        # --- Add charge force to the balance field derivative ---
        # Instead of directly modifying E (which creates positive feedback:
        # larger E → larger Q → larger force → larger E), charge forces
        # contribute to dE_dt which goes through the actualization gate.
        # The MAR threshold naturally limits the perturbation, and the
        # PAC tree redistribution handles conservation.
        #
        # This is also more physically correct: charge forces create
        # disequilibrium (potential for change), not instant field changes.
        dE_dt_existing = state.metrics.get("dE_dt", torch.zeros_like(E))
        dE_dt_new = dE_dt_existing + div_force

        E_new = E  # fields unchanged — force goes through dE_dt
        I_new = I

        # --- Metrics ---
        metrics = dict(state.metrics)
        metrics["dE_dt"] = dE_dt_new  # updated balance field with charge contribution
        metrics["charge_mean"] = Q.mean().item()
        metrics["charge_std"] = Q.std().item()
        metrics["charge_abs_mean"] = Q.abs().mean().item()
        metrics["e_local_mean"] = e_local.mean().item()
        metrics["charge_potential_max"] = phi_Q.abs().max().item()
        metrics["charge_force_mean"] = div_force.abs().mean().item()

        if bus is not None:
            bus.emit("charge_dynamics_applied", {
                "charge_abs_mean": metrics["charge_abs_mean"],
                "e_local_mean": metrics["e_local_mean"],
                "charge_force": metrics["charge_force_mean"],
            })

        return state.replace(E=E_new, I=I_new, metrics=metrics)
