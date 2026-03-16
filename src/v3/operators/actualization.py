"""ActualizationOperator — gated field integration via MAR threshold.

Replaces the naive Euler integrator with a physically-motivated actualization
gate derived from Minimum Actualization Resolution (MAR) theory.

The key insight: field changes don't continuously apply. They accumulate as
*potential* (quantum superposition) until they cross the MAR threshold, then
actualize as a discrete collapse event with two simultaneous mechanisms:

1. LOCAL (Landauer): f_local fraction of the change applies at the cell.
   This is the Landauer cost — actualizing information has an energy price.

2. GLOBAL (Entanglement): (1-f_local) fraction redistributes through PAC tree
   structure. Nearby cells get more than distant ones (Fibonacci decay).
   This creates NEW disequilibrium at other cells — the feedback loop that
   prevents heat death and drives continued evolution.

The local fraction f_local is NOT hardcoded — it EMERGES per cell from the
local actualization state: f = E²/(E²+I²). This is the ratio of actualized
to total field energy at each point. The system-wide mean of f converges to
ln(φ) ≈ 0.4812 as an ATTRACTOR, not a parameter. Locally it fluctuates —
the field looks like it's bubbling/boiling as PAC redistributes potential.

ln(φ) = A/(A+ξ) was validated across 11 MAR experiments as the theoretical
attractor. Here we let it emerge from dynamics and verify convergence.

PAC conservation: every E change is mirrored by -change in I (QBE duality).
Local: E += f*P, I -= f*P → net E+I change = 0.
Global: E += R, I -= R → net E+I change = 0.
Total: PAC = E + I + M unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


# ln(φ) — the theoretical attractor for the actualization ratio.
# The per-cell emergent ratio f = E²/(E²+I²) should converge to this mean.
LN_PHI = math.log((1 + math.sqrt(5)) / 2)  # 0.48121182505960344

# Small epsilon to prevent division by zero in f_local computation
_EPS = 1e-12


class ActualizationOperator:
    """Gated field integration with MAR threshold and emergent local/global split.

    Replaces EulerIntegrator. Reads dE/dt and dI/dt from metrics, accumulates
    potential P, and actualizes when |P| crosses the MAR threshold.

    The local fraction f = E²/(E²+I²) emerges per cell from the field state.
    The mean of f across the field converges to ln(φ) as an attractor —
    locally it fluctuates (bubbling/boiling) as PAC redistributes potential.
    """

    def __init__(self) -> None:
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "actualization"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.shape
        if self._manifold is None or self._manifold.nu != nu or self._manifold.nv != nv:
            self._manifold = MobiusManifold(nu, nv, device=state.device)
        return self._manifold

    def _pac_tree_redistribute(
        self,
        pool: torch.Tensor,
        manifold: MobiusManifold,
        depth: int = 6,
    ) -> torch.Tensor:
        """Redistribute field values through PAC tree structure.

        Uses iterated Laplacian diffusion with Fibonacci-weighted scales.
        Each level k applies one diffusion step (averaging with neighbors)
        and weights the result by 1/φ^k. This creates a multi-scale
        redistribution where nearby cells receive more than distant ones,
        following the PAC tree's Fibonacci decay.

        Uses the Möbius-aware Laplacian for correct boundary conditions.
        """
        PHI = (1 + math.sqrt(5)) / 2
        result = torch.zeros_like(pool)
        R = pool.clone()
        total_weight = 0.0

        for k in range(depth):
            weight = PHI ** (-(k + 1))  # 1/φ, 1/φ², 1/φ³, ...
            # One step of diffusion via Laplacian (spreads to neighbors)
            R = R + 0.25 * manifold.laplacian(R)
            result = result + weight * R
            total_weight += weight

        # Normalize to conserve total (prevent PAC leak)
        pool_sum = pool.sum()
        result_sum = result.sum()
        if abs(result_sum.item()) > 1e-10:
            result = result * (pool_sum / result_sum)

        return result

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        dt = config.dt
        dE_dt = state.metrics.get("dE_dt")
        dI_dt = state.metrics.get("dI_dt")

        # Fallback: if actualization disabled, behave like Euler integrator
        if not config.enable_actualization:
            E_new, I_new = state.E, state.I
            if dE_dt is not None:
                E_new = E_new + dt * dE_dt
            if dI_dt is not None:
                I_new = I_new + dt * dI_dt
            return state.replace(E=E_new, I=I_new)

        if dE_dt is None:
            return state

        # --- Accumulate potential ---
        # P tracks the balance field's "desire" for E to change.
        # Since QBE enforces dI = -dE, we only need one buffer.
        P = state.P + dt * dE_dt

        # --- Actualization gate ---
        # Cells where |P| exceeds the MAR threshold actualize.
        # Below threshold: potential remains latent (quantum superposition).
        #
        # Depth-dependent modulation (π/2-harmonic collapse):
        # If cascade_depth is available from PhiCascadeOperator, deeper
        # cascade levels require MORE potential to actualize:
        #   threshold_local = threshold_base × (π/2)^(depth/depth_scale)
        # This creates harmonic oscillator spacing: deeper structures
        # oscillate slower → E_n ∝ (n+1/2).
        base_threshold = config.actualization_threshold
        cascade_depth = state.metrics.get("cascade_depth", None)
        if cascade_depth is not None and isinstance(cascade_depth, torch.Tensor):
            # Scale depth modulation gently: (π/2)^(depth/4) so it's not too aggressive
            depth_mod = (math.pi / 2) ** (cascade_depth / 4.0)
            threshold = base_threshold * depth_mod
        else:
            threshold = base_threshold
        P_abs = P.abs()
        actualize_mask = P_abs > threshold

        n_actualized = actualize_mask.sum().item()

        if n_actualized == 0:
            # Nothing actualizes this tick — just accumulate potential
            metrics = dict(state.metrics)
            metrics["actualization_count"] = 0
            metrics["potential_mean"] = P_abs.mean().item()
            metrics["potential_max"] = P_abs.max().item()
            return state.replace(P=P, metrics=metrics)

        # --- Emergent local fraction per cell ---
        # f_local = E²/(E²+I²) — how actualized each cell is.
        # This is NOT hardcoded — it emerges from the field state.
        # The mean of f_local converges to ln(φ) as an attractor.
        # Locally it fluctuates — the field bubbles/boils.
        E2 = state.E.pow(2)
        I2 = state.I.pow(2)
        f_local = E2 / (E2 + I2 + _EPS)  # per-cell actualization ratio

        # --- Split actualized potential ---
        # Extract the potential at actualizing cells
        P_actual = torch.where(actualize_mask, P, torch.zeros_like(P))

        # LOCAL fraction: f_local of P stays at the cell (Landauer cost)
        local_change = f_local * P_actual

        # GLOBAL fraction: (1-f_local) redistributes via PAC tree (entanglement)
        global_pool = (1.0 - f_local) * P_actual

        # PAC tree redistribution: Fibonacci-weighted multi-scale diffusion
        manifold = self._get_manifold(state)
        redistributed = self._pac_tree_redistribute(
            global_pool, manifold, depth=config.actualization_tree_depth,
        )

        # --- Apply changes (PAC-conserving) ---
        # Every E change is mirrored by -change in I (QBE duality).
        # This ensures E + I + M = const at every step.
        total_dE = local_change + redistributed
        E_new = state.E + total_dE
        I_new = state.I - total_dE  # QBE: dI = -dE

        # Reset potential at actualized cells, keep accumulating elsewhere
        P_new = torch.where(actualize_mask, torch.zeros_like(P), P)

        # --- Metrics: track emergent ratio ---
        # f_actual = mean of f_local at actualizing cells only
        if n_actualized > 0:
            f_at_actual = f_local[actualize_mask]
            f_mean = f_at_actual.mean().item()
            f_std = f_at_actual.std().item() if n_actualized > 1 else 0.0
        else:
            f_mean = f_local.mean().item()
            f_std = f_local.std().item()

        metrics = dict(state.metrics)
        metrics["actualization_count"] = int(n_actualized)
        metrics["actualization_local_total"] = local_change.abs().sum().item()
        metrics["actualization_global_total"] = global_pool.abs().sum().item()
        metrics["potential_mean"] = P_new.abs().mean().item()
        metrics["potential_max"] = P_new.abs().max().item()
        metrics["f_local_mean"] = f_mean      # emergent ratio — should → ln(φ)
        metrics["f_local_std"] = f_std         # variance — should stay > 0 (boiling)
        metrics["f_local_deviation"] = f_mean - LN_PHI  # distance from attractor

        if bus is not None and n_actualized > 0:
            bus.emit("actualization_event", {
                "cells_actualized": int(n_actualized),
                "local_energy": local_change.abs().sum().item(),
                "global_energy": global_pool.abs().sum().item(),
                "f_local_mean": f_mean,
                "f_local_std": f_std,
            })

        return state.replace(E=E_new, I=I_new, P=P_new, metrics=metrics)
