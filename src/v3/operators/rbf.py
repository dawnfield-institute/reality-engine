"""RBFOperator — the Recursive Balance Field.

THE fundamental equation of Dawn Field Theory:
    B(x,t) = ∇²(E-I) + λ_local·M·∇²M - α_local·||E-I||² - γ·(E-I)

This ONE equation generates all physics. Energy follows the balance field.
The operator computes dE/dt = B and stores it in state.metrics for the integrator.

The coupling constants α and λ are NOT hardcoded — they EMERGE per cell:
    α_local = (E² + I²) / (E² + I² + M² + ε)  — field dominance ratio
    λ_local = M² / (E² + I² + M² + ε)          — memory dominance ratio
Note: α_local + λ_local ≈ 1 (they partition the total field energy).
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


_EPS = 1e-12


class RBFOperator:
    """Compute the Recursive Balance Field and store dE/dt in metrics.

    α_local and λ_local emerge per cell from field energy partition.
    The system finds its own coupling strengths through PAC dynamics.
    """

    def __init__(self) -> None:
        self._manifold: Optional[MobiusManifold] = None

    @property
    def name(self) -> str:
        return "rbf"

    def _get_manifold(self, state: FieldState, config: SimulationConfig) -> MobiusManifold:
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
        m = self._get_manifold(state, config)
        E, I, M = state.E, state.I, state.M

        disequilibrium = E - I

        # --- Emergent coupling constants per cell ---
        # Partition total field energy: α + λ ≈ 1
        E2 = E.pow(2)
        I2 = I.pow(2)
        M2 = M.pow(2)
        total = E2 + I2 + M2 + _EPS

        # α_local: field (E,I) dominance → controls collapse attraction
        alpha_local = (E2 + I2) / total

        # λ_local: memory (M) dominance → controls memory-mediated coupling
        lambda_local = M2 / total

        # ∇²(E - I)
        lap_diseq = m.laplacian(disequilibrium)

        # λ_local·M·∇²M  (emergent memory-mediated coupling)
        lap_M = m.laplacian(M)
        memory_coupling = lambda_local * M * lap_M

        # -α_local·||E-I||²  (emergent collapse attraction)
        collapse = -alpha_local * disequilibrium.pow(2)

        # -γ·(E-I)  (damping — γ from config, updated by AdaptiveOperator)
        damping = -config.gamma_damping * disequilibrium

        # B = ∇²(E-I) + λ_local·M·∇²M - α_local·||E-I||² - γ·(E-I)
        B = lap_diseq + memory_coupling + collapse + damping

        # Track emergent couplings
        alpha_mean = alpha_local.mean().item()
        lambda_mean = lambda_local.mean().item()

        metrics = dict(state.metrics)
        metrics["dE_dt"] = B
        metrics["balance_magnitude"] = B.abs().mean().item()
        metrics["alpha_local_mean"] = alpha_mean
        metrics["alpha_local_std"] = alpha_local.std().item()
        metrics["lambda_local_mean"] = lambda_mean
        metrics["lambda_local_std"] = lambda_local.std().item()

        if bus is not None:
            bus.emit("rbf_computed", {
                "balance_mag": metrics["balance_magnitude"],
                "alpha_local_mean": alpha_mean,
                "lambda_local_mean": lambda_mean,
            })

        return state.replace(metrics=metrics)
