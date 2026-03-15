"""RBFOperator — the Recursive Balance Field.

THE fundamental equation of Dawn Field Theory:
    B(x,t) = ∇²(E-I) + λ·M·∇²M - α·||E-I||² - γ·(E-I)

This ONE equation generates all physics. Energy follows the balance field.
The operator computes dE/dt = B and stores it in state.metrics for the integrator.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.substrate.manifold import MobiusManifold


class RBFOperator:
    """Compute the Recursive Balance Field and store dE/dt in metrics."""

    def __init__(self, gamma_damping: float = 0.005) -> None:
        self.gamma = gamma_damping
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

        # ∇²(E - I)
        lap_diseq = m.laplacian(disequilibrium)

        # λ·M·∇²M  (memory-mediated coupling)
        lap_M = m.laplacian(M)
        memory_coupling = config.lambda_freq * M * lap_M

        # -α·||E-I||²  (collapse attraction)
        collapse = -config.alpha_pac * disequilibrium.pow(2)

        # -γ·(E-I)  (damping)
        damping = -self.gamma * disequilibrium

        # B = ∇²(E-I) + λM∇²M - α||E-I||² - γ(E-I)
        B = lap_diseq + memory_coupling + collapse + damping

        metrics = dict(state.metrics)
        metrics["dE_dt"] = B
        metrics["balance_magnitude"] = B.abs().mean().item()

        if bus is not None:
            bus.emit("rbf_computed", {"balance_mag": metrics["balance_magnitude"]})

        return state.replace(metrics=metrics)
