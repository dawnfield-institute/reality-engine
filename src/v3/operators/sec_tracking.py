"""SECTrackingOperator — thermodynamic observer using fracton's SECFieldEvolver.

Read-only operator that computes SEC energy functional components and field
entropy without modifying any fields. Provides thermodynamic metrics for
downstream analysis and dashboard visualization.

Maps RE fields to SEC fields: A=E (actual), P=I (potential), T=T (temperature).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.v3.engine.state import FieldState
from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus

try:
    from fracton.field import SECFieldEvolver
    _HAS_FRACTON = True
except ImportError:
    _HAS_FRACTON = False

_EPS = 1e-12


class SECTrackingOperator:
    """Read-only SEC thermodynamic observer.

    Uses fracton's SECFieldEvolver to compute energy functional components
    and tracks field entropy. Does NOT modify state fields.
    """

    def __init__(self) -> None:
        self._evolver: Optional[object] = None
        self._prev_entropy: Optional[float] = None

    @property
    def name(self) -> str:
        return "sec_tracking"

    def _get_evolver(self, device: torch.device):
        if not _HAS_FRACTON:
            return None
        if self._evolver is None:
            self._evolver = SECFieldEvolver(
                alpha=0.1, beta=0.05, gamma=0.01, device=device,
            )
        return self._evolver

    @torch.no_grad()
    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        metrics = dict(state.metrics)

        # --- Field entropy ---
        # S = -Σ p·log(p) where p = normalized |E|² distribution
        E_sq = state.E.pow(2).flatten()
        E_total = E_sq.sum()
        if E_total > _EPS:
            p = E_sq / E_total
            p_safe = torch.clamp(p, min=_EPS)
            entropy = -(p_safe * p_safe.log()).sum().item()
        else:
            entropy = 0.0

        metrics["field_entropy"] = entropy

        if self._prev_entropy is not None:
            metrics["entropy_reduction_rate"] = self._prev_entropy - entropy
        else:
            metrics["entropy_reduction_rate"] = 0.0
        self._prev_entropy = entropy

        # --- SEC energy functional (via fracton) ---
        evolver = self._get_evolver(state.device)
        if evolver is not None:
            energy = evolver.compute_energy(state.E, state.I, state.T)
            metrics["sec_energy_total"] = energy["total"]
            metrics["sec_coupling"] = energy["coupling"]
            metrics["sec_smoothness"] = energy["smoothness"]
            metrics["sec_thermal"] = energy["thermal"]

            # Collapse region detection
            collapse_mask = evolver.detect_collapse_regions(state.E, threshold=0.1)
            collapse_frac = collapse_mask.float().mean().item()
            metrics["collapse_fraction"] = collapse_frac

        if bus is not None:
            bus.emit("sec_tracking", {
                "field_entropy": entropy,
                "entropy_reduction_rate": metrics["entropy_reduction_rate"],
                "sec_energy_total": metrics.get("sec_energy_total", 0),
                "collapse_fraction": metrics.get("collapse_fraction", 0),
            })

        return state.replace(metrics=metrics)
