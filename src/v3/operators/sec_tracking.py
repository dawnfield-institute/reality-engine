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

    # DFT Tier 1 attractor targets (derived constants, not tuneable)
    _TIER1_TARGETS = {
        "f_local_mean": 0.5772156649015329,   # gamma_EM
        "gamma_local_mean": 0.6180339887,      # 1/phi
        "alpha_local_mean": 0.6931471805599453, # ln(2)
        "G_local_mean": 0.3819660112,          # 1/phi^2
        "lambda_local_mean": 0.3068528194,     # 1 - ln(2)
    }
    _K_EQ = 2  # cascade depth at equilibrium

    def __init__(self) -> None:
        self._evolver: Optional[object] = None
        self._prev_entropy: Optional[float] = None
        self._initial_entropy: Optional[float] = None
        self._min_coupling_error: float = float('inf')
        self._tick_now_estimate: int = 1

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

        # Cumulative: initial_entropy - current_entropy (positive = net structure formed)
        if self._initial_entropy is None:
            self._initial_entropy = entropy
        metrics["entropy_reduction_cumulative"] = self._initial_entropy - entropy

        self._prev_entropy = entropy

        # --- Info fraction (best SEC duty cycle proxy, r=+0.954 with theory) ---
        E_abs = state.E.abs().mean().item()
        I_abs = state.I.abs().mean().item()
        metrics["info_fraction"] = I_abs / (E_abs + I_abs + 1e-30)

        # --- Log-time cascade depth (spike 11: r=+0.954 with DFT theory) ---
        tick = state.tick
        if tick > 0:
            # Running estimate of NOW tick from minimum Tier 1 coupling error
            avg_err = 0.0
            count = 0
            for key, target in self._TIER1_TARGETS.items():
                val = metrics.get(key)
                if val is not None:
                    avg_err += abs(val - target) / (abs(target) + _EPS)
                    count += 1
            if count > 0:
                avg_err /= count
                if avg_err < self._min_coupling_error:
                    self._min_coupling_error = avg_err
                    self._tick_now_estimate = tick

            tick_now = max(self._tick_now_estimate, 2)
            cascade_depth = self._K_EQ * math.log(max(tick, 1)) / math.log(tick_now)
            metrics["cascade_depth"] = cascade_depth
            metrics["tick_now_estimate"] = self._tick_now_estimate

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
