"""
RealityEngine — main simulation loop.

Composes substrate (Möbius manifold) with dynamics (SEC, PAC)
and analysis (spectral, diagnostics) into a single GPU-accelerated engine.

Architecture: the Möbius topology enters at THREE levels:
    1. Laplacian BCs   — diffusion respects the half-twist
    2. Spatial source   — the Möbius-natural antiperiodic mode (1/φ ratio)
    3. Reinforcement    — bidirectional spectral control via f_anti

The PAC cycle each timestep:
    1. Measure Ξ from current P spectrum
    2. SEC evolve P (continuous — topology through Laplacian + reinforcement)
    3. Actualize:  coherent P → A, dissolution → Θ
    4. Memory:     EMA accumulation
    5. Θ recycle:  collapse + dissolution → next SEC step
    6. PAC enforce: additive conservation P + A + M = const

KEY INSIGHT: P evolves CONTINUOUSLY via SEC.  The confluence operator
is used for ANALYSIS (spectral decomposition into symmetric/antisymmetric)
and for the REINFORCEMENT term (amplify/dampen f_anti).  Time does not
require overwriting P with C(A) — the topology-aware Laplacian provides
the temporal arrow through spatial boundary conditions.

"Potential expands and then contracts into necessity.
 Feedback loops are computation."
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .substrate.mobius import MobiusManifold
from .substrate.state import FieldState
from .substrate.constants import (
    ALPHA_REFERENCE,
    SEC_DEFAULTS,
    XI_REFERENCE,
)
from .dynamics.confluence import ConfluenceOperator
from .dynamics.sec import SECEvolver
from .dynamics.pac import PACTracker
from .analysis.spectral import SpectralAnalyzer
from .analysis.diagnostics import DiagnosticsMonitor


class RealityEngine:
    """GPU-accelerated PAC/SEC simulator on a Möbius manifold."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, config: dict[str, Any]) -> None:
        # --- Substrate ----------------------------------------------------
        mcfg = config.get("manifold", {})
        self.manifold = MobiusManifold(
            n_u=mcfg.get("n_u", 128),
            n_v=mcfg.get("n_v", 64),
            device=mcfg.get("device", "cuda"),
        )

        # --- Dynamics -----------------------------------------------------
        self.confluence = ConfluenceOperator()
        self.sec = SECEvolver(
            self.manifold,
            config.get("sec", SEC_DEFAULTS),
            confluence=self.confluence,
        )

        # --- Analysis -----------------------------------------------------
        self.spectral = SpectralAnalyzer(self.confluence)
        self.diagnostics = DiagnosticsMonitor(
            xi_target=XI_REFERENCE,
        )

        # --- State --------------------------------------------------------
        init_cfg = config.get("init", {})
        self.state = self._initialize(
            seed=init_cfg.get("seed", 42),
            P_mean=init_cfg.get("P_mean", 0.5),
            P_noise=init_cfg.get("P_noise", 0.01),
            A_mean=init_cfg.get("A_mean", 0.5),
            A_noise=init_cfg.get("A_noise", 0.01),
            antiperiodic_amp=init_cfg.get("antiperiodic_amp", 0.05),
        )
        self.pac = PACTracker(self.state)

        # --- Actualization tunables ---------------------------------------
        act_cfg = config.get("actualization", {})
        self._act_rate: float = act_cfg.get("rate", 0.1)
        self._act_scale: float = act_cfg.get("scale", 1.0)
        self._act_memory: float = act_cfg.get("memory", 0.9)

        # --- Θ recycling state --------------------------------------------
        self._theta: float = 0.0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialize(
        self,
        seed: int,
        P_mean: float,
        P_noise: float,
        A_mean: float,
        A_noise: float,
        antiperiodic_amp: float = 0.05,
    ) -> FieldState:
        """
        Near-uniform fields + Möbius-natural antiperiodic seed.

        The antiperiodic mode sin(u)·sin(πv) satisfies the Möbius
        identification f(u+π, 1−v) = −f(u, v), giving the topology
        something to amplify rather than pure noise.
        """
        torch.manual_seed(seed)
        P = self.manifold.uniform_noise(mean=P_mean, std=P_noise)

        # Antiperiodic seed: sin(u)·sin(πv) is Möbius-natural
        if antiperiodic_amp > 0:
            u = self.manifold.U  # (n_u, n_v)
            v = self.manifold.V  # (n_u, n_v)
            antiperiodic = antiperiodic_amp * torch.sin(u) * torch.sin(
                math.pi * v
            )
            P = P + antiperiodic

        A = self.manifold.uniform_noise(mean=A_mean, std=A_noise)
        M = torch.zeros(
            self.manifold.n_u, self.manifold.n_v, device=self.manifold.device
        )
        return FieldState(P=P, A=A, M=M, t=0)

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------
    def step(self) -> dict[str, Any]:
        """
        Execute one PAC cycle tick.

        P evolves continuously via SEC.  The topology enters through
        the Möbius Laplacian BCs, the spatial source, and the
        reinforcement that controls spectral balance.
        """
        s = self.state

        # 1. Measure Ξ (spectral + L² decomposition)
        xi = self.spectral.compute_xi(s.P)

        # Also track the L² decomposition ratio (more robust for
        # near-uniform fields)
        f_sym, f_anti = self.confluence.decompose(s.P)
        E_sym = f_sym.pow(2).sum().item()
        E_anti = f_anti.pow(2).sum().item()
        xi_L2 = E_anti / max(E_sym, 1e-14)

        # 2. SEC evolution: continuous, topology in Laplacian + reinforcement
        P_evolved, theta_sec = self.sec.step(s.P, xi_L2, self._theta)

        # 3. Actualization: crystallise where P is coherent, dissolve where turbulent
        A_new, theta_act = self._compute_actualization(P_evolved, s.A)

        # 4. Memory: accumulates record of the process
        M_new = ALPHA_REFERENCE * s.M + (1.0 - ALPHA_REFERENCE) * (
            P_evolved - s.P
        )

        # 5. Θ recycling: waste from collapse + dissolution → next step
        self._theta = theta_sec + theta_act

        # 6. Update state
        self.state = FieldState(P=P_evolved, A=A_new, M=M_new, t=s.t + 1)

        # 7. PAC enforcement (additive — no Ξ coefficient)
        self.state = self.pac.enforce(self.state)

        # 8. Diagnostics
        diag = self.pac.measure(self.state)
        diag["xi_spectral"] = xi
        diag["xi_L2"] = xi_L2
        diag["E_sym"] = E_sym
        diag["E_anti"] = E_anti
        diag["theta"] = self._theta
        diag["theta_sec"] = theta_sec
        diag["theta_act"] = theta_act
        diag["P_mean"] = self.state.P.mean().item()
        diag["A_mean"] = self.state.A.mean().item()
        diag["M_mean"] = self.state.M.mean().item()
        diag["P_std"] = self.state.P.std().item()
        diag["A_std"] = self.state.A.std().item()

        # RBF diagnostics
        diag["M_rbf"] = self.sec.M_rbf
        diag["xi_integral"] = self.sec.xi_integral

        self.diagnostics.record(diag)
        return diag

    # ------------------------------------------------------------------
    # Actualization — active crystallisation and dissolution
    # ------------------------------------------------------------------
    def _compute_actualization(
        self,
        P_new: torch.Tensor,
        A_old: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Active actualization: the contraction half of the PAC cycle.

        A grows where P has coherent structure (low Laplacian curvature)
        and dissolves where P is turbulent (high curvature).

        IMPORTANT: Uses signed P (not |P|) so that spatial structure
        (including antiperiodic modes) propagates from P into A.

        Returns
        -------
        (A_new, theta_dissolution) : tuple[Tensor, float]
        """
        # Coherence: where the Laplacian is small, the field is smooth → crystallise
        lap_mag = self.manifold.laplacian(P_new).abs()
        coherence = torch.exp(-lap_mag / max(self._act_scale, 1e-10))

        # Target A: coherence-weighted, normalised P (SIGNED — not |P|)
        # This preserves the spatial structure including antiperiodic modes.
        P_scale = P_new.abs().max().clamp(min=1e-10)
        target_A = coherence * (P_new / P_scale)

        # Relaxation toward target with EMA smoothing
        A_new = self._act_memory * A_old + (1.0 - self._act_memory) * target_A

        # Θ from dissolution: where A magnitude decreased, material returns to P
        dissolution = torch.clamp(A_old.abs() - A_new.abs(), min=0.0)
        theta_dissolution = dissolution.sum().item()

        return A_new, theta_dissolution

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------
    def run(
        self,
        n_steps: int,
        log_every: int = 100,
        save_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Run *n_steps* and return final summary.

        If *save_path* is given, dump full diagnostics to JSON on completion.
        """
        t0 = time.perf_counter()

        for i in range(n_steps):
            diag = self.step()

            if log_every and i % log_every == 0:
                print(
                    f"t={diag['t']:6d} | "
                    f"PAC_residual={diag['residual']:.2e} | "
                    f"Ξ_L2={diag.get('xi_L2', 0):.4f} | "
                    f"Ξ_spec={diag['xi_spectral']:.4f} | "
                    f"P_mean={diag['P_mean']:.4f} | "
                    f"M_rbf={diag.get('M_rbf', 0):.4f}"
                )

            if self.diagnostics.diverged:
                print(f"⚠ DIVERGENCE detected at step {diag['t']}. Stopping.")
                break

        elapsed = time.perf_counter() - t0
        summary = self.diagnostics.summary()
        summary["wall_seconds"] = elapsed
        summary["steps_per_second"] = self.diagnostics.n_steps / max(elapsed, 1e-9)

        if save_path:
            self.diagnostics.save(save_path)

        return summary

    # ------------------------------------------------------------------
    # Config loader
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "RealityEngine":
        """Construct engine from a YAML config file."""
        with open(path) as fh:
            config = yaml.safe_load(fh)
        return cls(config)
