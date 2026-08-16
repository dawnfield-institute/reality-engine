"""PACBalanceOperator — the balance field as derived from ADE closure.

Replaces `RBFOperator`. The Recursive Balance Field is superseded by Potential-
Actualization Conservation: RBF's reaction-diffusion form was never derived, and
`exp_30_arithmetic_dimension_emergence` (94/95 checks, March 2026) reconstructed what the
balance field must be from ADE arithmetic instead.

    exp_30p_rbf_from_ade.py:
        B(x,t) = lambda * [(E - I) / (1 + alpha*|M|)] * Phi(x)

        (E - I)            inter-level imbalance                        [Tier 1]
        1/(1 + alpha*|M|)  L3 bounded regularizer, Pade[0/1] of exp(-aM) [Tier 2]
        Phi(x)             antiperiodic Mobius eigenmodes, Phi(x+pi) = -Phi(x)  [Tier 1]

        "Structure is forced; parameters are not."

Phi's role is stated in the same script: it *projects onto the antiperiodic sector* — the
modes that "feel" the twist. On the manifold that projection is the Z_2 Reynolds operator
`(f - f∘T)/2`, whose 1/2 is a group average (1/|G|), geometric in origin rather than
normalisation. The engine already implements it as
`MobiusManifold.project_antiperiodic`.

Why this matters for structure, and what RBF could not do:

    RBF:  B = lap(E-I) + lambda*M*lap(M) - alpha*||E-I||^2 - gamma*(E-I)

A laplacian against damping fixes a correlation length, so the spatial spectrum is
stationary and the fractal dimension of every downstream field is pinned. Measured: D(M),
D(|E-I|) and D(E^2) all sit at ~1.44 and do not move over 10k ticks, and are UNCHANGED by
deleting gravity entirely, by the Poisson source exponent, by viscosity across 40x, or by
letting the tiling filter's Xi grow. Geometry was being set here and copied downstream.

An eigenmode basis has no single characteristic scale, so the mode content can shift over
time — which is the locality/globality ratio being able to grow rather than being frozen.

Also from exp_30 and honoured here: at the x=2 ADE confluence (x+x = x*x), E=I gives B=0,
and M acts as a bounded damping integrator rather than a source. Conservation is pure
transfer — dE/dt = -B, dI/dt = +B — which QBEOperator already implements; adding a source
term breaks conservation (rel err 9.9e-04 vs 0.0e+00). RBF's -alpha*||E-I||^2 and
-gamma*(E-I) are exactly such source terms.

HONEST: lambda and alpha are Tier 3 in exp_30 — "no clean ADE derivation". The structure
is forced; these two parameters are not, and are set empirically here.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.v3.engine.config import SimulationConfig
from src.v3.engine.event_bus import EventBus
from src.v3.engine.state import FieldState
from src.v3.substrate.manifold import MobiusManifold

_EPS = 1e-12


class PACBalanceOperator:
    """B = lambda * P_anti[(E - I) / (1 + alpha*|M|)] — ADE-derived balance field.

    Writes dE/dt into metrics exactly as RBFOperator did, so the integrator and QBE are
    unchanged. Emits the same alpha_local/lambda_local/balance_magnitude metrics so the
    scorecard and analyzers keep working.
    """

    def __init__(self, lam: float = 1.0, alpha_mem: float = 1.0) -> None:
        self._manifold: Optional[MobiusManifold] = None
        self.lam = lam
        self.alpha_mem = alpha_mem

    @property
    def name(self) -> str:
        return "pac_balance"

    def _get_manifold(self, state: FieldState) -> MobiusManifold:
        nu, nv = state.E.shape
        if self._manifold is None or self._manifold.nu != nu or self._manifold.nv != nv:
            self._manifold = MobiusManifold(nu, nv, device=state.E.device)
        return self._manifold

    def __call__(
        self,
        state: FieldState,
        config: SimulationConfig,
        bus: Optional[EventBus] = None,
    ) -> FieldState:
        E, I, M = state.E, state.I, state.M

        # Tier 1 — inter-level imbalance. E = I gives B = 0 (ADE confluence equilibrium).
        imbalance = E - I

        # Tier 2 — L3 bounded regularizer, Pade[0/1] of exp(-alpha*M). M damps, never
        # sources: as |M| grows the balance field is attenuated, not driven.
        regularized = imbalance / (1.0 + self.alpha_mem * M.abs())

        # Tier 1 — Phi: project onto the antiperiodic sector, (f - f∘T)/2.
        B = self.lam * self._get_manifold(state).project_antiperiodic(regularized)

        # Same partition metrics RBF emitted, so nothing downstream breaks.
        E2, I2, M2 = E.pow(2), I.pow(2), M.pow(2)
        denom = E2 + I2 + M2 + _EPS
        alpha_local = (E2 + I2) / denom
        lambda_local = M2 / denom

        metrics = dict(state.metrics)
        metrics["dE_dt"] = B
        metrics["balance_magnitude"] = B.abs().mean().item()
        metrics["alpha_local_mean"] = alpha_local.mean().item()
        metrics["alpha_local_std"] = alpha_local.std().item()
        metrics["lambda_local_mean"] = lambda_local.mean().item()
        metrics["lambda_local_std"] = lambda_local.std().item()

        if bus is not None:
            bus.emit("pac_balance", {"balance_mag": metrics["balance_magnitude"]})

        return state.replace(metrics=metrics)
