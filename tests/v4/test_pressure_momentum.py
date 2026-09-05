"""SEC pressure must be a pair force: equal and opposite, so it injects no net momentum.

Found 2026-09-05 by the derived-sink design pass. The rule inherited from exp_09 computes, for
every ordered pair, sec * (S_i - S_j) * exp(-r/r0) along the unit vector from j to i. Under the
swap i <-> j both the entropy difference and the unit vector flip sign, so the force on j from i
is the SAME vector as the force on i from j: the antisymmetric part of the pair interaction is
identically zero and the whole term is self-propulsion. Every pair injects net momentum 2F.

These tests fail on that rule and pass on any pair law with a magnitude symmetric in (i, j).
"""

from __future__ import annotations

import pytest
import torch

from ._proxy import P


def _pair(S_i: float, S_j: float, sep: float = 6.0):
    cfg = P.ParticleConfig(n=2, box=60.0, r0=10.0, g=0.0, sec_balance=1.0, dims=3, damping=1.0)
    pos = torch.tensor([[20.0, 30.0, 30.0], [20.0 + sep, 30.0, 30.0]])
    s = P.ParticleState(pos=pos, vel=torch.zeros(2, 3), mass=torch.ones(2),
                        entropy=torch.tensor([S_i, S_j]), box=60.0)
    return s, cfg


def test_pressure_pair_force_is_antisymmetric():
    s, cfg = _pair(5.0, 1.0)
    out = P.SECPressure()(s, cfg)
    F = out.acc                                   # unit masses: acc is force
    assert F[0].norm() > 0, "fixture must exert a force"
    assert torch.allclose(F[0], -F[1], atol=1e-6), f"F_i={F[0].tolist()} F_j={F[1].tolist()}"


def test_pressure_alone_conserves_momentum():
    """Random cloud, gravity off, no drag, no guard binding: total momentum is invariant."""
    cfg = P.ParticleConfig(n=200, box=30.0, r0=10.0, g=0.0, sec_balance=0.6541, dims=3, seed=7,
                           damping=1.0, entropy_init=5.0)
    eng = P.ParticleEngine(cfg, pipeline=[P.SECPressure, P.Integrator, P.PACLedger],
                           device=torch.device("cpu"))
    p0 = (eng.state.mass.unsqueeze(-1) * eng.state.vel).sum(0)
    for _ in range(20):
        s = eng.tick()
    assert s.metrics["at_cap_frac"] == 0.0, "guard must not bind in this fixture"
    p1 = (s.mass.unsqueeze(-1) * s.vel).sum(0)
    assert torch.allclose(p1, p0, atol=1e-4 * s.mass.sum()), f"p0={p0.tolist()} p1={p1.tolist()}"
