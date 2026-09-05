"""Landauer erasure as a dynamical step: released entropy costs LN2 of kinetic energy.

Fail-first: `xfail(strict=True)` until the Landauer commit. Spec R7.
"""

from __future__ import annotations

import pytest
import torch

from ._proxy import P, PROXY_1000

LAN = pytest.mark.xfail(strict=True, reason="LandauerErasure not yet implemented (spec R7)")


def _spread_state(n=200, box=60.0, entropy=20.0, speed=3.0, seed=0):
    """Nobody dense (uniform at low density), everyone moving, everyone carrying entropy."""
    torch.manual_seed(seed)
    pos = torch.rand(n, 3) * box
    vel = torch.randn(n, 3); vel = vel / vel.norm(dim=-1, keepdim=True) * speed
    return P.ParticleState(pos=pos, vel=vel, mass=torch.ones(n), entropy=torch.full((n,), entropy),
                           box=box)


@LAN
def test_landauer_loss_matches_the_rule():
    cfg = P.ParticleConfig(n=200, box=60.0, r0=5.0, g=0.0, sec_balance=0.0, dims=3, damping=1.0,
                           landauer=True)
    s0 = _spread_state()
    s1 = P.SECUpdate()(s0, cfg)                       # everyone non-dense -> pure decay, dS < 0
    assert bool((s1.d_entropy < 0).all())
    s2 = P.LandauerErasure()(s1, cfg)
    ke1 = 0.5 * s1.mass * (s1.vel ** 2).sum(-1); ke2 = 0.5 * s2.mass * (s2.vel ** 2).sum(-1)
    expected = torch.minimum(ke1, P.LN2 * (-s1.d_entropy))
    assert torch.allclose(ke1 - ke2, expected, rtol=1e-5, atol=1e-6)
    assert abs(s2.metrics["loss_landauer"] - expected.sum().item()) <= 1e-5 * expected.sum().item()


@LAN
def test_landauer_zero_on_growth():
    """A tight cluster with zero entropy: dS > 0 everywhere, so nothing is erased."""
    cfg = P.ParticleConfig(n=64, box=60.0, r0=10.0, g=0.0, sec_balance=0.0, dims=3, damping=1.0,
                           landauer=True)
    torch.manual_seed(1)
    pos = (30.0 + 0.5 * torch.randn(64, 3)) % 60.0
    s0 = P.ParticleState(pos=pos, vel=torch.randn(64, 3), mass=torch.ones(64), entropy=torch.zeros(64),
                         box=60.0)
    s1 = P.SECUpdate()(s0, cfg)
    assert bool((s1.d_entropy > 0).all())
    s2 = P.LandauerErasure()(s1, cfg)
    assert torch.equal(s2.vel, s1.vel)
    assert s2.metrics["loss_landauer"] == 0.0


@LAN
def test_landauer_is_dt_invariant():
    """Forces off: total erased over matched simulated time is LN2 x total entropy released,
    whatever the step."""
    def erased(dt, ticks):
        cfg = P.ParticleConfig(n=200, box=60.0, r0=5.0, g=0.0, sec_balance=0.0, dims=3, damping=1.0,
                               landauer=True, dt=dt, seed=0, entropy_init=20.0,
                               ic="zeldovich", ic_amplitude=0.05, max_speed=1e9)
        eng = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
        for _ in range(ticks):
            s = eng.tick()
        return s.metrics["loss_landauer_cum"], s.metrics["sim_time"]
    e1, t1 = erased(0.05, 40); e2, t2 = erased(0.025, 80)
    assert abs(t1 - t2) < 1e-9
    assert abs(e1 - e2) <= 1e-4 * max(e1, 1e-12), (e1, e2)
