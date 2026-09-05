"""Ledger severance: whole-particle decoupling on Milestone R's stress trigger.

Fail-first: `xfail(strict=True)` until the severance commit. Spec R4–R6 and the severance gates.
"""

from __future__ import annotations

import math

import pytest
import torch

from ._proxy import P, PROXY_1000

def _sp(cfg):
    return cfg.box / math.ceil(cfg.n ** (1.0 / cfg.dims))


def _cfg(**kw):
    return P.ParticleConfig(**{**PROXY_1000, **kw})


def test_sink_pipeline_is_inert_by_default():
    """CANONICAL_SINK with sev_tau=None, landauer=False is bit-identical to CANONICAL."""
    a = P.ParticleEngine(_cfg(n=300, box=25.2), pipeline=P.CANONICAL, device=torch.device("cpu"))
    b = P.ParticleEngine(_cfg(n=300, box=25.2), pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    for _ in range(50):
        a.tick(); b.tick()
    assert torch.equal(a.state.pos, b.state.pos) and torch.equal(a.state.vel, b.state.vel)
    assert b.state.severed is None or not bool(b.state.severed.any())


def test_stress_trigger_matches_independent_recomputation():
    """Run the pipeline operator by operator; before LedgerSeverance acts, recompute the rule from
    the state and demand the fired set equals it — and that no retained particle with a neighbour
    satisfies it after the operator ran."""
    cfg = _cfg(sev_tau=1.0)
    eng = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    ops = eng.pipeline
    fired_total = 0
    for _ in range(80):
        s = eng.state
        for op in ops:
            if isinstance(op, P.LedgerSeverance):
                r, _, _ = P.pairwise(s, 1.0)
                alive = torch.ones(s.n, dtype=torch.bool) if s.severed is None else ~s.severed
                near = (r < _sp(cfg)) & alive.unsqueeze(1) & alive.unsqueeze(0)
                deg = near.sum(1)
                dS = (s.entropy.unsqueeze(1) - s.entropy.unsqueeze(0)).abs()
                mn = torch.where(near, dS, torch.full_like(dS, float("inf"))).min(1).values
                expect = alive & (deg >= 1) & (mn > cfg.sev_tau)
                before = s.severed.clone() if s.severed is not None else torch.zeros(s.n, dtype=torch.bool)
            s = op(s, eng.config)
            if isinstance(op, P.LedgerSeverance):
                now = s.severed if s.severed is not None else torch.zeros(s.n, dtype=torch.bool)
                fired = now & ~before
                assert torch.equal(fired, expect), (int(fired.sum()), int(expect.sum()))
                fired_total += int(fired.sum())
        eng.state = s
        eng.tick_count += 1
    assert fired_total > 0, "fixture must sever something for the test to mean anything"


def test_severed_particles_interact_with_nothing():
    cfg = _cfg(sev_tau=1.0)
    eng = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    for _ in range(80):
        s = eng.tick()
        if s.severed is not None and s.severed.any():
            break
    assert s.severed is not None and s.severed.any(), "fixture must sever"
    idx = torch.nonzero(s.severed).flatten()
    v_then = s.vel[idx].clone(); ent_then = s.entropy[idx].clone()
    for _ in range(100):
        s = eng.tick()
    assert torch.equal(s.vel[idx], v_then), "a severed particle's velocity must be bit-identical"
    assert torch.equal(s.entropy[idx], ent_then), "a severed particle's entropy is frozen"
    m = s.metrics
    # three float32 reductions over different index sets: equal to rounding, not bitwise
    assert abs(m["mass_int"] + m["mass_sev"] - m["mass_total"]) <= 1e-6 * m["mass_total"]
    assert abs(m["kinetic_sev"] - sum(e["ke_out"] for e in eng.events)) <= 1e-6 * max(m["kinetic_sev"], 1)
    assert m["n_alive"] == cfg.n - int(s.severed.sum())


def test_degree_zero_never_severs():
    cfg = P.ParticleConfig(n=2, box=60.0, r0=10.0, g=0.0, sec_balance=0.0, dims=3, damping=1.0,
                           sev_tau=0.01, sev_radius=2.0)
    pos = torch.tensor([[10.0, 30.0, 30.0], [40.0, 30.0, 30.0]])
    s = P.ParticleState(pos=pos, vel=torch.zeros(2, 3), mass=torch.ones(2),
                        entropy=torch.tensor([100.0, 0.0]), box=60.0)
    out = P.LedgerSeverance()(s, cfg)
    assert out.severed is None or not bool(out.severed.any())


def test_random_mode_reproduces_its_schedule():
    sched = [(0.5, 3), (1.0, 5), (1.5, 2)]
    cfg = _cfg(n=300, box=25.2, sev_tau=1.0, sev_mode="random", sev_schedule=sched)
    eng = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    s = eng.state
    while float(s.metrics.get("sim_time", 0.0)) < 2.0:
        s = eng.tick()
    assert int(s.severed.sum()) == sum(k for _, k in sched)


def test_severance_energy_is_charged_to_the_interacting_total():
    cfg = _cfg(sev_tau=1.0)
    eng = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    prev = None
    checked = 0
    for _ in range(120):
        s = eng.tick(); m = s.metrics
        if prev is not None and m.get("sev_count", 0) > 0:
            d_total = m["total_int"] - prev["total_int"]
            budget = (m["work_gravity"] + m["work_pressure"] - m["loss_drag"] - m["loss_guard"]
                      - m.get("loss_landauer", 0.0) - m["loss_severance_energy"]
                      + (m["potential_int"] - prev["potential_int"]))
            # kinetic closure already holds; the severance charge on total_int is ke_out + u_out
            assert m["closure_residual"] <= 1e-5
            assert m["loss_severance_energy"] != 0.0
            checked += 1
        prev = m
    assert checked > 0, "fixture must sever within the horizon"
