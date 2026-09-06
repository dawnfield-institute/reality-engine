"""The PAC ledger on particles: the pair energy is exact, the budget pays, the total conserves.

Fail-first: `xfail(strict=True)` until the implementing commit; an unexpected pass is an error.
See `.spec/v4-pac-ledger.spec.md` R1–R9 and acceptance KA-0..iii.
"""

from __future__ import annotations

import math

import pytest
import torch

from ._proxy import P, PROXY_1000

SMALL = dict(n=300, box=25.2, r0=10.0, g=1.5, dims=3, seed=1, sec_balance=0.6541, damping=1.0)


def _run(cfg, ticks, pipeline=None):
    eng = P.ParticleEngine(cfg, pipeline=pipeline, device=torch.device("cpu"))
    rows = []
    for _ in range(ticks):
        rows.append(dict(eng.tick().metrics))
    return eng, rows


def test_pressure_is_gradient_of_shifted_pair_energy():
    """KA-0 / R1: F_i = -dE_SEC/dx_i at fixed entropy, finite difference on a 3-particle state."""
    c = P.ParticleConfig(n=3, box=60.0, r0=10.0, g=0.0, dims=3, sec_balance=0.6541, damping=1.0, pac_kappa=1.0)
    # float64 throughout: a central difference of an O(10) energy over 2e-3 is below single
    # precision (the first run of this test missed by 1.2e-3 relative for that reason alone)
    pos = torch.tensor([[20.0, 20.0, 20.0], [26.0, 21.0, 19.0], [23.0, 27.0, 22.0]], dtype=torch.float64)
    s = P.ParticleState(pos=pos, vel=torch.zeros(3, 3, dtype=torch.float64), mass=torch.ones(3, dtype=torch.float64),
                        entropy=torch.tensor([5.0, 1.0, 3.0], dtype=torch.float64), box=60.0)
    force = P.SECPressure()(s, c).acc_pressure * s.mass.unsqueeze(-1)
    h = 1e-3
    for k in range(3):
        for ax in range(3):
            def E(delta):
                p = pos.clone(); p[k, ax] += delta
                return P.sec_pair_energy(s.replace(pos=p), c)[0]
            grad = (E(h) - E(-h)) / (2 * h)
            assert abs(force[k, ax].item() + grad) <= 1e-3 * max(abs(grad), 1e-6), (k, ax, force[k, ax].item(), -grad)


def test_transfer_identity_every_tick():
    """KA-i / R2: sum (dE/dS) dS + d(sum P) = 0 to 1e-6 of P(0), every tick, budget binding or not."""
    cfg = P.ParticleConfig(**SMALL, pac_kappa=1.0)
    eng, rows = _run(cfg, 120, P.CANONICAL_SINK)
    p0 = eng.budget0
    assert p0 > 0
    prev = rows[0]["budget_int"]
    worst = 0.0
    for r in rows[1:]:
        worst = max(worst, abs(r["sec_transfer"] + (r["budget_int"] - prev)) / p0)
        prev = r["budget_int"]
        assert r["transfer_residual"] <= 1e-6, r["transfer_residual"]
    assert worst <= 1e-6, worst


def test_net_creation_bounded_by_budget_and_budget_binds():
    """KA-ii / R4: over a whole proxy run at kappa=1 the net pair energy created by entropy change
    never exceeds P(0) (exact), the budget actually binds, and pressure work stays within P(0) up to
    the integrator's declared truncation allowance."""
    cfg = P.ParticleConfig(**{**PROXY_1000, "n": 400, "box": 27.9}, pac_kappa=1.0)
    eng = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    m = None
    while True:
        m = eng.tick().metrics
        if m["sim_time"] >= 15.0: break
    p0 = eng.budget0
    assert m["sec_transfer_cum"] <= p0 * (1 + 1e-6), (m["sec_transfer_cum"], p0)
    assert eng.bounds["budget_bound_frac_max"] > 0.0, "the budget never bound: the test is vacuous"
    assert m["work_pressure_cum"] <= p0 * 1.10, (m["work_pressure_cum"], p0)


def test_total_ledger_conserved_to_truncation():
    """KA-iii / R3: KE + U + E_SEC + sum P drifts by at most the Courant bound on the gross work per
    tick, and halving the step over the smooth window (t <= 3.5, before the detonation is chaotic)
    reduces the end drift by >= 1.5x. Threads pinned: a ratio is a measurement, not a race."""
    torch.set_num_threads(1)
    def run(dt_ref, t_end):
        cfg = P.ParticleConfig(**{**PROXY_1000, "n": 400, "box": 27.9, "dt": dt_ref, "dt_ref": dt_ref,
                                  "max_speed": 1e9}, pac_kappa=1.0)
        eng = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu")); rows = []
        while True:
            rows.append(dict(eng.tick().metrics))
            if rows[-1]["sim_time"] >= t_end - 1e-9: break
        gross = sum(abs(r["work_gravity"]) + abs(r["work_pressure"]) for r in rows)
        drift = abs(rows[-1]["total_pac"] - rows[0]["total_pac"])
        cfl_max = max(r["cfl_number"] for r in rows)
        return drift, cfl_max * gross, rows
    d1, bound1, rows = run(0.05, 3.5)
    assert d1 <= bound1, (d1, bound1)
    assert max(r["closure_pac"] for r in rows[1:]) <= 0.05, max(r["closure_pac"] for r in rows[1:])
    d2, _, _ = run(0.025, 3.5)
    assert d1 / max(d2, 1e-12) >= 1.5, (d1, d2)


def test_kappa_none_is_structurally_inert_and_anchor_holds():
    """R5: with pac_kappa=None nothing about the budget exists, and CANONICAL_SINK (sinks off) is
    still bit-identical to CANONICAL."""
    cfg = P.ParticleConfig(**SMALL)
    a = P.ParticleEngine(cfg, pipeline=P.CANONICAL, device=torch.device("cpu"))
    b = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    for _ in range(50):
        sa, sb = a.tick(), b.tick()
    assert sb.budget is None and "budget_int" not in sb.metrics and "sec_energy_int" not in sb.metrics
    assert torch.equal(sa.pos, sb.pos) and torch.equal(sa.vel, sb.vel)
    assert not hasattr(b, "budget0") or b.budget0 is None


def test_kappa_zero_is_gravity_only():
    """R4: kappa = 0 means entropy cannot grow, so the run equals sec_balance = 0 bit for bit."""
    a = P.ParticleEngine(P.ParticleConfig(**SMALL, pac_kappa=0.0), pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    b = P.ParticleEngine(P.ParticleConfig(**{**SMALL, "sec_balance": 0.0}), pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    for _ in range(80):
        sa, sb = a.tick(), b.tick()
    assert sa.entropy.abs().max().item() == 0.0
    assert torch.equal(sa.pos, sb.pos) and torch.equal(sa.vel, sb.vel)


def test_severed_budget_frozen():
    """R6: a severed particle's budget does not change after severance."""
    # with the budget on the entropy stays below ~0.3 (it cannot create more pair energy than
    # P(0)), so the stress trigger needs tau = 0.1 to fire at all: measured 2026-09-06
    cfg = P.ParticleConfig(**PROXY_1000, pac_kappa=1.0, sev_tau=0.1)
    eng, rows = _run(cfg, 200, P.CANONICAL_SINK)
    s = eng.state
    assert s.severed is not None and bool(s.severed.any()), "nothing severed: fixture too short"
    idx = torch.nonzero(s.severed).flatten()
    b0 = s.budget[idx].clone()
    for _ in range(50):
        s = eng.tick()
    assert torch.equal(s.budget[idx], b0)
    assert s.metrics["budget_int"] <= s.budget[s.alive()].sum().item() * (1 + 1e-6)
