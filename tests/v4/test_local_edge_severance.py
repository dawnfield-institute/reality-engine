"""The local-edge severance trigger (R2, 2026-09-15): a retained particle severs the tick its cumulative
pressure work first exceeds zero at or after sev_t_arm. No threshold; the zero is the ledger's. Same
bookkeeping as the stress trigger (form: decoupling, not destruction; the amount is the particle)."""
from __future__ import annotations

import math

import torch

from ._proxy import P

SMALL = dict(n=300, box=25.2, r0=10.0, g=1.5, dims=3, seed=1, sec_balance=0.6541, damping=1.0, pac_kappa=0.5)


def _run(cfg, ticks):
    eng = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    states = []
    for _ in range(ticks):
        states.append(eng.tick())
    return eng, states


def test_unarmed_local_edge_is_bit_identical_to_no_severance():
    a = _run(P.ParticleConfig(**SMALL), 40)[1][-1]
    b = _run(P.ParticleConfig(**SMALL, sev_mode="local_edge", sev_t_arm=1e9), 40)[1][-1]
    assert torch.equal(a.pos, b.pos) and torch.equal(a.vel, b.vel) and torch.equal(a.entropy, b.entropy)
    assert b.severed is None or not bool(b.severed.any())


def test_fires_exactly_the_positive_work_set_at_arming_and_only_new_crossers_after():
    cfg = P.ParticleConfig(**SMALL, sev_mode="local_edge", sev_t_arm=2.0)
    eng, states = _run(cfg, 120)
    prev_sev = torch.zeros(cfg.n, dtype=torch.bool)
    armed_seen = False
    for k in range(1, len(states)):
        before, after = states[k - 1], states[k]
        t_at_sev = float(before.metrics["sim_time"])           # severance reads the state the previous tick left
        sev = after.severed if after.severed is not None else torch.zeros(cfg.n, dtype=torch.bool)
        new = sev & ~prev_sev
        wp = before.work_p_i
        if t_at_sev < cfg.sev_t_arm:
            assert not bool(new.any())
        else:
            expect = ~prev_sev & (wp > 0)
            assert torch.equal(new, expect), f"tick {k}: fired {int(new.sum())} vs expected {int(expect.sum())}"
            armed_seen = armed_seen or bool(new.any())
        assert not bool((sev & prev_sev & ~sev).any())          # never un-severed
        prev_sev = sev
    assert armed_seen, "the trigger never fired after arming — the proxy should have positive-work particles by t = 2"
    assert bool((eng.state.severed).any()) and float(eng.state.metrics["sev_frac_cum"]) < 1.0


def test_severed_particles_interact_with_nothing_and_their_work_freezes():
    cfg = P.ParticleConfig(**SMALL, sev_mode="local_edge", sev_t_arm=2.0)
    eng, states = _run(cfg, 80)
    sev = eng.state.severed
    assert bool(sev.any())
    # after severance a particle's acceleration is zero every tick, so its cumulative works stop changing
    first = next(i for i, s in enumerate(states) if s.severed is not None and bool(s.severed.any()))
    idx = torch.nonzero(states[first].severed).flatten()[:5]
    for i in idx.tolist():
        w_then = states[first + 1].work_p_i[i].item() if first + 1 < len(states) else None
        w_end = eng.state.work_p_i[i].item()
        if w_then is not None:
            assert math.isclose(w_then, w_end, rel_tol=0, abs_tol=1e-9)
    # and the retained set's kinetic change is still the works minus the losses, tick by tick (closure)
    assert all(s.metrics["closure_residual"] < 1e-6 for s in states[1:])


def test_random_control_replays_the_local_edge_schedule():
    cfg = P.ParticleConfig(**SMALL, sev_mode="local_edge", sev_t_arm=2.0)
    eng, _ = _run(cfg, 80)
    sched = [(e["sim_time"], e["count"]) for e in eng.events]
    assert sched
    cfg_r = P.ParticleConfig(**SMALL, sev_tau=1.0, sev_mode="random", sev_schedule=sched)
    eng_r, _ = _run(cfg_r, 80)
    assert sum(c for _, c in sched) == int(eng_r.state.severed.sum().item())


def test_unbound_fraction_is_reported_for_every_event():
    cfg = P.ParticleConfig(**SMALL, sev_mode="local_edge", sev_t_arm=2.0)
    eng, states = _run(cfg, 80)
    ev = [s.metrics for s in states if s.metrics.get("sev_count", 0) > 0]
    assert ev and all(0.0 <= m["sev_unbound_frac"] <= 1.0 and m["sev_wp_min"] > 0 for m in ev)
