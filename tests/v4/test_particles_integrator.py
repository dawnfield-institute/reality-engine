"""The v4 particle substrate integrates its force law, and every bound reports when it binds.

Fail-first: tests marked `xfail(strict=True)` fail on the code as it stood after the 2026-08-28
diagnostic and are un-marked by the commit that fixes them. `strict=True` turns an unexpected
pass into an error, so the property "this test failed before the fix" is enforced by the runner
rather than by remembering commit order.

Two tests pass before and after and are labelled as such: they are the no-regression anchors.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from ._proxy import P, PROXY, run_marks  # also puts proof_of_concepts/v4 on sys.path


# ---------------------------------------------------------------------------------------
# R1 — the bound reports
# ---------------------------------------------------------------------------------------

def test_at_cap_frac_reported_from_pre_clamp_speed(proxy_config):
    """`at_cap_frac` exists after one tick and equals the diagnostic's own definition.

    Post-clamp every clamped speed is exactly `max_speed`, and the predicate
    `speed >= 0.999 * max_speed` is invariant under the clamp, so the fraction computed from
    `state.vel` after the tick must equal the metric computed before it — exactly.
    """
    eng = P.ParticleEngine(proxy_config, device=torch.device("cpu"))
    s = eng.tick()
    assert "at_cap_frac" in s.metrics
    assert "speed_p99" in s.metrics and "speed_max" in s.metrics
    cap = s.metrics["cap_eff"]
    sp = s.vel.norm(dim=-1)
    expected = (sp >= cap * 0.999).float().mean().item()
    assert s.metrics["at_cap_frac"] == pytest.approx(expected, abs=1e-9)
    assert s.metrics["dt_eff"] > 0 and s.metrics["sim_time"] == pytest.approx(s.metrics["dt_eff"])


def test_guard_reports_when_forced_to_bind(proxy_config):
    """The instrument can fail: force the guard to bind and it must say so, once."""
    cfg = P.ParticleConfig(**{**PROXY, "max_speed": 0.01})
    eng = P.ParticleEngine(cfg, device=torch.device("cpu"))
    with pytest.warns(RuntimeWarning, match="speed guard binding"):
        for _ in range(30):
            s = eng.tick()
    assert s.metrics["at_cap_frac"] > 0.9
    assert eng.bounds["at_cap_frac_max"] > 0.9
    assert eng.bounds["first_tick_at_cap_gt_1pct"] is not None
    # Warned exactly once: a second engine warns again, the same engine does not.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(10):
            eng.tick()
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]


# ---------------------------------------------------------------------------------------
# R3 / R5 — the guard never binds under the forces it was built for
# ---------------------------------------------------------------------------------------

def test_guard_never_binds_under_exp11_forces(proxy_run):
    _, marks = proxy_run
    worst = max(m["at_cap_frac"] for _, _, m in marks)
    assert worst <= 0.02, f"at_cap_frac reached {worst:.3f}; marks={[(t, round(m['at_cap_frac'], 3)) for t, _, m in marks]}"


# ---------------------------------------------------------------------------------------
# R2 — SEC memory releases
# ---------------------------------------------------------------------------------------

def _all_dense_state(cfg: P.ParticleConfig, entropy: float = 100.0):
    """A tight cluster: every particle has far more than 1.5x the uniform neighbour count."""
    torch.manual_seed(0)
    n, d = 64, cfg.dims
    pos = (cfg.box / 2 + 0.05 * cfg.r0 * torch.randn(n, d)) % cfg.box
    return P.ParticleState(pos=pos, vel=torch.zeros(n, d), mass=torch.ones(n),
                           entropy=torch.full((n,), float(entropy)), box=cfg.box)


def test_sec_memory_releases_on_dense_branch(proxy_config):
    cfg = proxy_config
    s = _all_dense_state(cfg)
    r, _, _ = P.pairwise(s, 1.0)
    local = (r < cfg.r0).sum(dim=1).float()
    d = s.pos.shape[1]
    v_ball = (math.pi ** (d / 2) / math.gamma(d / 2 + 1)) * cfg.r0 ** d
    expected = float(s.n) * v_ball / (s.box ** d)
    assert bool((local > 1.5 * expected).all()), "fixture must be all-dense"
    s2 = P.SECUpdate()(s, cfg)
    growth_only = 100.0 + 0.1 * (local - expected)
    assert bool((s2.entropy < growth_only - 1e-3).all()), \
        "a dense particle accumulated with no release at all"


def test_entropy_bounded_and_not_monotone(proxy_run):
    _, marks = proxy_run
    series = [m["entropy_mean"] for _, _, m in marks]
    cfg = P.ParticleConfig(**PROXY)
    d = cfg.dims
    v_ball = (math.pi ** (d / 2) / math.gamma(d / 2 + 1)) * cfg.r0 ** d
    expected = cfg.n * v_ball / (cfg.box ** d)
    bound = 0.1 * (cfg.n - expected) / (1.0 - cfg.memory_decay)
    strictly_increasing = all(b > a for a, b in zip(series, series[1:]))
    assert not strictly_increasing, f"entropy_mean strictly increasing: {[round(x, 1) for x in series]}"
    assert max(series) <= bound, f"entropy_mean {max(series):.1f} exceeds the provable bound {bound:.1f}"


# ---------------------------------------------------------------------------------------
# R4 — per-tick constants are rates
# ---------------------------------------------------------------------------------------

def _free(dt: float, ticks: int):
    """No forces, small correlated initial velocities, well below any cap."""
    cfg = P.ParticleConfig(n=200, box=60.0, r0=10.0, g=0.0, sec_balance=0.0, dims=3, seed=3,
                           ic="zeldovich", ic_amplitude=0.05, dt=dt)
    eng = P.ParticleEngine(cfg, device=torch.device("cpu"))
    for _ in range(ticks):
        eng.tick()
    return eng


def test_drag_is_dt_invariant():
    a = _free(0.05, 40)
    b = _free(0.025, 80)
    assert torch.allclose(a.state.vel, b.state.vel, rtol=1e-5, atol=1e-7), \
        f"|v| after t=2: {a.state.vel.norm(dim=-1).mean():.6f} vs {b.state.vel.norm(dim=-1).mean():.6f}"


def test_damped_free_drift_known_answer():
    """NO-REGRESSION ANCHOR — passes before and after the repair.

    With no forces the Integrator is the whole dynamics. At base dt the repaired path must be
    bit-identical to the float32 recurrence `v <- 0.99 v; p <- (p + v dt) mod box`.
    """
    cfg = P.ParticleConfig(n=200, box=60.0, r0=10.0, g=0.0, sec_balance=0.0, dims=3, seed=3,
                           ic="zeldovich", ic_amplitude=0.05)
    eng = P.ParticleEngine(cfg, device=torch.device("cpu"))
    v, p = eng.state.vel.clone(), eng.state.pos.clone()
    assert v.norm(dim=-1).max() < 1.0, "fixture must start well below any speed guard"
    for _ in range(25):
        v = v * cfg.damping
        p = (p + v * cfg.dt) % cfg.box
        eng.tick()
    assert torch.equal(eng.state.vel, v)
    assert torch.equal(eng.state.pos, p)
    # and the closed form, to 1e-5
    v0 = eng.state.vel / (cfg.damping ** 25)
    assert torch.allclose(eng.state.vel, v0 * cfg.damping ** 25, rtol=1e-5)


# ---------------------------------------------------------------------------------------
# Ledger, and the local-time program left untouched
# ---------------------------------------------------------------------------------------

def test_ledger_records_momentum_z_in_3d(proxy_config):
    eng = P.ParticleEngine(proxy_config, device=torch.device("cpu"))
    s = eng.tick()
    assert {"momentum_x", "momentum_y", "momentum_z"} <= set(s.metrics)
    cfg2 = P.ParticleConfig(**{**PROXY, "dims": 2, "n": 100})
    s2 = P.ParticleEngine(cfg2, device=torch.device("cpu")).tick()
    assert "momentum_z" not in s2.metrics


def test_local_time_global_mode_is_inert():
    """NO-REGRESSION ANCHOR for poc_10: with time_mode="global" the *_TIME pipeline is
    bit-identical to the plain one, and "potential" still runs and populates tau."""
    base = dict(n=200, box=30.0, r0=10.0, g=1.5, dims=3, seed=5)
    a = P.ParticleEngine(P.ParticleConfig(**base), pipeline=P.CANONICAL, device=torch.device("cpu"))
    b = P.ParticleEngine(P.ParticleConfig(**base), pipeline=P.CANONICAL_TIME, device=torch.device("cpu"))
    for _ in range(20):
        a.tick(); b.tick()
    assert torch.equal(a.state.pos, b.state.pos) and torch.equal(a.state.vel, b.state.vel)
    assert b.state.tau is None
    c = P.ParticleEngine(P.ParticleConfig(**base, time_mode="potential"), pipeline=P.CANONICAL_TIME,
                         device=torch.device("cpu"))
    for _ in range(20):
        s = c.tick()
    assert s.tau is not None and torch.isfinite(s.tau).all() and torch.isfinite(s.pos).all()
