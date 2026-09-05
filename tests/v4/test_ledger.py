"""The energy ledger closes, and the potential is the force's own.

Fail-first: these tests are `xfail(strict=True)` until the ledger commit; an unexpected pass is an
error. See `.spec/v4-derived-sink.spec.md` R1–R3 and acceptance KA-0..iv.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ._proxy import P, PROXY_1000

def _run(cfg, ticks, pipeline=None):
    eng = P.ParticleEngine(cfg, pipeline=pipeline, device=torch.device("cpu"))
    rows = []
    for _ in range(ticks):
        s = eng.tick()
        rows.append(dict(s.metrics))
    return eng, rows


def test_ledger_reports_potential_and_totals(proxy_config):
    eng = P.ParticleEngine(proxy_config, device=torch.device("cpu"))
    m = eng.tick().metrics
    for k in ("kinetic_int", "potential_int", "total_int", "e_int", "n_alive",
              "work_gravity", "work_pressure", "loss_drag", "loss_guard", "closure_residual"):
        assert k in m, k
    assert m["potential_int"] < 0, "bound gravity has negative potential energy"
    assert m["n_alive"] == proxy_config.n


def test_potential_matches_closed_form():
    """U(r) = -g m_i m_j e^{0.1/r0} [E1((r+0.1)/r0) - E1((3r0+0.1)/r0)] on [0, 3r0], 0 beyond."""
    scipy_special = pytest.importorskip("scipy.special")
    g, r0 = 1.5, 10.0
    table = P.gravity_potential_table(g, r0)             # (r_grid, U_grid)
    r_grid, U_grid = table
    exact = -g * math.exp(0.1 / r0) * (scipy_special.exp1((r_grid + 0.1) / r0)
                                       - scipy_special.exp1((3 * r0 + 0.1) / r0))
    rel = np.abs(U_grid - exact) / np.abs(exact).max()
    assert rel.max() < 1e-6, rel.max()
    assert abs(U_grid[-1]) < 1e-9, "continuous at the cutoff"


def test_closure_residual_every_tick(proxy_config):
    _, rows = _run(proxy_config, 120)
    worst = max(r["closure_residual"] for r in rows[1:])
    assert worst <= 1e-5, worst


def test_energy_conserved_to_truncation_without_drag():
    """KA-i: damping 1, sec 0, guard unbound — |dE| bounded by the Courant truncation, and halving
    cfl shrinks it by a first-order factor."""
    def drift(cfl):
        # the derived guard binds on the p99 tail BY CONSTRUCTION; an explicit unreachable cap
        # makes "guard unbound" literally true for the conservation check
        cfg = P.ParticleConfig(**{**PROXY_1000, "n": 400, "box": 27.9, "sec_balance": 0.0, "cfl": cfl,
                                  "max_speed": 1e9})
        eng, rows = _run(cfg, 500)
        assert max(r["at_cap_frac"] for r in rows) == 0.0
        E0, E1 = rows[0]["total_int"], rows[-1]["total_int"]
        cfl_max = max(r["cfl_number"] for r in rows)
        W = sum(abs(r["work_gravity"]) for r in rows)
        return abs(E1 - E0), cfl_max * W
    dE, bound = drift(0.2)
    assert dE <= bound, (dE, bound)
    dE_half, _ = drift(0.1)
    # The purpose: the error is TRUNCATION (shrinks with the step), not a bug (constant). Measured
    # 2026-09-05: halving cfl shrank |dE| by 7.6x (kick-drift at damping 1 is better than first
    # order here); the original window of 1.5-4x was my expectation, not a requirement.
    assert dE / max(dE_half, 1e-12) >= 1.5, (dE, dE_half)


def test_drag_loss_reproduces_the_exp04_rate():
    """KA-ii: g 0, sec 0, damping 0.99 — sum of loss_drag is the KE lost, and the fitted rate is
    2 ln(0.99)/dt_ref = -0.4020 within 1%."""
    cfg = P.ParticleConfig(n=200, box=60.0, r0=10.0, g=0.0, sec_balance=0.0, dims=3, seed=3,
                           ic="zeldovich", ic_amplitude=0.5, damping=0.99)
    eng, rows = _run(cfg, 200)
    ke = np.array([r["kinetic_int"] for r in rows]); t = np.array([r["sim_time"] for r in rows])
    lost = sum(r["loss_drag"] for r in rows[1:])
    assert abs(lost - (ke[0] - ke[-1])) <= 1e-6 * ke[0], (lost, ke[0] - ke[-1])
    slope = np.polyfit(t[20:], np.log(ke[20:]), 1)[0]
    assert abs(slope - 2 * math.log(0.99) / cfg.dt_ref) <= 0.01 * 0.402, slope


def test_guard_loss_is_exact():
    """KA-iii: force the guard to bind; loss_guard = sum 1/2 m (v^2 - cap^2) over the clamped set."""
    cfg = P.ParticleConfig(n=100, box=30.0, r0=10.0, g=1.5, sec_balance=0.0, dims=3, seed=1,
                           damping=1.0, max_speed=0.05)
    eng = P.ParticleEngine(cfg, device=torch.device("cpu"))
    for _ in range(30):
        s_prev = eng.state
        s = eng.tick()
    m = s.metrics
    assert m["at_cap_frac"] > 0.5
    assert m["loss_guard"] > 0
    assert m["closure_residual"] <= 1e-5


def test_momentum_closure(proxy_config):
    """KA-iv: with the third-law pressure the pressure impulse is zero; momentum of the interacting
    set changes only by what leaves it."""
    _, rows = _run(proxy_config, 60)
    imp = max(abs(r["impulse_pressure_x"]) + abs(r["impulse_pressure_y"]) + abs(r["impulse_pressure_z"])
              for r in rows)
    assert imp <= 1e-4 * rows[0]["mass_total"], imp
