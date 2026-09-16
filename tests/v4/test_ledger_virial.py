"""Instrumentation for the edge scoping (2026-09-14 §7): per-particle work by force, pair-form virial
terms, and the gross legs of the ledger transfer. No physics change: every quantity here is a
re-partition of numbers the engine already computed, and each test is an identity."""
import math

import torch

from ._proxy import P

SMALL = dict(n=300, box=25.2, r0=10.0, g=1.5, dims=3, seed=1, sec_balance=0.6541, damping=1.0)


def _run(cfg, ticks):
    eng = P.ParticleEngine(cfg, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    rows = [dict(eng.tick().metrics) for _ in range(ticks)]
    return eng, rows


def test_per_particle_work_sums_to_the_global_work():
    eng, rows = _run(P.ParticleConfig(**SMALL, pac_kappa=0.5), 60)
    s = eng.state
    assert s.work_g_i is not None and s.work_p_i is not None
    assert math.isclose(s.work_g_i.double().sum().item(), rows[-1]["work_gravity_cum"], rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(s.work_p_i.double().sum().item(), rows[-1]["work_pressure_cum"], rel_tol=1e-6, abs_tol=1e-6)
    # the local-sign summaries are consistent with the per-particle array
    alive = s.alive()
    assert math.isclose(rows[-1]["work_pressure_pos_frac"], (s.work_p_i[alive] > 0).double().mean().item(), abs_tol=1e-12)
    assert math.isclose(rows[-1]["work_pressure_pos_sum"] + rows[-1]["work_pressure_neg_sum"], rows[-1]["work_pressure_cum"], rel_tol=1e-6, abs_tol=1e-6)


def test_gross_transfer_legs_sum_to_the_net_transfer():
    _, rows = _run(P.ParticleConfig(**SMALL, pac_kappa=0.5), 60)
    for m in rows:
        assert math.isclose(m["transfer_growth"] - m["transfer_credit"], m["sec_transfer"], rel_tol=1e-9, abs_tol=1e-9)
        assert m["transfer_growth"] >= 0.0 and m["transfer_credit"] >= 0.0
    assert math.isclose(rows[-1]["transfer_growth_cum"] - rows[-1]["transfer_credit_cum"], rows[-1]["sec_transfer_cum"], rel_tol=1e-9, abs_tol=1e-9)


def test_pair_virial_equals_sum_r_dot_F_on_an_unwrapped_cluster():
    """Put a small cluster in the middle of a large box so no pair wraps; then the pair form
    sum_{i<j} r_ij . f_ij must equal sum_i (r_i - c) . F_i for any origin c (sum F = 0)."""
    c = P.ParticleConfig(n=40, box=400.0, r0=10.0, g=1.5, dims=3, seed=3, sec_balance=0.6541, damping=1.0, pac_kappa=0.5)
    eng = P.ParticleEngine(c, pipeline=P.CANONICAL_SINK, device=torch.device("cpu"))
    st = eng.state
    torch.manual_seed(7)
    pos = 200.0 + 6.0 * torch.randn(c.n, 3)            # a cluster of radius ~6 at the box centre
    st = st.replace(pos=pos, vel=torch.zeros_like(pos), entropy=torch.rand(c.n) * 2.0, acc=None, acc_gravity=None, acc_pressure=None)
    sg = P.LocalGravity()(st, c)
    sp = P.SECPressure()(st, c)
    Fg = sg.acc_gravity * st.mass.unsqueeze(-1)
    Fp = sp.acc_pressure * st.mass.unsqueeze(-1)
    centre = pos.mean(0)
    vg = ((pos - centre) * Fg).sum().item()
    vp = ((pos - centre) * Fp).sum().item()
    assert math.isclose(sg.metrics["virial_gravity"], vg, rel_tol=1e-4, abs_tol=1e-6)
    assert math.isclose(sp.metrics["virial_pressure"], vp, rel_tol=1e-4, abs_tol=1e-6)
    assert sg.metrics["virial_gravity"] < 0 < sp.metrics["virial_pressure"]


def test_inert_path_unchanged_without_a_ledger():
    """pac_kappa=None still runs and the new keys are present and finite (no behavioural change)."""
    eng, rows = _run(P.ParticleConfig(**SMALL), 20)
    assert "transfer_growth" not in rows[-1]
    assert math.isfinite(rows[-1]["virial_gravity"]) and math.isfinite(rows[-1]["virial_pressure"])
    assert eng.state.work_p_i is not None
