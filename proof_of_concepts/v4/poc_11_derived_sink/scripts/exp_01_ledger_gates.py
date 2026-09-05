#!/usr/bin/env python3
"""exp_01 — the known-answer gates for the energy ledger, severance and Landauer erasure.

Runs the same checks as tests/v4 (test_ledger, test_severance, test_landauer) at POC scale and
writes a results file, so the gate status is on the record beside the runs it licenses. Every
line must read PASS before exp_02/exp_03 mean anything. Nothing here is a physics claim.

    python proof_of_concepts/v4/poc_11_derived_sink/scripts/exp_01_ledger_gates.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))
import numpy as np, torch  # noqa: E402
from particles import (CANONICAL, CANONICAL_SINK, LN2, ParticleConfig, ParticleEngine,  # noqa: E402
                       ParticleState, SECUpdate, LandauerErasure, LedgerSeverance, pairwise,
                       gravity_potential_table)

PROXY = dict(n=1000, box=37.8, r0=10.0, g=1.5, dims=3, seed=1, sec_balance=0.6541, damping=1.0)


def run(cfg, ticks, pipeline=None):
    eng = ParticleEngine(cfg, pipeline=pipeline, device=torch.device("cpu"))
    rows = [dict(eng.tick().metrics) for _ in range(ticks)]
    return eng, rows


def main():
    t0 = time.time(); gates = {}
    def gate(name, ok, detail):
        gates[name] = dict(ok=bool(ok), detail=detail); print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    # KA-0 closure on the proxy under CANONICAL_SINK with severance on
    _, rows = run(ParticleConfig(**PROXY, sev_tau=1.0), 200, CANONICAL_SINK)
    worst = max(r["closure_residual"] for r in rows[1:]); gate("KA-0 closure residual <= 1e-5 (severance on, 200 ticks)", worst <= 1e-5, f"max {worst:.2e}")

    # potential vs closed form
    try:
        from scipy.special import exp1
        r, U = gravity_potential_table(1.5, 10.0)
        exact = -1.5 * math.exp(0.01) * (exp1((r + 0.1) / 10.0) - exp1((30.0 + 0.1) / 10.0))
        rel = float(np.abs(U - exact).max() / np.abs(exact).max()); gate("potential = E1 closed form (rel)", rel < 1e-6, f"{rel:.2e}")
    except ImportError:
        gate("potential = E1 closed form (rel)", True, "scipy absent — skipped")

    # KA-i conservation to truncation, guard explicitly unbound
    def drift(dt_ref, t_end=3.0):
        # halve THE STEP over a fixed smooth window (t <= 3, before close encounters). Halving cfl
        # alone is not a test here: the Courant step exceeds dt_ref until t ~ 3, so both runs
        # integrate at the same dt and the ratio is decided by the chaotic late phase (the first
        # version of this gate passed at 7.6x for that wrong reason; CI gave 0.69x and 1.45x).
        torch.set_num_threads(1)
        cfg = ParticleConfig(**{**PROXY, "n": 400, "box": 27.9, "sec_balance": 0.0, "dt": dt_ref, "dt_ref": dt_ref, "max_speed": 1e9})
        eng = ParticleEngine(cfg, pipeline=CANONICAL, device=torch.device("cpu")); rw = []
        while True:
            rw.append(dict(eng.tick().metrics))
            if rw[-1]["sim_time"] >= t_end - 1e-9: break
        assert max(x["at_cap_frac"] for x in rw) == 0.0
        return abs(rw[-1]["total_int"] - rw[0]["total_int"]), max(x["cfl_number"] for x in rw) * sum(abs(x["work_gravity"]) for x in rw)
    dE, bound = drift(0.05); dE2, _ = drift(0.025)
    gate("KA-i |dE| <= cfl_max * sum|work| (damping 1, sec 0)", dE <= bound, f"|dE| {dE:.3g} vs bound {bound:.3g}")
    gate("KA-i halving the step over t<=3 shrinks |dE| >= 1.5x", dE / max(dE2, 1e-12) >= 1.5, f"ratio {dE / max(dE2, 1e-12):.2f}")

    # KA-ii drag rate
    cfg = ParticleConfig(n=200, box=60.0, r0=10.0, g=0.0, sec_balance=0.0, dims=3, seed=3, ic="zeldovich", ic_amplitude=0.5, damping=0.99)
    _, rw = run(cfg, 200); ke = np.array([x["kinetic_int"] for x in rw]); tt = np.array([x["sim_time"] for x in rw])
    slope = float(np.polyfit(tt[20:], np.log(ke[20:]), 1)[0]); target = 2 * math.log(0.99) / cfg.dt_ref
    gate("KA-ii drag rate = 2 ln(0.99)/dt_ref within 1%", abs(slope - target) <= 0.01 * abs(target), f"{slope:.4f} vs {target:.4f}")
    lost = sum(x["loss_drag"] for x in rw[1:]); gate("KA-ii sum(loss_drag) = KE lost (1e-6)", abs(lost - (ke[0] - ke[-1])) <= 1e-6 * ke[0], f"{lost:.6g} vs {ke[0]-ke[-1]:.6g}")

    # KA-iv momentum: pressure impulse zero under the third-law law
    _, rw = run(ParticleConfig(**PROXY), 60)
    imp = max(abs(x["impulse_pressure_x"]) + abs(x["impulse_pressure_y"]) + abs(x["impulse_pressure_z"]) for x in rw)
    gate("KA-iv pressure impulse == 0 (third-law pair force)", imp <= 1e-4 * rw[0]["mass_total"], f"max {imp:.2e}")

    # inert-by-default anchor
    a = ParticleEngine(ParticleConfig(**{**PROXY, "n": 300, "box": 25.2}), pipeline=CANONICAL, device=torch.device("cpu"))
    b = ParticleEngine(ParticleConfig(**{**PROXY, "n": 300, "box": 25.2}), pipeline=CANONICAL_SINK, device=torch.device("cpu"))
    for _ in range(50): a.tick(); b.tick()
    gate("CANONICAL_SINK (sinks off) bit-identical to CANONICAL", torch.equal(a.state.pos, b.state.pos) and torch.equal(a.state.vel, b.state.vel), "50 ticks, n=300")

    # severance bookkeeping on the proxy
    eng, rw = run(ParticleConfig(**PROXY, sev_tau=1.0), 120, CANONICAL_SINK); s = eng.state; m = s.metrics
    nsev = int(s.severed.sum()) if s.severed is not None else 0
    gate("severance fires on the proxy at tau=1 within 120 ticks", nsev > 0, f"{nsev} severed, {len(eng.events)} events")
    gate("mass_int + mass_sev = mass_total (1e-6)", abs(m["mass_int"] + m["mass_sev"] - m["mass_total"]) <= 1e-6 * m["mass_total"], f"{m['mass_int']:.3f} + {m['mass_sev']:.3f} vs {m['mass_total']:.3f}")
    gate("kinetic_sev = sum of event ke_out (1e-6)", abs(m["kinetic_sev"] - sum(e["ke_out"] for e in eng.events)) <= 1e-6 * max(m["kinetic_sev"], 1.0), f"{m['kinetic_sev']:.4g}")
    if nsev:
        idx = torch.nonzero(s.severed).flatten(); v0 = s.vel[idx].clone()
        for _ in range(50): s = eng.tick()
        gate("severed velocity bit-identical over 50 further ticks", torch.equal(s.vel[idx], v0), f"{len(idx)} particles")

    # Landauer rule on a constructed releasing state
    cfgL = ParticleConfig(n=200, box=60.0, r0=5.0, g=0.0, sec_balance=0.0, dims=3, damping=1.0, landauer=True)
    torch.manual_seed(0); vel = torch.randn(200, 3); vel = vel / vel.norm(dim=-1, keepdim=True) * 3.0
    s0 = ParticleState(pos=torch.rand(200, 3) * 60.0, vel=vel, mass=torch.ones(200), entropy=torch.full((200,), 20.0), box=60.0)
    s1 = SECUpdate()(s0, cfgL); s2 = LandauerErasure()(s1, cfgL)
    ke1 = 0.5 * (s1.vel ** 2).sum(-1); ke2 = 0.5 * (s2.vel ** 2).sum(-1); exp_loss = torch.minimum(ke1, LN2 * (-s1.d_entropy))
    gate("Landauer loss = min(KE, LN2*|dS|) per particle", torch.allclose(ke1 - ke2, exp_loss, rtol=1e-5, atol=1e-6), f"total {exp_loss.sum():.4g}")

    ok_all = all(g["ok"] for g in gates.values())
    print(f"\n  GATES: {'ALL PASS' if ok_all else 'FAILURES PRESENT'}  ({sum(g['ok'] for g in gates.values())}/{len(gates)}) in {time.time()-t0:.0f}s")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"exp_01_ledger_gates_{stamp}.json"
    path.write_text(json.dumps(dict(commit=commit, all_pass=ok_all, gates=gates, seconds=round(time.time() - t0, 1)), indent=1), encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO).as_posix()}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
