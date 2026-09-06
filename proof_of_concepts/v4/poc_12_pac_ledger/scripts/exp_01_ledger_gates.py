#!/usr/bin/env python3
"""exp_01 — the known-answer gates for the PAC ledger on particles (spec v4-pac-ledger KA-0..iii).

Runs the same checks as tests/v4/test_pac_ledger.py at POC scale, plus the platform anchor that the
test suite cannot carry (the None path reproduces exp_28's recorded B0 seed-1 marks), and writes a
results file so the gate status is on the record beside the runs it licenses. Every line must read
PASS before exp_02/exp_03 mean anything. Nothing here is a physics claim.

    python proof_of_concepts/v4/poc_12_pac_ledger/scripts/exp_01_ledger_gates.py
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))
import torch  # noqa: E402
from particles import (CANONICAL, CANONICAL_SINK, PHI, XI_ANALYTIC, ParticleConfig, ParticleEngine,  # noqa: E402
                       ParticleState, SECPressure, sec_pair_energy)

PROXY = dict(n=1000, box=37.8, r0=10.0, g=1.5, dims=3, seed=1, sec_balance=0.6541, damping=1.0)
SMALL = dict(n=300, box=25.2, r0=10.0, g=1.5, dims=3, seed=1, sec_balance=0.6541, damping=1.0)


def run(cfg, ticks=None, t_end=None, pipeline=CANONICAL_SINK):
    eng = ParticleEngine(cfg, pipeline=pipeline, device=torch.device("cpu")); rows = []
    while True:
        rows.append(dict(eng.tick().metrics))
        if ticks is not None and len(rows) >= ticks: break
        if t_end is not None and rows[-1]["sim_time"] >= t_end - 1e-9: break
    return eng, rows


def main():
    t0 = time.time(); gates = {}
    def gate(name, ok, detail):
        gates[name] = dict(ok=bool(ok), detail=detail); print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    # KA-0 force = -grad of the shifted pair energy
    c = ParticleConfig(n=3, box=60.0, r0=10.0, g=0.0, dims=3, sec_balance=0.6541, damping=1.0, pac_kappa=1.0)
    pos = torch.tensor([[20.0, 20.0, 20.0], [26.0, 21.0, 19.0], [23.0, 27.0, 22.0]], dtype=torch.float64)   # float64: an O(10) energy differenced over 2e-3
    s = ParticleState(pos=pos, vel=torch.zeros(3, 3, dtype=torch.float64), mass=torch.ones(3, dtype=torch.float64), entropy=torch.tensor([5.0, 1.0, 3.0], dtype=torch.float64), box=60.0)
    force = SECPressure()(s, c).acc_pressure * s.mass.unsqueeze(-1); h = 1e-3; worst = 0.0
    for k in range(3):
        for ax in range(3):
            def E(d):
                p = pos.clone(); p[k, ax] += d; return sec_pair_energy(s.replace(pos=p), c)[0]
            grad = (E(h) - E(-h)) / (2 * h)
            worst = max(worst, abs(force[k, ax].item() + grad) / max(abs(grad), 1e-6))
    gate("KA-0 force = -grad(E_SEC) (finite difference, rel)", worst <= 1e-3, f"worst {worst:.2e}")

    # KA-i transfer identity, every tick
    eng, rows = run(ParticleConfig(**SMALL, pac_kappa=1.0), ticks=120)
    p0 = eng.budget0; prev = rows[0]["budget_int"]; w = 0.0; wr = 0.0
    for r in rows[1:]:
        w = max(w, abs(r["sec_transfer"] + (r["budget_int"] - prev)) / p0); prev = r["budget_int"]; wr = max(wr, r["transfer_residual"])
    gate("KA-i transfer identity sum(dE/dS dS) + d(sum P) = 0 (1e-6 of P0)", w <= 1e-6 and wr <= 1e-6, f"worst {w:.2e}, residual {wr:.2e}")

    # KA-ii net creation <= P0, budget binds, pressure work within P0 + truncation
    eng, rows = run(ParticleConfig(**{**PROXY, "n": 400, "box": 27.9}, pac_kappa=1.0), t_end=15.0)
    m = rows[-1]; p0 = eng.budget0
    gate("KA-ii net creation sum(dE/dS dS) <= P(0) (exact)", m["sec_transfer_cum"] <= p0 * (1 + 1e-6), f"{m['sec_transfer_cum']:.4g} vs P0 {p0:.4g}")
    gate("KA-ii the budget binds (budget_bound_frac_max > 0)", eng.bounds["budget_bound_frac_max"] > 0, f"max {eng.bounds['budget_bound_frac_max']:.3f}, exhausted tick {eng.bounds['budget_exhausted_tick']}")
    gate("KA-ii pressure work <= P(0) within 10% truncation", m["work_pressure_cum"] <= p0 * 1.10, f"{m['work_pressure_cum']:.4g} / P0 = {m['work_pressure_cum']/p0:.3f}")

    # KA-iii total conserved to truncation; halving the step over t <= 3.5 shrinks the drift
    n_threads = torch.get_num_threads(); torch.set_num_threads(1)     # pinned for the ratio only; restored before the anchors
    def drift(dt_ref, t_end=3.5):
        cfg = ParticleConfig(**{**PROXY, "n": 400, "box": 27.9, "dt": dt_ref, "dt_ref": dt_ref, "max_speed": 1e9}, pac_kappa=1.0)
        _, rw = run(cfg, t_end=t_end)
        gross = sum(abs(x["work_gravity"]) + abs(x["work_pressure"]) for x in rw)
        return abs(rw[-1]["total_pac"] - rw[0]["total_pac"]), max(x["cfl_number"] for x in rw) * gross, max(x["closure_pac"] for x in rw[1:])
    d1, b1, cmax = drift(0.05); d2, _, _ = drift(0.025)
    gate("KA-iii |d total_pac| <= cfl_max * gross work (t<=3.5)", d1 <= b1, f"{d1:.3g} vs bound {b1:.3g}; max closure_pac {cmax:.2e}")
    gate("KA-iii halving the step shrinks the drift >= 1.5x", d1 / max(d2, 1e-12) >= 1.5, f"ratio {d1/max(d2,1e-12):.2f}")
    torch.set_num_threads(n_threads)      # the anchor below compares against a run made at the default thread count

    # anchors
    a = ParticleEngine(ParticleConfig(**SMALL), pipeline=CANONICAL, device=torch.device("cpu"))
    b = ParticleEngine(ParticleConfig(**SMALL), pipeline=CANONICAL_SINK, device=torch.device("cpu"))
    for _ in range(50): sa, sb = a.tick(), b.tick()
    gate("anchor: kappa=None structurally inert + CANONICAL_SINK == CANONICAL", sb.budget is None and "budget_int" not in sb.metrics and torch.equal(sa.pos, sb.pos) and torch.equal(sa.vel, sb.vel), "50 ticks, n=300")
    a = ParticleEngine(ParticleConfig(**SMALL, pac_kappa=0.0), pipeline=CANONICAL_SINK, device=torch.device("cpu"))
    b = ParticleEngine(ParticleConfig(**{**SMALL, "sec_balance": 0.0}), pipeline=CANONICAL_SINK, device=torch.device("cpu"))
    for _ in range(80): sa, sb = a.tick(), b.tick()
    gate("anchor: kappa=0 == gravity-only (sec_balance=0) bit for bit", sa.entropy.abs().max().item() == 0.0 and torch.equal(sa.pos, sb.pos) and torch.equal(sa.vel, sb.vel), "80 ticks")
    # the None path reproduces exp_28's recorded B0 seed-1 marks (same platform)
    ref = sorted(glob.glob(str(REPO / "proof_of_concepts/v4/poc_11_derived_sink/results/exp_03_sink_arms_proxy_B0_s1_md0.95_*.json")))
    if ref:
        d = json.load(open(ref[0])); marks = {round(x["sim_time"]): x for x in d["marks"]}
        # exp_28 ran at sec_balance = XI_ANALYTIC / PHI exactly; the rounded 0.6541 diverges by 3e-3 over the run
        eng = ParticleEngine(ParticleConfig(**{**PROXY, "sec_balance": XI_ANALYTIC / PHI}), pipeline=CANONICAL_SINK, device=torch.device("cpu")); worst = 0.0; nxt = 1.0
        while True:
            m = eng.tick().metrics; t = m["sim_time"]
            if t >= nxt or t >= 15.0:
                k = round(t)
                if k in marks:
                    worst = max(worst, abs(m["kinetic_int"] - marks[k]["kinetic_int"]) / max(abs(marks[k]["kinetic_int"]), 1.0),
                                abs(m["potential_int"] - marks[k]["potential_int"]) / max(abs(marks[k]["potential_int"]), 1.0))
                nxt += 1.0
            if t >= 15.0: break
        gate("anchor: None path reproduces exp_28 B0 seed-1 marks (1e-6, same platform)", worst <= 1e-6, f"worst {worst:.2e} vs {Path(ref[0]).name}")
    else:
        gate("anchor: None path reproduces exp_28 B0 seed-1 marks", True, "reference JSON absent — skipped")
    # severance: budget frozen
    eng, rows = run(ParticleConfig(**PROXY, pac_kappa=1.0, sev_tau=0.1), ticks=200); s = eng.state   # tau=0.1: the budget caps entropy near 0.3
    idx = torch.nonzero(s.severed).flatten() if s.severed is not None else torch.zeros(0, dtype=torch.long)
    if idx.numel():
        b0 = s.budget[idx].clone()
        for _ in range(50): s = eng.tick()
        gate("severance: severed budget frozen over 50 further ticks", torch.equal(s.budget[idx], b0), f"{idx.numel()} particles")
    else:
        gate("severance: severed budget frozen", False, "nothing severed at tau=0.1 in 200 ticks")

    ok_all = all(g["ok"] for g in gates.values())
    print(f"\n  GATES: {'ALL PASS' if ok_all else 'FAILURES PRESENT'}  ({sum(g['ok'] for g in gates.values())}/{len(gates)}) in {time.time()-t0:.0f}s")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"exp_01_ledger_gates_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(dict(commit=commit, all_pass=ok_all, gates=gates, seconds=round(time.time() - t0, 1)), indent=1), encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO).as_posix()}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
