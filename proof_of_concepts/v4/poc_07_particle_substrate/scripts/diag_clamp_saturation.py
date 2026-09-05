"""Is the speed clamp a stability guard, or is it the equation of motion?

A numerical guard should fire rarely. If it fires on every particle every tick, force
MAGNITUDE is discarded and only direction survives -- the substrate is then integrating a
direction field, not the force law anyone is testing.

Measures, on exp_11's own 3D config:
  * fraction of particles pinned at max_speed, over time
  * sec_pressure vs gravity magnitude
  * entropy and dense_fraction (does the SEC memory ever release?)
  * whether sec_balance can influence anything once the clamp saturates
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from particles import PHI, ParticleConfig, ParticleEngine  # noqa: E402

XI = 0.5772156649015329 + math.log(PHI)
BASE = dict(n=4000, box=60.0, r0=10.0, g=1.5, dims=3, seed=1)   # exp_11's config


def run(sec_balance, steps=300, marks=(50, 100, 200, 300)):
    cfg = ParticleConfig(sec_balance=sec_balance, **BASE)
    e = ParticleEngine(cfg)
    rows = []
    for t in range(steps + 1):
        if t in marks:
            sp = e.state.vel.norm(dim=-1)
            m = e.state.metrics
            # The guard is derived by the Integrator since 2026-09-05 (cap_eff); before that
            # it was the config constant. Same tolerance either way, so the columns compare.
            cap = float(m.get("cap_eff", cfg.max_speed if cfg.max_speed is not None else float("inf")))
            rows.append(dict(
                tick=t,
                at_cap=float((sp >= cap * 0.999).float().mean()),
                mean_speed=float(sp.mean()), p99=float(torch.quantile(sp, 0.99)),
                grav=float(m.get("gravity_force_mean", 0.0)),
                press=float(m.get("sec_pressure_mean", 0.0)),
                entropy=float(m.get("entropy_mean", 0.0)),
                dense=float(m.get("dense_fraction", 0.0))))
        if t < steps:
            e.tick()
    return rows


def main():
    print("terminal speed with no clamp:  a*dt/(1-damping)")
    c = ParticleConfig(**BASE)
    print(f"   a~9.25 (gravity at t=100), dt={c.dt}, damping={c.damping}"
          f"  ->  {9.25*c.dt/(1-c.damping):.1f}   vs max_speed = "
          f"{c.max_speed if c.max_speed is not None else 'derived: cfl*r0/dt_eff (cap_eff in metrics)'}")
    print("   any force above ~0.4 saturates the cap.\n")

    for sec in (0.35, XI / PHI, XI):
        print(f"sec_balance = {sec:.4f}")
        print(f"   {'tick':>5}{'at_cap':>9}{'mean_sp':>9}{'p99':>7}"
              f"{'gravity':>10}{'sec_press':>12}{'press/grav':>12}{'entropy':>10}{'dense_f':>9}")
        for r in run(sec):
            ratio = r["press"] / r["grav"] if r["grav"] else float("nan")
            print(f"   {r['tick']:>5}{r['at_cap']:>9.4f}{r['mean_speed']:>9.4f}{r['p99']:>7.3f}"
                  f"{r['grav']:>10.2f}{r['press']:>12.1f}{ratio:>12.1f}"
                  f"{r['entropy']:>10.1f}{r['dense']:>9.4f}")
        print()

    print("READING")
    print("  * at_cap -> 1.0000 means EVERY particle is pinned every tick: force magnitude is")
    print("    discarded and only the direction of the net force survives.")
    print("  * press/grav in the thousands means that direction is the pressure gradient,")
    print("    so scaling sec_balance changes neither the post-clamp speed nor the direction.")
    print("  * entropy rising monotonically with dense_fraction -> ~0.97 is the cause:")
    print("    SECUpdate only decays entropy on the NON-dense branch, so once nearly every")
    print("    particle is flagged dense the memory never releases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
