#!/usr/bin/env python3
"""exp_04 — exp_11's config on the substrate that integrates its force law.

Evidence for `.spec/v4-particle-substrate.spec.md`, not a physics claim. Two kinds of number
are recorded and they are kept apart:

  HYGIENE (acceptance; must hold)   at_cap_frac, dt_eff, dt_at_floor, entropy bounded and
                                    released, no NaN
  PHYSICS (measurements; reported)  press/grav, kinetic, speed_p99, xi_u, percolation,
                                    occupancy, and M17's connectivity_length when importable

The physics numbers are handed to the M18 dynamics conversation as measurements. Whether the
substrate "moves" with sec_balance is read as: between-arm range > 2x the within-seed std over
three seeds. A null there is a finding about the force law, not a failure of the repair.

Runs to matched SIMULATED time (sim_time = 15 = 300 ticks x 0.05 under the old fixed step),
because the repaired step is adaptive and tick counts no longer compare across arms.

    python proof_of_concepts/v4/poc_07_particle_substrate/scripts/exp_04_integrator_owns_dt.py
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
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))
import numpy as np, torch  # noqa: E402
from particles import PHI, ParticleConfig, ParticleEngine  # noqa: E402
from structure import web_metrics  # noqa: E402
from worldmodel import matched_res  # noqa: E402

XI = 0.5772156649015329 + math.log(PHI)
BASE = dict(n=4000, box=60.0, r0=10.0, g=1.5, dims=3)            # exp_11's config
ARMS = [("0.35", 0.35), ("Xi/phi", XI / PHI), ("1.0", 1.0), ("1.25", 1.25)]
SEEDS = [1, 2, 3]
T_END, MARK_EVERY = 15.0, 1.0                                     # matched simulated time

try:  # M17's estimator lives in the sibling repo; report it when present, never require it
    sys.path.insert(0, str(REPO.parent / "dawn-field-theory" / "experiments" / "milestones"
                          / "milestone17" / "core"))
    from criticality import connectivity_length  # type: ignore
except Exception:                                                 # noqa: BLE001
    connectivity_length = None


def occupancy_field(eng, res, overdensity=2.0):
    F = eng.density_field(res).cpu().numpy().astype(float)
    return F, F > overdensity * F.mean()


def run(sec: float, seed: int):
    cfg = ParticleConfig(sec_balance=sec, seed=seed, **BASE)
    eng = ParticleEngine(cfg)
    res = matched_res(cfg.n, cfg.dims)
    marks, next_mark, t0 = [], MARK_EVERY, time.time()
    while True:
        s = eng.tick()
        m = s.metrics
        if m["sim_time"] >= next_mark or m["sim_time"] >= T_END:
            F, occ = occupancy_field(eng, res)
            w = web_metrics(F)
            row = dict(tick=eng.tick_count, sim_time=m["sim_time"],
                       at_cap_frac=m["at_cap_frac"], dt_eff=m["dt_eff"], dt_at_floor=m["dt_at_floor"],
                       cap_eff=m["cap_eff"], accel_p99=m["accel_p99"], speed_p99=m["speed_p99"],
                       entropy_mean=m["entropy_mean"], entropy_max=m["entropy_max"],
                       dense_fraction=m["dense_fraction"], kinetic=m["kinetic"],
                       gravity=m["gravity_force_mean"], pressure=m["sec_pressure_mean"],
                       press_over_grav=(m["sec_pressure_mean"] / m["gravity_force_mean"]
                                        if m["gravity_force_mean"] else float("nan")),
                       xi_u=w["xi_u"], percolation=w["percolation"], occupancy=w["occupancy"],
                       is_web=bool(w["is_web"]))
            if connectivity_length is not None:
                try:
                    cl = connectivity_length(occ)
                    row["connectivity_length_m17"] = float(cl[0] if isinstance(cl, tuple) else cl)
                except Exception:                                 # noqa: BLE001
                    row["connectivity_length_m17"] = None
            marks.append(row)
            next_mark += MARK_EVERY
        if m["sim_time"] >= T_END or not torch.isfinite(s.pos).all():
            break
    return dict(sec_balance=sec, seed=seed, res=res, ticks=eng.tick_count,
                wall_s=round(time.time() - t0, 1), bounds=eng.bounds,
                finite=bool(torch.isfinite(s.pos).all()), marks=marks)


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    ent_bound = None
    d = BASE["dims"]; v_ball = (math.pi ** (d / 2) / math.gamma(d / 2 + 1)) * BASE["r0"] ** d
    expected = BASE["n"] * v_ball / BASE["box"] ** d
    ent_bound = 0.1 * (BASE["n"] - expected) / (1 - ParticleConfig().memory_decay)
    print(f"  exp_11 config {BASE}, arms {[a for a, _ in ARMS]}, seeds {SEEDS}, to sim_time {T_END}, commit {commit}")
    print(f"  entropy bound {ent_bound:.1f}")
    runs = []
    print(f"  {'arm':>7} {'seed':>4} {'ticks':>6} {'wall':>6} {'at_cap_max':>10} {'dt_min':>7} {'floor':>5} "
          f"{'ent_max':>8} {'ent_end':>8} {'pg_max':>7} {'pg_avg5-15':>10} {'xi_u_end':>8} {'perc_end':>8}")
    for label, sec in ARMS:
        for seed in SEEDS:
            r = run(sec, seed); runs.append(dict(arm=label, **r))
            mk = r["marks"]; late = [x for x in mk if 5.0 <= x["sim_time"] <= 15.0]
            pg_avg = float(np.nanmean([x["press_over_grav"] for x in late])) if late else float("nan")
            print(f"  {label:>7} {seed:>4} {r['ticks']:>6} {r['wall_s']:>6.0f} {r['bounds']['at_cap_frac_max']:>10.3f} "
                  f"{r['bounds']['dt_eff_min']:>7.4f} {r['bounds']['ticks_at_dt_floor']:>5} "
                  f"{max(x['entropy_max'] for x in mk):>8.1f} {mk[-1]['entropy_mean']:>8.2f} "
                  f"{max(x['press_over_grav'] for x in mk):>7.1f} {pg_avg:>10.2f} "
                  f"{mk[-1]['xi_u']:>8.3f} {mk[-1]['percolation']:>8.3f}", flush=True)

    # hygiene acceptance (spec) and the sec_balance response (measurement)
    acc = dict(
        at_cap_frac_le_0p02=all(r["bounds"]["at_cap_frac_max"] <= 0.02 for r in runs),
        entropy_within_bound=all(max(x["entropy_max"] for x in r["marks"]) <= ent_bound for r in runs),
        entropy_released=all(r["marks"][-1]["entropy_mean"] <= 0.5 * max(x["entropy_mean"] for x in r["marks"]) for r in runs),
        press_over_grav_le_100=all(max(x["press_over_grav"] for x in r["marks"]) <= 100 for r in runs),
        no_nan=all(r["finite"] for r in runs),
        dt_floor_frac_le_0p15=all(r["bounds"]["ticks_at_dt_floor"] / max(r["ticks"], 1) <= 0.15 for r in runs),
    )
    resp = {}
    for key in ("xi_u", "percolation"):
        by_arm = {a: [r["marks"][-1][key] for r in runs if r["arm"] == a] for a, _ in ARMS}
        means = {a: float(np.mean(v)) for a, v in by_arm.items()}
        within = float(np.mean([np.std(v) for v in by_arm.values()]))
        rng = max(means.values()) - min(means.values())
        resp[key] = dict(arm_means=means, within_seed_std=within, between_arm_range=rng,
                         moves=bool(rng > 2 * within))
    print("\n  HYGIENE:", acc)
    print("  RESPONSE to sec_balance (end state, 3 seeds):", json.dumps(resp, indent=None))

    out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"exp_04_integrator_owns_dt_{stamp}.json"
    path.write_text(json.dumps(dict(commit=commit, base=BASE, arms=ARMS, seeds=SEEDS, t_end=T_END,
                                    entropy_bound=ent_bound, hygiene=acc, response=resp, runs=runs),
                               indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {path.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
