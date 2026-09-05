#!/usr/bin/env python3
"""exp_03 — one arm, one seed, one run: the sink arms of Milestone R exp_28.

    python .../exp_03_sink_arms.py --size proxy --arm S  --tau 1 --seed 1
    python .../exp_03_sink_arms.py --size proxy --arm B0 --seed 1
    python .../exp_03_sink_arms.py --size proxy --arm D  --seed 1 --match-from <S run json> --baseline <B0 run json>
    python .../exp_03_sink_arms.py --size proxy --arm R  --seed 1 --schedule-from <S run json>
    python .../exp_03_sink_arms.py --size proxy --arm L  --seed 1
    python .../exp_03_sink_arms.py --size proxy --arm SL --tau 1 --seed 1

Arms: B0 no sink, damping 1.0 · S severance (tau) · D matched-energy drag (damping chosen so that
the retained KINETIC energy per particle at t_end matches S's — positive-definite, unlike the signed
total, which crosses zero as the set unbinds: rho_K = (KE/N)_S / (KE/N)_B0, damping_D =
rho_K^(dt_ref/(2 t_end)), one secant refinement if the achieved ratio is outside [0.8, 1.25];
defined by matching, never a registered coordinate) · R matched-count random severance (S's event
log replayed by sim_time) · L Landauer only · SL severance + Landauer. All arms run CANONICAL_SINK
and differ by config only.

Records at every unit of simulated time: the ledger (energy, losses, closure), the sink's
bookkeeping, and the structure of the RETAINED set at matched_res(n_alive) (occupancy beside
every percolation). Positions of the retained set at each mark go to a sidecar .npz so the
aggregator can apply the count-matched frame (a uniform random subset of B0/D at S's n_alive).
One JSON per run, append-only, with the commit and the Xi variant. Scoring lives in
dawn-field-theory (milestone-r/scripts/exp_28_dynamical_severance.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))
import numpy as np, torch  # noqa: E402
from particles import CANONICAL_SINK, PHI, XI_ANALYTIC, ParticleConfig, ParticleEngine  # noqa: E402
from structure import web_metrics  # noqa: E402
from worldmodel import matched_res  # noqa: E402

SIZES = {
    "proxy": dict(n=1000, box=37.8),          # exp_11's density at n = 1000
    "full":  dict(n=4000, box=60.0),          # exp_11's config
}
BASE = dict(r0=10.0, g=1.5, dims=3, sec_balance=XI_ANALYTIC / PHI)
XI_VARIANT = "XI_ANALYTIC"


def retained_field(eng, res):
    """Density field of the RETAINED particles only, binned like ParticleEngine.density_field."""
    s = eng.state
    alive = s.alive()
    pos, mass = s.pos[alive], s.mass[alive]
    d = pos.shape[1]
    idx = [(pos[:, k] / s.box * res).long().clamp(0, res - 1) for k in range(d)]
    flat = idx[0]
    for k in range(1, d):
        flat = flat * res + idx[k]
    f = torch.zeros(res ** d, device=pos.device)
    f.scatter_add_(0, flat, mass)
    return f.view(*([res] * d)).cpu().numpy().astype(float)


def ke_per_particle_end(path_or_marks, t_end):
    marks = json.loads(path_or_marks.read_text())["marks"] if isinstance(path_or_marks, Path) else path_or_marks
    m = min(marks, key=lambda m: abs(m["sim_time"] - t_end))
    return m["kinetic_int"] / max(m["n_alive"], 1)


def damping_estimate(s_path: Path, b0_path: Path, t_end, dt_ref=0.05):
    """D's target is S's retained KINETIC energy per particle at t_end (positive-definite; the
    signed total crosses zero when the set unbinds). rho_K = (KE/N)_S / (KE/N)_B0; pure drag
    over the run gives rho_K = damping^(2 t_end / dt_ref)."""
    kS, kB = ke_per_particle_end(s_path, t_end), ke_per_particle_end(b0_path, t_end)
    rho = kS / kB if kB > 0 else float("nan")
    if not (rho > 0) or rho >= 1.0:
        return 1.0, rho, kS        # severance did not reduce the retained KE per particle: D == B0, T3 uninformative
    return rho ** (dt_ref / (2.0 * t_end)), rho, kS


def simulate(cfg, t_end):
    """Run one arm to t_end; returns (engine, marks, pos_marks, wall_s)."""
    eng = ParticleEngine(cfg, pipeline=CANONICAL_SINK)
    marks, pos_marks, next_mark, t0 = [], [], 1.0, time.time()
    while True:
        s = eng.tick(); m = s.metrics; t = m["sim_time"]
        if t >= next_mark or t >= t_end:
            n_alive = int(m["n_alive"]); res = matched_res(max(n_alive, 8), cfg.dims)
            w = web_metrics(retained_field(eng, res))
            row = dict(tick=eng.tick_count, sim_time=t, n_alive=n_alive, matched_res=res,
                       sev_frac_cum=m.get("sev_frac_cum", 0.0), sev_count=m.get("sev_count", 0.0),
                       kinetic_int=m["kinetic_int"], potential_int=m["potential_int"], total_int=m["total_int"],
                       e_int=m["e_int"], kinetic_sev=m.get("kinetic_sev", 0.0),
                       ke_over_u=(m["kinetic_int"] / abs(m["potential_int"]) if m["potential_int"] else float("nan")),
                       entropy_mean=m["entropy_mean"], entropy_max=m["entropy_max"], dense_fraction=m["dense_fraction"],
                       at_cap_frac=m["at_cap_frac"], dt_eff=m["dt_eff"], dt_at_floor=m["dt_at_floor"],
                       accel_p99=m["accel_p99"], speed_p99=m["speed_p99"],
                       press_over_grav=(m["sec_pressure_mean"] / m["gravity_force_mean"] if m["gravity_force_mean"] else float("nan")),
                       work_gravity_cum=m.get("work_gravity_cum", 0.0), work_pressure_cum=m.get("work_pressure_cum", 0.0),
                       loss_drag_cum=m.get("loss_drag_cum", 0.0), loss_guard_cum=m.get("loss_guard_cum", 0.0),
                       loss_landauer_cum=m.get("loss_landauer_cum", 0.0),
                       loss_severance_ke_cum=m.get("loss_severance_ke_cum", 0.0),
                       loss_severance_energy_cum=m.get("loss_severance_energy_cum", 0.0),
                       closure_residual=m["closure_residual"],
                       percolation=w["percolation"], xi_u=w["xi_u"], occupancy=w["occupancy"], void=w["void"],
                       cv=w["cv"], filament=w["filament"], is_web=bool(w["is_web"]))
            marks.append(row)
            pos_marks.append(s.pos[s.alive()].cpu().numpy().astype(np.float32))
            print(f"    t={t:6.2f} tick={eng.tick_count:5d} alive={n_alive:5d} sev={row['sev_frac_cum']:.3f} "
                  f"E/N={row['e_int']:9.3g} KE/|U|={row['ke_over_u']:8.2f} at_cap={row['at_cap_frac']:.3f} "
                  f"dt={row['dt_eff']:.4f} perc={row['percolation']:.3f} xi_u={row['xi_u']:.3f} occ={row['occupancy']:.3f}", flush=True)
            next_mark += 1.0
        if t >= t_end or not torch.isfinite(s.pos).all():
            break
    return eng, marks, pos_marks, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="proxy")
    ap.add_argument("--arm", choices=["B0", "S", "D", "R", "L", "SL"], required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--md", type=float, default=0.95, help="memory_decay (the one inherited tuned rate)")
    ap.add_argument("--damping", type=float, default=None, help="explicit damping (D arm); else --match-from")
    ap.add_argument("--match-from", type=Path, default=None, help="S run JSON whose retained KE/N at t_end D matches")
    ap.add_argument("--baseline", type=Path, default=None, help="B0 run JSON of the same (size, seed): D's damping estimate")
    ap.add_argument("--schedule-from", type=Path, default=None, help="S run JSON whose event log R replays")
    ap.add_argument("--t-end", type=float, default=15.0)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    a = ap.parse_args()

    cfg_kw = dict(**SIZES[a.size], **BASE, seed=a.seed, memory_decay=a.md, damping=1.0)
    rho, matched_tau, matched_file, target_ke, refinement = None, None, None, None, None
    if a.arm in ("S", "SL"):
        assert a.tau is not None, "--tau required for S/SL"
        cfg_kw["sev_tau"] = a.tau
    if a.arm in ("L", "SL"):
        cfg_kw["landauer"] = True
    if a.arm == "D":
        if a.damping is not None:
            cfg_kw["damping"] = a.damping
        else:
            assert a.match_from and a.baseline, "--damping, or --match-from S + --baseline B0, required for D"
            src = json.loads(a.match_from.read_text()); b0 = json.loads(a.baseline.read_text())
            for r in (src, b0):
                assert r["seed"] == a.seed and r["size"] == a.size, "D must be matched within its own (size, seed)"
            assert src["arm"] == "S" and b0["arm"] == "B0"
            cfg_kw["damping"], rho, target_ke = damping_estimate(a.match_from, a.baseline, a.t_end)
            matched_tau, matched_file = src["tau"], a.match_from.name
    if a.arm == "R":
        assert a.schedule_from, "--schedule-from required for R"
        src = json.loads(a.schedule_from.read_text())
        assert src["seed"] == a.seed and src["size"] == a.size, "R must replay the S run of its own (size, seed)"
        cfg_kw["sev_tau"] = src["config"].get("sev_tau") or 1.0     # any non-None value arms the operator
        cfg_kw["sev_mode"] = "random"
        cfg_kw["sev_schedule"] = [(e["sim_time"], e["count"]) for e in src["events"]]
        a.tau = matched_tau = src["tau"]; matched_file = a.schedule_from.name   # R carries the tau it replays
    cfg = ParticleConfig(**cfg_kw)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    label = f"{a.size}_{a.arm}" + (f"_tau{a.tau:g}" if a.tau is not None else "") + f"_s{a.seed}_md{a.md:g}"
    print(f"  {label}: n={cfg.n} box={cfg.box} damping={cfg.damping:.6f} sev_tau={cfg.sev_tau} mode={cfg.sev_mode} "
          f"landauer={cfg.landauer} xi_variant={XI_VARIANT} commit={commit}"
          + (f" rho_K={rho:.4f} target_KE/N={target_ke:.4g}" if rho is not None else ""), flush=True)

    eng, marks, pos_marks, wall = simulate(cfg, a.t_end)
    if a.arm == "D" and target_ke is not None and cfg.damping < 1.0:
        # one pre-declared secant refinement if the achieved KE/N misses the target by more than x1.25
        achieved = ke_per_particle_end(marks, a.t_end); ratio = achieved / target_ke if target_ke > 0 else float("nan")
        refinement = dict(damping_first=cfg.damping, ke_ratio_first=ratio)
        if not (0.8 <= ratio <= 1.25):
            damping2 = min(1.0, cfg.damping * (target_ke / achieved) ** (0.05 / (2.0 * a.t_end)))
            print(f"  D refinement: achieved/target = {ratio:.3f}; damping {cfg.damping:.6f} -> {damping2:.6f}", flush=True)
            cfg_kw["damping"] = damping2; cfg = ParticleConfig(**cfg_kw)
            eng, marks, pos_marks, wall2 = simulate(cfg, a.t_end); wall += wall2
            refinement["damping_second"] = damping2
            refinement["ke_ratio_second"] = ke_per_particle_end(marks, a.t_end) / target_ke
    s = eng.state; t0 = time.time() - wall
    finite = bool(torch.isfinite(s.pos).all())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"exp_03_sink_arms_{label}_{stamp}"
    cfg_rec = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not hasattr(v, "__dict__")}
    out = dict(commit=commit, xi_variant=XI_VARIANT, size=a.size, arm=a.arm, tau=a.tau, seed=a.seed, md=a.md,
               damping=cfg.damping, rho_matched=rho, matched_tau=matched_tau, matched_file=matched_file,
               target_ke_per_particle=target_ke, refinement=refinement,
               t_end=a.t_end, ticks=eng.tick_count, wall_s=round(wall, 1),
               finite=finite, bounds=eng.bounds, events=eng.events, config=cfg_rec, marks=marks,
               pos_sidecar=f"{stem}_pos.npz")
    (a.out_dir / f"{stem}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(a.out_dir / f"{stem}_pos.npz", **{f"t{i}": p for i, p in enumerate(pos_marks)},
                        sim_times=np.array([m["sim_time"] for m in marks]))
    print(f"  wrote results/{stem}.json (+ _pos.npz)  [{eng.tick_count} ticks, {wall:.0f}s, finite={finite}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
