#!/usr/bin/env python3
"""exp_01 — one arm, one seed, one run: the R2 arms (dawn-field-theory Milestone R exp_33).

    python .../exp_01_arms.py --kappa 0.5 --arm B --seed 13
    python .../exp_01_arms.py --kappa 0.5 --arm S --seed 13 --t-arm 8
    python .../exp_01_arms.py --kappa 0.5 --arm R --seed 13 --schedule-from <S run json>
    python .../exp_01_arms.py --kappa 0.5 --arm D --seed 13 --match-from <S run json> --baseline <B run json>

The ledgered substrate of POC-12 (pac_kappa = kappa, damping 1.0, sec_balance Xi/phi — inert in the
ledgered engine, spec R4') with the local-edge severance of `particles.LedgerSeverance`
(sev_mode = "local_edge": a retained particle severs the tick its cumulative pressure work first
exceeds zero at or after t_arm). Arms: B no severance · S local-edge severance · R count-matched
random severance (S's event log replayed by sim_time, same bookkeeping) · D energy-matched drag
(damping chosen so the retained KINETIC energy per particle at t_end matches S's, POC-11's rule:
rho_K = (KE/N)_S / (KE/N)_B, damping = rho_K^(dt_ref / (2 t_end)); defined by matching, never a
registered coordinate). All arms run CANONICAL_SINK and differ by config only.

Records at every unit of simulated time: the ledger (energies, works, the pair-form virial terms
over the RETAINED set, gross transfer legs), the severance bookkeeping (count, energy out and its
sign, the fired particles' unbound fraction), the structure of the retained set — connectivity at
fixed occupancy on a CIC count field at matched_res(n_alive) beside the legacy percolation — and,
at the last mark, the severed set's kinetic energies (frozen at severance: a severed particle
interacts with nothing) for the carrier statistic. Positions, entropies, per-particle potential
and works of the whole set at each mark go to a sidecar. One JSON per run, append-only, with the
commit. Scoring lives in dawn-field-theory (milestone-r/scripts/exp_33_r2.py).
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
from structure import cic_deposit, connectivity_at_occupancy, web_metrics  # noqa: E402
from worldmodel import matched_res  # noqa: E402

SIZES = {"proxy": dict(n=1000, box=37.8), "full": dict(n=4000, box=60.0)}
BASE = dict(r0=10.0, g=1.5, dims=3, sec_balance=XI_ANALYTIC / PHI, damping=1.0)
XI_VARIANT = "XI_ANALYTIC"
CONN_Q = (0.05, 0.10, 0.20)


def density_field(pos, mass, box, res):
    idx = [(pos[:, k] / box * res).long().clamp(0, res - 1) for k in range(pos.shape[1])]
    flat = idx[0]
    for k in range(1, pos.shape[1]): flat = flat * res + idx[k]
    f = torch.zeros(res ** pos.shape[1], device=pos.device); f.scatter_add_(0, flat, mass)
    return f.view(*([res] * pos.shape[1])).cpu().numpy().astype(float)


def ke_per_particle_end(path, t_end):
    marks = json.loads(Path(path).read_text())["marks"]
    m = min(marks, key=lambda m: abs(m["sim_time"] - t_end))
    return m["kinetic_int"] / max(m["n_alive"], 1)


def damping_estimate(s_path, b_path, t_end, dt_ref=0.05):
    """POC-11's rule: D's target is S's retained kinetic energy per particle at t_end."""
    kS, kB = ke_per_particle_end(s_path, t_end), ke_per_particle_end(b_path, t_end)
    rho = kS / kB if kB > 0 else float("nan")
    if not (rho > 0) or rho >= 1.0:
        return 1.0, rho, kS
    return rho ** (dt_ref / (2.0 * t_end)), rho, kS


def simulate(cfg, t_end):
    eng = ParticleEngine(cfg, pipeline=CANONICAL_SINK)
    marks, side, next_mark, t0 = [], [], 1.0, time.time()
    while True:
        s = eng.tick(); m = s.metrics; t = m["sim_time"]
        if t >= next_mark or t >= t_end:
            alive = s.alive(); n_alive = int(alive.sum().item()); res = matched_res(max(n_alive, 8), cfg.dims)
            pos_a = s.pos[alive]; w = web_metrics(density_field(pos_a, s.mass[alive], s.box, res))
            C = cic_deposit(pos_a.detach().cpu().numpy(), cfg.box, res)
            conn = {f"conn_q{int(q * 100):02d}": connectivity_at_occupancy(C, q) for q in CONN_Q}
            row = dict(tick=eng.tick_count, sim_time=t, n_alive=n_alive, matched_res=res,
                       kinetic_int=m["kinetic_int"], potential_int=m["potential_int"], sec_energy_int=m.get("sec_energy_int", 0.0),
                       budget_int=m.get("budget_int", 0.0), total_pac=m.get("total_pac", m["total_int"]), closure_pac=m.get("closure_pac", 0.0),
                       transfer_residual=m.get("transfer_residual", 0.0), budget_bound_frac=m.get("budget_bound_frac", 0.0),
                       sec_transfer_cum=m.get("sec_transfer_cum", 0.0), transfer_growth_cum=m.get("transfer_growth_cum", 0.0),
                       transfer_credit_cum=m.get("transfer_credit_cum", 0.0),
                       ke_over_u=(m["kinetic_int"] / abs(m["potential_int"]) if m["potential_int"] else float("nan")),
                       virial_gravity=m.get("virial_gravity", float("nan")), virial_pressure=m.get("virial_pressure", float("nan")),
                       work_gravity_cum=m.get("work_gravity_cum", 0.0), work_pressure_cum=m.get("work_pressure_cum", 0.0),
                       work_pressure_pos_frac=m.get("work_pressure_pos_frac", float("nan")), work_pressure_median=m.get("work_pressure_median", float("nan")),
                       loss_drag_cum=m.get("loss_drag_cum", 0.0), loss_guard_cum=m.get("loss_guard_cum", 0.0),
                       sev_count=m.get("sev_count", 0.0), sev_frac_cum=m.get("sev_frac_cum", 0.0),
                       loss_severance_ke_cum=m.get("loss_severance_ke_cum", 0.0), loss_severance_u_cum=m.get("loss_severance_u_cum", 0.0),
                       loss_severance_energy_cum=m.get("loss_severance_energy_cum", 0.0),
                       sev_unbound_frac=m.get("sev_unbound_frac", float("nan")),
                       entropy_mean=m["entropy_mean"], dense_fraction=m["dense_fraction"], at_cap_frac=m["at_cap_frac"],
                       dt_eff=m["dt_eff"], dt_at_floor=m["dt_at_floor"], closure_residual=m["closure_residual"],
                       percolation=w["percolation"], occupancy=w["occupancy"], xi_u=w["xi_u"], void=w["void"], cv=w["cv"], **conn)
            marks.append(row)
            side.append(dict(pos=s.pos.cpu().numpy().astype(np.float32), S=s.entropy.cpu().numpy().astype(np.float32),
                             U=(s.potential_i.cpu().numpy().astype(np.float32) if s.potential_i is not None else None),
                             wp=(s.work_p_i.cpu().numpy().astype(np.float32) if s.work_p_i is not None else None),
                             wg=(s.work_g_i.cpu().numpy().astype(np.float32) if s.work_g_i is not None else None)))
            print(f"    t={t:6.2f} tick={eng.tick_count:5d} alive={n_alive:5d} sev={row['sev_frac_cum']:.3f} KE/|U|={row['ke_over_u']:6.2f} "
                  f"E_sev={row['loss_severance_energy_cum']:9.3g} vir={2*row['kinetic_int']/max(abs(row['virial_gravity'])-row['virial_pressure'],1e-9):5.2f} "
                  f"q10={row['conn_q10']:.2f} pos%={100*row['work_pressure_pos_frac']:4.1f}", flush=True)
            next_mark += 1.0
        if t >= t_end or not torch.isfinite(s.pos).all():
            break
    return eng, marks, side, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="full")
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--arm", choices=["B", "S", "R", "D"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--t-end", type=float, default=15.0)
    ap.add_argument("--t-arm", type=float, default=8.0, help="S: the declared arming time of the local-edge trigger")
    ap.add_argument("--schedule-from", type=Path, default=None, help="R: S run JSON whose event log is replayed")
    ap.add_argument("--match-from", type=Path, default=None, help="D: S run JSON (retained KE/N target)")
    ap.add_argument("--baseline", type=Path, default=None, help="D: B run JSON of the same (kappa, seed)")
    ap.add_argument("--damping", type=float, default=None, help="D: explicit damping (else --match-from + --baseline)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    a = ap.parse_args()
    cfg_kw = dict(**SIZES[a.size], **BASE, seed=a.seed, pac_kappa=a.kappa)
    rho = target_ke = None
    if a.arm == "S":
        cfg_kw.update(sev_mode="local_edge", sev_t_arm=a.t_arm)
    elif a.arm == "R":
        assert a.schedule_from, "--schedule-from <S run json> required for R"
        src = json.loads(a.schedule_from.read_text())
        cfg_kw.update(sev_tau=1.0, sev_mode="random", sev_schedule=[(e["sim_time"], e["count"]) for e in src["events"]])
    elif a.arm == "D":
        if a.damping is not None:
            cfg_kw["damping"] = a.damping
        else:
            assert a.match_from and a.baseline, "--damping, or --match-from S + --baseline B, required for D"
            cfg_kw["damping"], rho, target_ke = damping_estimate(a.match_from, a.baseline, a.t_end)
    cfg = ParticleConfig(**cfg_kw)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    label = f"{a.size}_k{a.kappa:g}_{a.arm}_s{a.seed}"
    print(f"  {label}: n={cfg.n} box={cfg.box} kappa={a.kappa} damping={cfg.damping:.6f} mode={cfg.sev_mode} t_arm={cfg.sev_t_arm} commit={commit}", flush=True)
    eng, marks, side, wall = simulate(cfg, a.t_end)
    refinement = None
    if a.arm == "D" and target_ke is not None and cfg.damping < 1.0:
        # POC-11's one pre-declared secant refinement if the achieved KE/N misses the target by more than x1.25
        achieved = marks[-1]["kinetic_int"] / max(marks[-1]["n_alive"], 1); ratio = achieved / target_ke if target_ke > 0 else float("nan")
        refinement = dict(damping_first=cfg.damping, ke_ratio_first=ratio)
        if not (0.8 <= ratio <= 1.25):
            damping2 = min(1.0, cfg.damping * (target_ke / achieved) ** (0.05 / (2.0 * a.t_end)))
            print(f"  D refinement: achieved/target = {ratio:.3f}; damping {cfg.damping:.6f} -> {damping2:.6f}", flush=True)
            cfg_kw["damping"] = damping2; cfg = ParticleConfig(**cfg_kw)
            eng, marks, side, wall2 = simulate(cfg, a.t_end); wall += wall2
            refinement["damping_second"] = damping2; refinement["ke_ratio_second"] = (marks[-1]["kinetic_int"] / max(marks[-1]["n_alive"], 1)) / target_ke
    s = eng.state; finite = bool(torch.isfinite(s.pos).all())
    sev = s.severed if s.severed is not None else torch.zeros(cfg.n, dtype=torch.bool)
    ke_i = 0.5 * s.mass * (s.vel ** 2).sum(-1)
    severed_ke = ke_i[sev].cpu().numpy().astype(float).tolist()          # frozen at severance
    severed_time = (s.sev_time[sev].cpu().numpy().astype(float).tolist() if s.sev_time is not None else [])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"exp_01_arms_{label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    cfg_rec = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not hasattr(v, "__dict__") and k != "sev_schedule"}
    out = dict(commit=commit, xi_variant=XI_VARIANT, size=a.size, kappa=a.kappa, arm=a.arm, seed=a.seed, t_end=a.t_end,
               t_arm=(a.t_arm if a.arm == "S" else None), budget0=getattr(eng, "budget0", None), ticks=eng.tick_count,
               wall_s=round(wall, 1), finite=finite, bounds=eng.bounds, config=cfg_rec, marks=marks, events=eng.events,
               drag=(dict(rho_K=rho, target_ke_per_particle=target_ke, damping=cfg.damping, refinement=refinement) if a.arm == "D" else None),
               severed=dict(count=int(sev.sum().item()), ke=severed_ke, sim_time=severed_time), pos_sidecar=f"{stem}_pos.npz")
    (a.out_dir / f"{stem}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(a.out_dir / f"{stem}_pos.npz", sim_times=np.array([m["sim_time"] for m in marks]),
                        mass=s.mass.cpu().numpy(), severed=sev.cpu().numpy(),
                        **{f"{k}{i}": v for i, sm in enumerate(side) for k, v in sm.items() if v is not None})
    print(f"  wrote results/{stem}.json (+ _pos.npz)  [{eng.tick_count} ticks, {wall:.0f}s, finite={finite}, severed={int(sev.sum())}]")


if __name__ == "__main__":
    main()
