#!/usr/bin/env python3
"""exp_03 — one arm, one seed, one run: the ledger arms of Milestone R exp_29.

    python .../exp_03_ledger_arms.py --size proxy --kappa 0.5 --seed 1
    python .../exp_03_ledger_arms.py --size full  --kappa inf --seed 2

Arms are kappa: 0 (gravity only — the engine removed), 0.5 / 1 / 2 (the ledgered engine, the
sweep), inf (today's substrate — the unbounded engine). All run CANONICAL_SINK with severance and
Landauer off, damping 1.0, and differ by pac_kappa only. Records at every unit of simulated time
the ledger (KE, U, E_SEC, sum P, the conserved total, closure), the budget's bounds, and the
structure of the WHOLE set at matched_res(n) with occupancy beside every percolation; positions
go to a sidecar so the aggregator can draw the random-field floor at the same count. One JSON per
run, append-only, with the commit and the Xi variant. Scoring lives in dawn-field-theory
(milestone-r/scripts/exp_29_pac_ledger.py).
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
from structure import web_metrics, cic_deposit, connectivity_at_occupancy  # noqa: E402
from worldmodel import matched_res  # noqa: E402

CONN_Q = (0.05, 0.10, 0.20)   # spine, the registered occupancy, the body (exp_31)
SIZES = {"proxy": dict(n=1000, box=37.8), "full": dict(n=4000, box=60.0)}
BASE = dict(r0=10.0, g=1.5, dims=3, sec_balance=XI_ANALYTIC / PHI, damping=1.0)
XI_VARIANT = "XI_ANALYTIC"


def density_field(eng, res):
    s = eng.state; alive = s.alive(); pos, mass = s.pos[alive], s.mass[alive]; d = pos.shape[1]
    idx = [(pos[:, k] / s.box * res).long().clamp(0, res - 1) for k in range(d)]
    flat = idx[0]
    for k in range(1, d): flat = flat * res + idx[k]
    f = torch.zeros(res ** d, device=pos.device); f.scatter_add_(0, flat, mass)
    return f.view(*([res] * d)).cpu().numpy().astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZES, default="proxy")
    ap.add_argument("--kappa", required=True, help="0 | 0.5 | 1 | 2 | inf")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t-end", type=float, default=15.0)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    a = ap.parse_args()
    kappa = None if a.kappa in ("inf", "none", "None") else float(a.kappa)
    cfg = ParticleConfig(**SIZES[a.size], **BASE, seed=a.seed, pac_kappa=kappa)
    eng = ParticleEngine(cfg, pipeline=CANONICAL_SINK)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    klabel = "inf" if kappa is None else f"{kappa:g}"
    label = f"{a.size}_k{klabel}_s{a.seed}"
    print(f"  {label}: n={cfg.n} box={cfg.box} pac_kappa={kappa} P0={getattr(eng, 'budget0', None)} xi_variant={XI_VARIANT} commit={commit}", flush=True)
    res = matched_res(cfg.n, cfg.dims)
    marks, pos_marks, next_mark, t0 = [], [], 1.0, time.time()
    while True:
        s = eng.tick(); m = s.metrics; t = m["sim_time"]
        if t >= next_mark or t >= a.t_end:
            w = web_metrics(density_field(eng, res))
            C = cic_deposit(s.pos[s.alive()].detach().cpu().numpy(), cfg.box, res)   # count deposit: no mass draw, no threshold boundary
            conn = {f"conn_q{int(q * 100):02d}": connectivity_at_occupancy(C, q) for q in CONN_Q}
            row = dict(tick=eng.tick_count, sim_time=t, n_alive=int(m["n_alive"]), matched_res=res,
                       kinetic_int=m["kinetic_int"], potential_int=m["potential_int"], total_int=m["total_int"], e_int=m["e_int"],
                       sec_energy_int=m.get("sec_energy_int", 0.0), budget_int=m.get("budget_int", 0.0), budget_frac=m.get("budget_frac", float("nan")),
                       total_pac=m.get("total_pac", m["total_int"]), closure_pac=m.get("closure_pac", 0.0), transfer_residual=m.get("transfer_residual", 0.0),
                       budget_bound_frac=m.get("budget_bound_frac", 0.0), sec_transfer_cum=m.get("sec_transfer_cum", 0.0),
                       ke_over_u=(m["kinetic_int"] / abs(m["potential_int"]) if m["potential_int"] else float("nan")),
                       entropy_mean=m["entropy_mean"], entropy_max=m["entropy_max"], dense_fraction=m["dense_fraction"],
                       at_cap_frac=m["at_cap_frac"], dt_eff=m["dt_eff"], dt_at_floor=m["dt_at_floor"], accel_p99=m["accel_p99"],
                       press_over_grav=(m["sec_pressure_mean"] / m["gravity_force_mean"] if m["gravity_force_mean"] else float("nan")),
                       work_gravity_cum=m.get("work_gravity_cum", 0.0), work_pressure_cum=m.get("work_pressure_cum", 0.0),
                       loss_guard_cum=m.get("loss_guard_cum", 0.0), closure_residual=m["closure_residual"],
                       percolation=w["percolation"], xi_u=w["xi_u"], occupancy=w["occupancy"], void=w["void"], cv=w["cv"],
                       filament=w["filament"], is_web=bool(w["is_web"]), **conn)
            marks.append(row); pos_marks.append(s.pos.cpu().numpy().astype(np.float32))
            print(f"    t={t:6.2f} tick={eng.tick_count:5d} KE/|U|={row['ke_over_u']:7.2f} E_tot={row['total_pac']:9.3g} P/P0={row['budget_frac']:.3f} "
                  f"bound={row['budget_bound_frac']:.3f} clos={row['closure_pac']:.1e} perc={row['percolation']:.3f} occ={row['occupancy']:.3f} conn05/10/20={row['conn_q05']:.2f}/{row['conn_q10']:.2f}/{row['conn_q20']:.2f}", flush=True)
            next_mark += 1.0
        if t >= a.t_end or not torch.isfinite(s.pos).all():
            break
    finite = bool(torch.isfinite(s.pos).all())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"exp_03_ledger_arms_{label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    cfg_rec = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not hasattr(v, "__dict__")}
    out = dict(commit=commit, xi_variant=XI_VARIANT, size=a.size, kappa=kappa, kappa_label=klabel, seed=a.seed,
               budget0=getattr(eng, "budget0", None), t_end=a.t_end, ticks=eng.tick_count, wall_s=round(time.time() - t0, 1),
               finite=finite, bounds=eng.bounds, config=cfg_rec, marks=marks, pos_sidecar=f"{stem}_pos.npz")
    (a.out_dir / f"{stem}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(a.out_dir / f"{stem}_pos.npz", **{f"t{i}": p for i, p in enumerate(pos_marks)}, sim_times=np.array([m["sim_time"] for m in marks]))
    print(f"  wrote results/{stem}.json (+ _pos.npz)  [{eng.tick_count} ticks, {time.time()-t0:.0f}s, finite={finite}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
