#!/usr/bin/env python3
"""exp_02 — where and when the budget binds on the proxy, across kappa.

Pre-seal calibration for Milestone R exp_29. Per kappa: the budget exhaustion time, the maximum
fraction of would-grow particles clipped, the entropy peak, KE/|U| at t = 5, 10, 15, the total
ledger at t = 15, and pressure work over P(0). It computes NO structure metric — it does not import
structure.py — which is what makes this calibration rather than tuning.

    python proof_of_concepts/v4/poc_12_pac_ledger/scripts/exp_02_budget_calibration_proxy.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))
import torch  # noqa: E402
from particles import CANONICAL_SINK, PHI, XI_ANALYTIC, ParticleConfig, ParticleEngine  # noqa: E402

PROXY = dict(n=1000, box=37.8, r0=10.0, g=1.5, dims=3, seed=1, sec_balance=XI_ANALYTIC / PHI, damping=1.0)
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)   # up to the unbounded engine's appetite (~11 |U0| measured)
T_END = 15.0


def run(kappa):
    cfg = ParticleConfig(**PROXY, pac_kappa=kappa)
    eng = ParticleEngine(cfg, pipeline=CANONICAL_SINK, device=torch.device("cpu"))
    marks = []; t0 = time.time(); ent_peak = 0.0
    while True:
        s = eng.tick(); m = s.metrics; t = m["sim_time"]; ent_peak = max(ent_peak, m["entropy_mean"])
        if not marks or t - marks[-1]["sim_time"] >= 1.0 or t >= T_END:
            marks.append(dict(sim_time=t, tick=eng.tick_count, ke_int=m["kinetic_int"], u_int=m["potential_int"],
                              e_sec=m["sec_energy_int"], budget_frac=m["budget_frac"], total_pac=m["total_pac"],
                              ke_over_u=(m["kinetic_int"] / abs(m["potential_int"]) if m["potential_int"] else float("nan")),
                              entropy_mean=m["entropy_mean"], at_cap=m["at_cap_frac"], dt_eff=m["dt_eff"]))
        if t >= T_END or not torch.isfinite(s.pos).all():
            break
    at = lambda tt: min(marks, key=lambda x: abs(x["sim_time"] - tt))
    return dict(kappa=kappa, ticks=eng.tick_count, wall_s=round(time.time() - t0, 1), budget0=eng.budget0,
                exhausted_tick=eng.bounds["budget_exhausted_tick"], bound_frac_max=eng.bounds["budget_bound_frac_max"],
                entropy_peak=ent_peak, ke_over_u_5=at(5)["ke_over_u"], ke_over_u_10=at(10)["ke_over_u"], ke_over_u_15=at(15)["ke_over_u"],
                total_pac_15=at(15)["total_pac"], work_press_over_p0=marks[-1] and (s.metrics["work_pressure_cum"] / max(eng.budget0, 1e-9)),
                at_cap_max=eng.bounds["at_cap_frac_max"], marks=marks)


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    print(f"  proxy {PROXY} (xi_variant=XI_ANALYTIC), kappas {KAPPAS}, to sim_time {T_END}, commit {commit}")
    print(f"  {'kappa':>5} {'ticks':>5} {'wall':>4} {'P0':>8} {'exh.tick':>8} {'bound%':>6} {'S_peak':>6} {'KE/U@5':>7} {'@10':>7} {'@15':>7} {'E_tot@15':>9} {'Wp/P0':>6} {'at_cap':>6}")
    runs = []
    for k in KAPPAS:
        r = run(k); runs.append(r)
        print(f"  {k:>5.2f} {r['ticks']:>5} {r['wall_s']:>4.0f} {r['budget0']:>8.3g} {str(r['exhausted_tick']):>8} {100*r['bound_frac_max']:>6.1f} "
              f"{r['entropy_peak']:>6.2f} {r['ke_over_u_5']:>7.2f} {r['ke_over_u_10']:>7.2f} {r['ke_over_u_15']:>7.2f} {r['total_pac_15']:>9.3g} "
              f"{r['work_press_over_p0']:>6.3f} {r['at_cap_max']:>6.3f}", flush=True)
    out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"exp_02_budget_calibration_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(dict(commit=commit, proxy=PROXY, xi_variant="XI_ANALYTIC", kappas=KAPPAS, t_end=T_END, runs=runs), indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {path.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
