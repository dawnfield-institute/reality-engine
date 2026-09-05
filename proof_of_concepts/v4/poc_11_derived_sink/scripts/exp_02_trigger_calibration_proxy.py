#!/usr/bin/env python3
"""exp_02 — where and how often the severance trigger fires on the proxy, WITH removal, across tau.

Pre-seal calibration for Milestone R exp_28. It reports, per tau: the cumulative severed fraction,
the sim_time and KE/|U| at first firing, the fired/retained kinetic ratio, and the retained count
at t = 10. It computes NO structure metric — it does not import structure.py — which is what makes
this calibration rather than tuning: the tau set is fixed from firing rates alone, never from
what structure the arms then show.

    python proof_of_concepts/v4/poc_11_derived_sink/scripts/exp_02_trigger_calibration_proxy.py
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
TAUS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
T_END = 15.0


def run(tau):
    cfg = ParticleConfig(**PROXY, sev_tau=tau)
    eng = ParticleEngine(cfg, pipeline=CANONICAL_SINK, device=torch.device("cpu"))
    first = None; marks = []; t0 = time.time()
    while True:
        s = eng.tick(); m = s.metrics; t = m["sim_time"]
        if first is None and m.get("sev_count", 0) > 0:
            first = dict(sim_time=t, tick=eng.tick_count, ke_over_u=(m["kinetic_int"] / abs(m["potential_int"]) if m["potential_int"] else float("nan")),
                         sev_ke_ratio=m.get("sev_ke_ratio"), count=int(m["sev_count"]))
        if not marks or t - marks[-1]["sim_time"] >= 1.0 or t >= T_END:
            marks.append(dict(sim_time=t, tick=eng.tick_count, n_alive=m["n_alive"], sev_frac_cum=m.get("sev_frac_cum", 0.0),
                              ke_int=m["kinetic_int"], u_int=m["potential_int"], entropy_mean=m["entropy_mean"],
                              at_cap=m["at_cap_frac"], dt_eff=m["dt_eff"]))
        if t >= T_END or not torch.isfinite(s.pos).all():
            break
    n_ret_10 = next((x["n_alive"] for x in marks if x["sim_time"] >= 10.0), marks[-1]["n_alive"])
    return dict(tau=tau, ticks=eng.tick_count, wall_s=round(time.time() - t0, 1), first_firing=first,
                sev_frac_cum=marks[-1]["sev_frac_cum"], n_alive_end=marks[-1]["n_alive"], n_alive_t10=n_ret_10,
                at_cap_max=eng.bounds["at_cap_frac_max"], dt_floor_ticks=eng.bounds["ticks_at_dt_floor"],
                events=len(eng.events), marks=marks)


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    print(f"  proxy {PROXY} (xi_variant=XI_ANALYTIC), taus {TAUS}, to sim_time {T_END}, commit {commit}")
    print(f"  {'tau':>5} {'ticks':>6} {'wall':>5} {'first t':>8} {'KE/|U|@1st':>10} {'ke_ratio':>8} {'sev_frac':>8} {'n@t10':>6} {'n_end':>6} {'at_cap':>6}")
    runs = []
    for tau in TAUS:
        r = run(tau); runs.append(r); f = r["first_firing"]
        print(f"  {tau:>5.1f} {r['ticks']:>6} {r['wall_s']:>5.0f} {(f['sim_time'] if f else float('nan')):>8.2f} "
              f"{(f['ke_over_u'] if f else float('nan')):>10.2f} {(f['sev_ke_ratio'] if f and f['sev_ke_ratio'] is not None else float('nan')):>8.2f} "
              f"{r['sev_frac_cum']:>8.3f} {r['n_alive_t10']:>6} {r['n_alive_end']:>6} {r['at_cap_max']:>6.3f}", flush=True)
    out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"exp_02_trigger_calibration_{stamp}.json"
    path.write_text(json.dumps(dict(commit=commit, proxy=PROXY, xi_variant="XI_ANALYTIC", taus=TAUS, t_end=T_END, runs=runs),
                               indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {path.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
