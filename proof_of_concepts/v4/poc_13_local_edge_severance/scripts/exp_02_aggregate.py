#!/usr/bin/env python3
"""exp_02 — aggregate the R2 arms into one grid JSON with per-run summaries (dawn-field-theory
exp_33 scores it; nothing is scored here).

    python .../exp_02_aggregate.py [--results-dir DIR]

Per run, `_summary` carries: window means (t in [10, 15]) of KE/|U|, the retained set's pair-form
virial ratio 2K / (|V_g| − V_p), conn_q05/q10/q20 and percolation; the severance bookkeeping at the
end (count, fraction, ke_out, u_out, their sum and its sign, the fired particles' unbound fraction
averaged over events); the severed set's kinetic-energy spread (interquartile range over median,
frozen at severance) and its per-particle energy out; the retained kinetic energy per particle at
the end; the run's disequilibrium before arming, <dG/dt>/2<K> over t in [5, 8] (a negative number
is net infall); and the drag calibration record for D arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
WINDOW = (10.0, 15.0)
PRE_ARM = (5.0, 8.0)


def wmean(marks, key, lo, hi):
    xs = [m[key] for m in marks if lo <= m["sim_time"] <= hi + 1e-9 and key in m and m[key] == m[key]]
    return float(np.mean(xs)) if xs else float("nan")


def virial_ratio(m):
    d = abs(m["virial_gravity"]) - m["virial_pressure"]
    return 2 * m["kinetic_int"] / d if d > 0 else float("nan")


def summarise(d):
    mk = d["marks"]; end = mk[-1]
    vr = [virial_ratio(m) for m in mk if WINDOW[0] <= m["sim_time"] <= WINDOW[1] + 1e-9]
    dq = [virial_ratio(m) - 1.0 for m in mk if PRE_ARM[0] <= m["sim_time"] <= PRE_ARM[1] + 1e-9]
    ke = np.array(d["severed"]["ke"], float)
    ev = [e for e in d.get("events", [])]
    unb = [m["sev_unbound_frac"] for m in mk if m.get("sev_count", 0) > 0 and m["sev_unbound_frac"] == m["sev_unbound_frac"]]
    return dict(ke_over_u=wmean(mk, "ke_over_u", *WINDOW), virial_ratio=float(np.mean(vr)) if vr else float("nan"),
                virial_dev_pre_arm=float(np.mean(dq)) if dq else float("nan"),
                conn_q05=wmean(mk, "conn_q05", *WINDOW), conn_q10=wmean(mk, "conn_q10", *WINDOW), conn_q20=wmean(mk, "conn_q20", *WINDOW),
                perc=wmean(mk, "percolation", *WINDOW), occ=wmean(mk, "occupancy", *WINDOW),
                n_alive_end=end["n_alive"], sev_frac=end["sev_frac_cum"], sev_count=d["severed"]["count"], n_events=len(ev),
                sev_ke_out=end["loss_severance_ke_cum"], sev_u_out=end["loss_severance_u_cum"], sev_energy_out=end["loss_severance_energy_cum"],
                sev_energy_sign=(1 if end["loss_severance_energy_cum"] > 0 else -1 if end["loss_severance_energy_cum"] < 0 else 0),
                sev_energy_per_particle=(end["loss_severance_energy_cum"] / d["severed"]["count"] if d["severed"]["count"] else float("nan")),
                sev_unbound_frac_mean=float(np.mean(unb)) if unb else float("nan"),
                sev_ke_spread=(float((np.percentile(ke, 75) - np.percentile(ke, 25)) / max(np.median(ke), 1e-12)) if ke.size >= 8 else float("nan")),
                sev_ke_median=float(np.median(ke)) if ke.size else float("nan"),
                ke_per_particle_end=end["kinetic_int"] / max(end["n_alive"], 1),
                work_pressure_cum=end["work_pressure_cum"], u0=abs(mk[0]["potential_int"]), e_sec_end=end["sec_energy_int"],
                closure_pac_max=max(m["closure_pac"] for m in mk[1:]), transfer_residual_max=max(m["transfer_residual"] for m in mk),
                at_cap_max=d["bounds"]["at_cap_frac_max"], finite=d["finite"], budget_bound_frac_max=d["bounds"].get("budget_bound_frac_max"),
                drag=d.get("drag"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results"); a = ap.parse_args()
    files = sorted(p for p in a.results_dir.glob("exp_01_arms_*.json") if "_grid_" not in p.name)
    runs = []
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8")); d["_file"] = p.name; d["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        d["_summary"] = summarise(d); d.pop("marks"); runs.append(d)
    if not runs:
        print("no exp_01 runs found"); return 1
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    grid = dict(commit=commit, window=WINDOW, pre_arm=PRE_ARM, n_runs=len(runs), runs=runs)
    out = a.results_dir / f"exp_01_arms_grid_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(grid, indent=1, default=str), encoding="utf-8")
    print(f"  {len(runs)} runs -> {out.name}")
    for d in sorted(runs, key=lambda r: (r["kappa"], r["seed"], r["arm"])):
        s = d["_summary"]
        print(f"   k={d['kappa']:<4} s={d['seed']:<3} {d['arm']}: alive {s['n_alive_end']:5d} sev {s['sev_frac']:.3f} E_out {s['sev_energy_out']:+9.3g} (unbound frac {s['sev_unbound_frac_mean']:.2f}) "
              f"vir {s['virial_ratio']:5.2f} (pre {s['virial_dev_pre_arm']:+.2f}) KE/N {s['ke_per_particle_end']:7.1f} q10 {s['conn_q10']:.2f} spread {s['sev_ke_spread']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
