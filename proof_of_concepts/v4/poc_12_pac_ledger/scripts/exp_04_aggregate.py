#!/usr/bin/env python3
"""exp_04 — aggregate the exp_03 runs into one grid JSON for dawn-field-theory's scorer.

Aggregates only: per-run window means over the registered window t in [10, 15] on the whole set at
matched_res(n); the random-field floor at the same count and resolution (uniform positions, 20
draws, mean and std of percolation and xi_u) beside them. It does not score: thresholds, verdicts
and kills live in milestone-r/scripts/exp_29_pac_ledger.py, byte-equal to the sealed registration.

    python .../exp_04_aggregate.py [--size proxy|full|all] [--results-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))
import numpy as np  # noqa: E402
from structure import web_metrics, cic_deposit, connectivity_at_occupancy  # noqa: E402
from worldmodel import matched_res  # noqa: E402

WINDOW = (10.0, 15.0)


def field_from(pos, box, res, dims=3):
    idx = np.clip((pos / box * res).astype(int), 0, res - 1); flat = idx[:, 0]
    for k in range(1, dims): flat = flat * res + idx[:, k]
    return np.bincount(flat, minlength=res ** dims).astype(float).reshape(*([res] * dims))


def random_floor(n, box, res, draws=20, seed=7):
    rng = np.random.RandomState(seed); p, x, o = [], [], []
    conn = {q: [] for q in (0.05, 0.10, 0.20)}
    for _ in range(draws):
        u = rng.uniform(0, box, size=(n, 3))
        w = web_metrics(field_from(u, box, res)); p.append(w["percolation"]); x.append(w["xi_u"]); o.append(w["occupancy"])
        C = cic_deposit(u, box, res)
        for q in conn: conn[q].append(connectivity_at_occupancy(C, q))
    out = dict(percolation_mean=float(np.mean(p)), percolation_std=float(np.std(p, ddof=1)), xi_u_mean=float(np.mean(x)),
               xi_u_std=float(np.std(x, ddof=1)), occupancy_mean=float(np.mean(o)), draws=draws)
    for q, v in conn.items():
        out[f"conn_q{int(q * 100):02d}_mean"] = float(np.mean(v)); out[f"conn_q{int(q * 100):02d}_std"] = float(np.std(v, ddof=1))
    return out


def wmean(marks, key):
    xs = [m[key] for m in marks if WINDOW[0] <= m["sim_time"] <= WINDOW[1] + 1e-9 and key in m]
    return float(np.mean(xs)) if xs else float("nan")   # nan when a run predates the key (exp_29/30 runs lack conn_*)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="all")
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results"); a = ap.parse_args()
    files = sorted(p for p in a.results_dir.glob("exp_03_ledger_arms_*.json") if "_grid_" not in p.name)
    runs = []
    for p in files:
        d = json.loads(p.read_text())
        if a.size != "all" and d["size"] != a.size: continue
        d["_file"] = p.name; d["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest(); runs.append(d)
    if not runs: print("no exp_03 runs found"); return 1
    floors = {}
    for d in runs:
        mk = d["marks"]; n = d["config"]["n"]; box = d["config"]["box"]; res = mk[0]["matched_res"]
        if d["size"] not in floors: floors[d["size"]] = random_floor(n, box, res)
        d["_summary"] = dict(perc=wmean(mk, "percolation"), xi_u=wmean(mk, "xi_u"), occ=wmean(mk, "occupancy"),
                             ke_over_u=wmean(mk, "ke_over_u"), total_pac=wmean(mk, "total_pac"), e_sec=wmean(mk, "sec_energy_int"),
                             ke_over_u_max=max([m["ke_over_u"] for m in mk if WINDOW[0] <= m["sim_time"] <= WINDOW[1] + 1e-9] or [float("nan")]),
                             budget_frac_end=mk[-1]["budget_frac"], budget_bound_frac_max=d["bounds"].get("budget_bound_frac_max"),
                             budget_exhausted_tick=d["bounds"].get("budget_exhausted_tick"),
                             budget_exhausted_time=next((m["sim_time"] for m in mk if m["budget_frac"] == m["budget_frac"] and m["budget_frac"] < 0.01), None),
                             work_pressure_over_p0=(mk[-1]["work_pressure_cum"] / d["budget0"] if d.get("budget0") else None),
                             sec_transfer_cum=mk[-1]["sec_transfer_cum"], closure_pac_max=max(m["closure_pac"] for m in mk[1:]),
                             transfer_residual_max=max(m["transfer_residual"] for m in mk), at_cap_max=d["bounds"]["at_cap_frac_max"],
                             floor_ticks=d["bounds"]["ticks_at_dt_floor"], finite=d["finite"], perc_peak=max(m["percolation"] for m in mk),
                             conn_q05=wmean(mk, "conn_q05"), conn_q10=wmean(mk, "conn_q10"), conn_q20=wmean(mk, "conn_q20"),
                             cv=wmean(mk, "cv"), void=wmean(mk, "void"),   # window means of the recorded one-point stats (exp_31 T2/T3)
                             virial_gravity=wmean(mk, "virial_gravity"), virial_pressure=wmean(mk, "virial_pressure"),
                             transfer_growth_cum=mk[-1].get("transfer_growth_cum"), transfer_credit_cum=mk[-1].get("transfer_credit_cum"),
                             work_pressure_pos_frac=wmean(mk, "work_pressure_pos_frac"), loss_guard_cum=mk[-1].get("loss_guard_cum"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    grid = dict(commit=commit, window=WINDOW, n_runs=len(runs), floors=floors,
                runs=[{k: v for k, v in d.items() if k not in ("marks", "config")} | {"config": d["config"]} for d in runs])
    out = a.results_dir / f"exp_03_ledger_arms_grid_{a.size}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(grid, indent=1, default=str), encoding="utf-8")
    print(f"  {len(runs)} runs -> results/{out.name}; floors: " + ", ".join(f"{k}: perc {v['percolation_mean']:.3f}±{v['percolation_std']:.3f}" for k, v in floors.items()))
    for d in sorted(runs, key=lambda r: (r["size"], r["seed"], -1 if r["kappa"] is None else r["kappa"])):
        s = d["_summary"]
        print(f"  {d['size']:>5} k={d['kappa_label']:>4} s={d['seed']} perc={s['perc']:.3f} (peak {s['perc_peak']:.3f}) xi_u={s['xi_u']:.3f} occ={s['occ']:.3f} "
              f"KE/|U|={s['ke_over_u']:6.2f} E_tot={s['total_pac']:9.3g} P/P0_end={s['budget_frac_end'] if s['budget_frac_end']==s['budget_frac_end'] else float('nan'):.3f} "
              f"exh_t={s['budget_exhausted_time']} Wp/P0={s['work_pressure_over_p0'] if s['work_pressure_over_p0'] is not None else float('nan'):.3f} "
              f"clos={s['closure_pac_max']:.1e} at_cap={s['at_cap_max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
