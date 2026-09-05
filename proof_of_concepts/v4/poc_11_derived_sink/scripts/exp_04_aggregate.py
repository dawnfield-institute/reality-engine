#!/usr/bin/env python3
"""exp_04 — aggregate the exp_03 runs into one grid JSON for dawn-field-theory's scorer.

Aggregates only; it applies the registered FRAME (the retained set of an S run at matched_res,
compared with a uniform random subset of B0/D at the same count, 5 draws, and with R which is
count-matched by construction) and computes the raw comparisons. It does not score: thresholds,
verdicts and kills live in milestone-r/scripts/exp_28_dynamical_severance.py, byte-equal to the
sealed registration.

    python .../exp_04_aggregate.py [--size proxy|full|all] [--results-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))
import numpy as np  # noqa: E402
from structure import web_metrics  # noqa: E402
from worldmodel import matched_res  # noqa: E402

WINDOW = (10.0, 15.0)      # the registered late-time window: means over marks, never the endpoint


def field_from(pos: np.ndarray, box: float, res: int, dims: int = 3):
    idx = np.clip((pos / box * res).astype(int), 0, res - 1)
    flat = idx[:, 0]
    for k in range(1, dims):
        flat = flat * res + idx[:, k]
    f = np.bincount(flat, minlength=res ** dims).astype(float)
    return f.reshape(*([res] * dims))


def subset_metrics(pos: np.ndarray, box: float, n_sub: int, seed: int, draws: int = 5):
    """Uniform random subsets of size n_sub, the frame's expectation for B0/D at S's count."""
    rng = np.random.RandomState(seed)
    res = matched_res(max(n_sub, 8), 3); acc = {"percolation": [], "xi_u": [], "occupancy": []}
    for _ in range(draws):
        sub = pos[rng.choice(len(pos), size=min(n_sub, len(pos)), replace=False)]
        w = web_metrics(field_from(sub, box, res))
        for k in acc: acc[k].append(float(w[k]))
    return {k: float(np.mean(v)) for k, v in acc.items()}, res


def window_mean(marks, key):
    xs = [m[key] for m in marks if WINDOW[0] <= m["sim_time"] <= WINDOW[1] + 1e-9]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="all")
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    a = ap.parse_args()
    files = sorted(p for p in a.results_dir.glob("exp_03_sink_arms_*.json"))
    runs = []
    for p in files:
        d = json.loads(p.read_text())
        if a.size != "all" and d["size"] != a.size: continue
        d["_file"] = p.name; d["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        runs.append(d)
    if not runs:
        print("no exp_03 runs found"); return 1

    # per-run late-window summaries on the run's OWN retained set
    for d in runs:
        mk = d["marks"]
        d["_summary"] = dict(
            perc=window_mean(mk, "percolation"), xi_u=window_mean(mk, "xi_u"), occ=window_mean(mk, "occupancy"),
            e_int=window_mean(mk, "e_int"), n_alive_t10=next((m["n_alive"] for m in mk if m["sim_time"] >= 10.0), mk[-1]["n_alive"]),
            sev_frac=mk[-1]["sev_frac_cum"], ke_over_u_end=mk[-1]["ke_over_u"],
            at_cap_max=d["bounds"]["at_cap_frac_max"], floor_ticks=d["bounds"]["ticks_at_dt_floor"],
            closure_max=max(m["closure_residual"] for m in mk), finite=d["finite"],
            e_int_series={round(m["sim_time"]): m["e_int"] for m in mk})

    # the frame: for each S run, B0 and D of the same (size, seed, md) evaluated on a random subset
    # at S's n_alive per mark in the window; R is count-matched by construction
    def key(d): return (d["size"], d["seed"], d["md"])
    by_arm = {}
    for d in runs: by_arm.setdefault(d["arm"], []).append(d)
    comparisons = []
    for S in by_arm.get("S", []):
        k = key(S)
        for arm in ("B0", "D"):
            for other in by_arm.get(arm, []):
                if key(other) != k: continue
                if arm == "D" and other.get("matched_tau") != S["tau"]: continue   # D is matched to ONE S run
                side = np.load(a.results_dir / other["pos_sidecar"])
                times = side["sim_times"]; vals = []
                for i, m in enumerate(S["marks"]):
                    if not (WINDOW[0] <= m["sim_time"] <= WINDOW[1] + 1e-9): continue
                    j = int(np.argmin(np.abs(times - m["sim_time"])))
                    sm, res = subset_metrics(side[f"t{j}"], other["config"]["box"], m["n_alive"], seed=1000 * S["seed"] + i)
                    vals.append(sm)
                if vals:
                    comparisons.append(dict(S_file=S["_file"], other_file=other["_file"], other_arm=arm, size=S["size"],
                                            seed=S["seed"], tau=S["tau"], md=S["md"],
                                            S_perc=S["_summary"]["perc"], other_perc_matched=float(np.mean([v["percolation"] for v in vals])),
                                            S_xi_u=S["_summary"]["xi_u"], other_xi_u_matched=float(np.mean([v["xi_u"] for v in vals])),
                                            S_occ=S["_summary"]["occ"], other_occ_matched=float(np.mean([v["occupancy"] for v in vals]))))
        for other in by_arm.get("R", []):
            if key(other) == k and other.get("matched_tau") == S["tau"]:          # R replays ONE S run
                comparisons.append(dict(S_file=S["_file"], other_file=other["_file"], other_arm="R", size=S["size"],
                                        seed=S["seed"], tau=S["tau"], md=S["md"],
                                        S_perc=S["_summary"]["perc"], other_perc_matched=other["_summary"]["perc"],
                                        S_xi_u=S["_summary"]["xi_u"], other_xi_u_matched=other["_summary"]["xi_u"],
                                        S_occ=S["_summary"]["occ"], other_occ_matched=other["_summary"]["occ"]))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    grid = dict(commit=commit, window=WINDOW, n_runs=len(runs),
                runs=[{k: v for k, v in d.items() if k not in ("marks", "events", "config")} | {"config": d["config"]} for d in runs],
                comparisons=comparisons)
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = a.results_dir / f"exp_03_sink_arms_grid_{a.size}_{stamp}.json"
    out.write_text(json.dumps(grid, indent=1, default=str), encoding="utf-8")
    print(f"  {len(runs)} runs, {len(comparisons)} frame comparisons -> results/{out.name}")
    for d in runs:
        s = d["_summary"]
        print(f"  {d['size']:>5} {d['arm']:>2} tau={str(d['tau']):>5} s={d['seed']} md={d['md']:.2f} damp={d['damping']:.5f} "
              f"perc={s['perc']:.3f} xi_u={s['xi_u']:.3f} occ={s['occ']:.3f} sev={s['sev_frac']:.3f} n@10={s['n_alive_t10']} "
              f"KE/|U|end={s['ke_over_u_end']:.1f} at_cap={s['at_cap_max']:.3f} floor={s['floor_ticks']} closure={s['closure_max']:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
