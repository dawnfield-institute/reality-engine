#!/usr/bin/env python3
"""POC-01 (v4) exp_01 — does the PAC residual survive refinement?

Runs the default pipeline with `enforce_pac=False` across a grid x timestep sweep and
measures the RELATIVE DRIFT RATE. See README.md for the pre-registration; this script
implements it and does not decide anything the registration did not already fix.

The measured quantity, per the registration (Amendment 1):

    rate     = |residual| / dt          absolute drift per unit simulated time
    rel_rate = rate / |E + I + M|       fractional drift per unit simulated time
    P        = median rel_rate over the final 20% of ticks

Runs are compared at EQUAL SIMULATED TIME, not equal tick count: the tick budget is
T / dt. Comparing at equal ticks would compare different amounts of evolution and the
per-tick residual would fall with dt for trivial reasons.

    python proof_of_concepts/v4/poc_01_conservation_attractor/scripts/exp_01_refinement_sweep.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from src.v3.dashboard.server import build_default_pipeline  # noqa: E402
from src.v3.engine.config import SimulationConfig           # noqa: E402
from src.v3.engine.engine import Engine                     # noqa: E402

GRIDS = [(32, 16), (64, 32), (128, 64)]
DTS = [1e-3, 5e-4, 2.5e-4]
SIM_TIME = 0.4          # simulated time each run covers; ticks = SIM_TIME / dt
TAIL_FRACTION = 0.2     # plateau measured over the final 20% of ticks


def run_one(nu: int, nv: int, dt: float, enforce: bool, sim_time: float,
            noise: float = 0.0, seed: int | None = None) -> dict:
    """One run. Returns the trajectory and the plateau statistic.

    noise=0.0 by default: the primary sweep is DETERMINISTIC. Thermal noise scales as
    sqrt(dt), so a noise-dominated residual gives rel_rate ~ dt^(-1/2) and would be
    misread as a failure to converge (README Amendment 2).
    """
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    ticks = int(round(sim_time / dt))
    cfg = SimulationConfig(nu=nu, nv=nv, dt=dt, enforce_pac=enforce, noise_scale=noise)
    eng = Engine(config=cfg, pipeline=build_default_pipeline())
    eng.initialize(mode="big_bang")

    rel_rates, totals = [], []
    t0 = time.time()
    for _ in range(ticks):
        eng.tick()
        st = eng.state
        residual = st.metrics.get("pac_correction")
        if residual is None:
            continue
        total = (st.E + st.I + st.M).sum().item()
        if abs(total) < 1e-12:
            continue
        rel_rates.append(abs(residual) / dt / abs(total))
        totals.append(total)

    tail = max(1, int(len(rel_rates) * TAIL_FRACTION))
    plateau = statistics.median(rel_rates[-tail:]) if rel_rates else float("nan")
    return {
        "nu": nu, "nv": nv, "dt": dt, "enforce_pac": enforce,
        "noise_scale": noise, "seed": seed,
        "ticks": ticks, "sim_time": sim_time,
        "cells": nu * nv,
        "plateau_rel_rate": plateau,
        "rel_rate_first": rel_rates[0] if rel_rates else None,
        "rel_rate_last": rel_rates[-1] if rel_rates else None,
        "total_initial": totals[0] if totals else None,
        "total_final": totals[-1] if totals else None,
        "wall_seconds": round(time.time() - t0, 2),
    }


def classify(runs: list[dict]) -> dict:
    """Apply the registered criteria. Reports 'ambiguous' rather than rounding."""
    by_grid: dict[tuple, list[dict]] = {}
    for r in runs:
        by_grid.setdefault((r["nu"], r["nv"]), []).append(r)

    dt_trends = {}
    for grid, rs in by_grid.items():
        rs = sorted(rs, key=lambda r: -r["dt"])          # coarse -> fine
        ps = [r["plateau_rel_rate"] for r in rs]
        ratios = [ps[i + 1] / ps[i] for i in range(len(ps) - 1) if ps[i]]
        dt_trends[f"{grid[0]}x{grid[1]}"] = {
            "plateaus_coarse_to_fine": ps,
            "successive_ratios": ratios,
            "max_abs_change_pct": max((abs(1 - x) * 100 for x in ratios), default=None),
            "monotonic_decreasing": all(x < 1.0 for x in ratios) if ratios else None,
        }

    changes = [v["max_abs_change_pct"] for v in dt_trends.values()
               if v["max_abs_change_pct"] is not None]
    all_mono = all(v["monotonic_decreasing"] for v in dt_trends.values()
                   if v["monotonic_decreasing"] is not None)
    worst = max(changes) if changes else None

    all_rising = all(
        all(x > 1.0 for x in v["successive_ratios"])
        for v in dt_trends.values() if v["successive_ratios"]
    )
    if worst is not None and worst < 10.0:
        verdict = "PHYSICAL — plateau converged (<10% across refinements)"
    elif all_mono and worst is not None and worst > 10.0:
        verdict = "NUMERICAL — plateau falls monotonically with refinement"
    elif all_rising:
        verdict = "NOISE-DOMINATED — plateau rises under refinement (~dt^-1/2)"
    else:
        verdict = "AMBIGUOUS — neither registered criterion met"

    return {"dt_trends": dt_trends, "worst_change_pct": worst,
            "monotonic_across_all_grids": all_mono, "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-time", type=float, default=SIM_TIME)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="noise_scale; 0.0 = deterministic primary sweep")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="smallest grid only — for wiring checks, not for results")
    args = ap.parse_args()

    grids = GRIDS[:1] if args.quick else GRIDS
    runs = []
    print(f"sweep: {len(grids)} grids x {len(DTS)} timesteps, sim_time={args.sim_time}")
    print(f"{'grid':>9} {'dt':>9} {'ticks':>7} {'plateau rel_rate':>18} {'wall s':>7}")
    for nu, nv in grids:
        for dt in DTS:
            reps = [run_one(nu, nv, dt, enforce=False, sim_time=args.sim_time,
                            noise=args.noise, seed=1000 + k)
                    for k in range(args.repeats)]
            r = dict(reps[0])
            ps = [x["plateau_rel_rate"] for x in reps]
            r["plateau_rel_rate"] = statistics.median(ps)
            r["repeats"] = args.repeats
            r["plateau_spread"] = (max(ps) - min(ps)) if len(ps) > 1 else 0.0
            runs.append(r)
            print(f"{nu:>4}x{nv:<4} {dt:>9.2e} {r['ticks']:>7} "
                  f"{r['plateau_rel_rate']:>18.6e} {r['wall_seconds']:>7}")

    # Control: the same configuration WITH enforcement, to confirm the flag is what
    # changes the answer rather than anything else in the run.
    control = run_one(grids[0][0], grids[0][1], DTS[0], enforce=True,
                      sim_time=args.sim_time, noise=args.noise, seed=1000)

    analysis = classify(runs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).resolve().parents[1] / "results" / f"exp_01_refinement_sweep_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "poc_01_conservation_attractor / exp_01_refinement_sweep",
        "registered": "README.md (pre-registered 2026-08-11, Amendment 1 before any run)",
        "measured": "median relative drift rate over final 20% of ticks, equal simulated time",
        "noise_scale": args.noise, "repeats": args.repeats,
        "runs": runs, "control_with_enforcement": control, "analysis": analysis,
    }, indent=2), encoding="utf-8")

    print()
    for grid, v in analysis["dt_trends"].items():
        print(f"  {grid}: plateaus (coarse->fine) "
              f"{[f'{p:.3e}' for p in v['plateaus_coarse_to_fine']]}")
        print(f"  {'':<{len(grid)}}  successive ratios "
              f"{[f'{x:.3f}' for x in v['successive_ratios']]}")
    print()
    print(f"  control (enforce_pac=True): plateau {control['plateau_rel_rate']:.6e}")
    print(f"  worst change across refinements: {analysis['worst_change_pct']:.1f}%")
    print(f"  VERDICT: {analysis['verdict']}")
    print(f"  wrote {out.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
