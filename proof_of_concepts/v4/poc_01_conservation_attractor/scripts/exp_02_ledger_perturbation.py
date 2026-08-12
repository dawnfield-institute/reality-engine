#!/usr/bin/env python3
"""POC-01 (v4) exp_02 — is the global PAC ledger an attractor?

Perturbation-recovery test. An attractor is defined by its basin: displace the system and
it returns. This measures that directly and takes no position on continuum limits,
convergence, or which discretization is "true" — the questions that made exp_01 invalid.

Method: run to a settled state, fork it, apply a single impulse to one copy's global
ledger, evolve both on identical seeds, and track the displacement between them.

    D(t) = Q_perturbed(t) - Q_reference(t)
    R    = |median D over final 20%| / |D just after impulse|

Differencing against a twin cancels the engine's own baseline drift: whatever the
unperturbed dynamics do, both copies do it. That is why this design does not need to know
whether the drift is physical or numerical.

Local and global are recorded SEPARATELY and not graded against each other. SEC is local;
only the global ledger balances. A local leak is expected behaviour, not an error.

    python proof_of_concepts/v4/poc_01_conservation_attractor/scripts/exp_02_ledger_perturbation.py
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

import torch  # noqa: E402

from src.v3.dashboard.server import build_default_pipeline  # noqa: E402
from src.v3.engine.config import SimulationConfig           # noqa: E402
from src.v3.engine.engine import Engine                     # noqa: E402


def ledger(state) -> float:
    return (state.E + state.I + state.M).sum().item()


def local_balance(state) -> float:
    """Mean per-cell |E-I|/(|E|+|I|) — a local imbalance measure.

    Recorded for context only. Graded against nothing: local leaks are not defects.
    """
    num = (state.E - state.I).abs()
    den = state.E.abs() + state.I.abs() + 1e-12
    return (num / den).mean().item()


def make_engine(nu, nv, dt, seed, noise):
    torch.manual_seed(seed)
    cfg = SimulationConfig(nu=nu, nv=nv, dt=dt, enforce_pac=False, noise_scale=noise)
    eng = Engine(config=cfg, pipeline=build_default_pipeline())
    eng.initialize(mode="big_bang")
    return eng


def run_trial(nu, nv, dt, seed, noise, settle, observe, impulse_frac) -> dict:
    """One perturbation trial. Returns displacement trajectory and recovery ratio."""
    ref = make_engine(nu, nv, dt, seed, noise)
    per = make_engine(nu, nv, dt, seed, noise)

    for _ in range(settle):
        ref.tick(); per.tick()

    # The two copies must be identical before the impulse, or the difference is meaningless.
    pre_gap = abs(ledger(per.state) - ledger(ref.state))
    q_settled = ledger(ref.state)

    # Impulse: add a uniform increment to E across the perturbed copy only.
    delta = impulse_frac * abs(q_settled)
    st = per.state
    per.state = st.replace(E=st.E + delta / st.E.numel())
    d0 = ledger(per.state) - ledger(ref.state)

    disp, loc_ref, loc_per = [], [], []
    for _ in range(observe):
        ref.tick(); per.tick()
        disp.append(ledger(per.state) - ledger(ref.state))
        loc_ref.append(local_balance(ref.state))
        loc_per.append(local_balance(per.state))

    tail = max(1, int(len(disp) * 0.2))
    d_end = statistics.median(disp[-tail:])
    R = abs(d_end) / abs(d0) if d0 else float("nan")

    return {
        "nu": nu, "nv": nv, "dt": dt, "seed": seed, "noise_scale": noise,
        "settle_ticks": settle, "observe_ticks": observe,
        "impulse_fraction": impulse_frac,
        "pre_impulse_gap": pre_gap,
        "q_settled": q_settled,
        "D0": d0, "D_end": d_end, "recovery_ratio_R": R,
        "displacement_trajectory": disp,
        "local_balance_reference_final": loc_ref[-1] if loc_ref else None,
        "local_balance_perturbed_final": loc_per[-1] if loc_per else None,
    }


def classify(R_values: list[float]) -> str:
    if not R_values or any(r != r for r in R_values):
        return "INVALID — non-finite recovery ratio"
    R = statistics.median(R_values)
    if R < 0.5:
        return f"ATTRACTOR — displacement decays (median R={R:.3f})"
    if 0.9 <= R <= 1.1:
        return f"NEUTRAL — conserved but non-restoring (median R={R:.3f})"
    if R > 1.5:
        return f"UNSTABLE — displacement grows (median R={R:.3f})"
    return f"AMBIGUOUS — median R={R:.3f} in no registered band"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settle", type=int, default=300)
    ap.add_argument("--observe", type=int, default=700)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    # Config varied only to check the phenomenon is not unique to one setting.
    # This is robustness, NOT a convergence study — no discretization is privileged.
    configs = [(64, 32, 1e-3), (32, 16, 1e-3), (64, 32, 5e-4)]
    impulses = [0.05, -0.05, 0.20]

    trials, t0 = [], time.time()
    print(f"{'grid':>9} {'dt':>8} {'impulse':>8} {'seed':>5} {'D0':>11} {'D_end':>11} {'R':>8}")
    for nu, nv, dt in configs:
        for imp in impulses:
            for s in range(args.seeds):
                t = run_trial(nu, nv, dt, 2000 + s, args.noise,
                              args.settle, args.observe, imp)
                trials.append(t)
                print(f"{nu:>4}x{nv:<4} {dt:>8.1e} {imp:>8.2f} {2000+s:>5} "
                      f"{t['D0']:>11.4e} {t['D_end']:>11.4e} {t['recovery_ratio_R']:>8.3f}")

    Rs = [t["recovery_ratio_R"] for t in trials]
    verdict = classify(Rs)

    by_cfg = {}
    for t in trials:
        by_cfg.setdefault(f"{t['nu']}x{t['nv']}@dt{t['dt']:.0e}", []).append(
            t["recovery_ratio_R"])
    by_imp = {}
    for t in trials:
        by_imp.setdefault(f"{t['impulse_fraction']:+.2f}", []).append(t["recovery_ratio_R"])

    max_pre_gap = max(t["pre_impulse_gap"] for t in trials)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).resolve().parents[1] / "results" / f"exp_02_ledger_perturbation_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "poc_01_conservation_attractor / exp_02_ledger_perturbation",
        "registered": "README.md registration v2 (2026-08-11), pre-registered",
        "measured": "recovery ratio R = |median D over final 20%| / |D0| against a twin",
        "trials": trials,
        "median_R": statistics.median(Rs),
        "R_by_config": {k: statistics.median(v) for k, v in by_cfg.items()},
        "R_by_impulse": {k: statistics.median(v) for k, v in by_imp.items()},
        "max_pre_impulse_gap": max_pre_gap,
        "verdict": verdict,
        "wall_seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")

    print()
    print(f"  twin sanity — max pre-impulse gap: {max_pre_gap:.3e} (must be ~0)")
    print("  R by config :", {k: round(statistics.median(v), 4) for k, v in by_cfg.items()})
    print("  R by impulse:", {k: round(statistics.median(v), 4) for k, v in by_imp.items()})
    print(f"  VERDICT: {verdict}")
    print(f"  wrote {out.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
