#!/usr/bin/env python3
"""POC-02 (v4) exp_01 — how durable is balance?

Displaces the E/I balance WITHOUT touching the global ledger (E += d, I -= d, so
Q = sum(E+I+M) is unchanged by construction), then measures whether each balance
observable returns. Then repeats across a one-at-a-time parameter sweep to measure how
wide the basin is.

See README.md for the registration. This implements it and decides nothing it did not fix.

    python proof_of_concepts/v4/poc_02_balance_durability/scripts/exp_01_balance_durability.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))

from harness import durability, one_at_a_time  # noqa: E402

# Observables: all already emitted by the engine. metrics are written by the operators
# during tick(), so they are read from state.metrics after the tick.
def _info_fraction(e) -> float:
    """|I|/(|E|+|I|) computed from the fields.

    sec_tracking emits this metric only on some ticks, so it is computed here rather than
    read, to guarantee a value every tick. Same definition as the operator's.
    """
    st = e.state
    i = st.I.abs().sum().item()
    return i / (st.E.abs().sum().item() + i + 1e-30)


OBSERVABLES = {
    "info_fraction":     _info_fraction,
    "balance_magnitude": lambda e: e.state.metrics.get("balance_magnitude", float("nan")),
    "alpha_local_mean":  lambda e: e.state.metrics.get("alpha_local_mean", float("nan")),
    "lambda_local_mean": lambda e: e.state.metrics.get("lambda_local_mean", float("nan")),
    "gamma_local_mean":  lambda e: e.state.metrics.get("gamma_local_mean", float("nan")),
    "xi_s_mean":         lambda e: e.state.metrics.get("xi_s_mean", float("nan")),
}

BASE = {
    "nu": 64, "nv": 32, "dt": 1e-3, "enforce_pac": False, "noise_scale": 0.0,
    "quantum_pressure_coeff": 0.020,
    "deactualization_rate": 0.025,
    "mass_gen_coeff": 0.63,
    "confluence_weight": 0.3,
}

FACTORS = {
    "quantum_pressure_coeff": [0.5, 2.0],
    "deactualization_rate":   [0.5, 2.0],
    "mass_gen_coeff":         [0.5, 2.0],
    "confluence_weight":      [0.5, 2.0],
}


def make_balance_impulse(frac: float):
    """E += d, I -= d. Ledger-preserving by construction — that is the whole point."""
    def perturb(eng) -> None:
        st = eng.state
        scale = frac * st.E.abs().mean().item()
        n = st.E.numel()
        eng.state = st.replace(E=st.E + scale, I=st.I - scale)
        _ = n
    return perturb


def ledger(eng) -> float:
    st = eng.state
    return (st.E + st.I + st.M).sum().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=int, default=250)
    ap.add_argument("--observe", type=int, default=600)
    ap.add_argument("--impulse", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="base config only, no sweep")
    args = ap.parse_args()

    configs = [dict(BASE) | {"_varied": "none", "_factor": 1.0}] if args.quick \
        else list(one_at_a_time(BASE, FACTORS))

    rows, t0 = [], time.time()
    print(f"{'varied':>24} {'x':>5} {'seed':>5} " +
          " ".join(f"{k[:13]:>14}" for k in OBSERVABLES))
    for cfg in configs:
        varied, factor = cfg.pop("_varied"), cfg.pop("_factor")
        for s in range(args.seeds):
            res = durability(
                OBSERVABLES, make_balance_impulse(args.impulse),
                seed=3000 + s, settle=args.settle, observe=args.observe, **cfg,
            )
            row = {"varied": varied, "factor": factor, "seed": 3000 + s,
                   "pre_impulse_gap": next(iter(res.values())).pre_impulse_gap,
                   "R": {k: v.R for k, v in res.items()},
                   "verdict": {k: v.verdict for k, v in res.items()},
                   "D0": {k: v.D0 for k, v in res.items()}}
            rows.append(row)
            print(f"{varied:>24} {factor:>5.1f} {3000+s:>5} " +
                  " ".join(f"{res[k].R:>14.4f}" for k in OBSERVABLES))
        cfg["_varied"], cfg["_factor"] = varied, factor

    # Per-observable summary across the whole sweep.
    summary = {}
    for k in OBSERVABLES:
        Rs = [r["R"][k] for r in rows if r["R"][k] == r["R"][k]]
        verdicts = Counter(r["verdict"][k] for r in rows)
        durable = verdicts.get("durable", 0)
        summary[k] = {
            "median_R": statistics.median(Rs) if Rs else float("nan"),
            "min_R": min(Rs) if Rs else None,
            "max_R": max(Rs) if Rs else None,
            "verdict_counts": dict(verdicts),
            "durable_fraction": durable / len(rows) if rows else 0.0,
        }

    finite = {k: v for k, v in summary.items() if v["min_R"] is not None}
    all_neutral_or_worse = bool(finite) and all(
        v["median_R"] >= 0.9 for v in finite.values())
    kill = ("KILL SENTENCE FIRED — every balance observable has median R >= 0.9; "
            "balance is not durable in this engine either."
            if all_neutral_or_worse else
            "NOT killed — at least one balance observable restores.")

    max_gap = max(r["pre_impulse_gap"] for r in rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).resolve().parents[1] / "results" / f"exp_01_balance_durability_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "poc_02_balance_durability / exp_01",
        "registered": "README.md (2026-08-11), pre-registered",
        "measured": "recovery ratio R per balance observable, twin-differenced; "
                    "impulse is ledger-preserving (E+=d, I-=d)",
        "impulse_fraction": args.impulse,
        "settle": args.settle, "observe": args.observe,
        "rows": rows, "summary": summary,
        "max_pre_impulse_gap": max_gap,
        "kill_sentence": kill,
        "wall_seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")

    print()
    print(f"  twin sanity — max pre-impulse gap: {max_gap:.3e} (must be ~0)")
    for k, s in summary.items():
        if s["min_R"] is None:
            print(f"  {k:>20}: no finite R (observable unavailable)")
            continue
        print(f"  {k:>20}: median R={s['median_R']:.4f}  "
              f"range [{s['min_R']:.4f}, {s['max_R']:.4f}]  "
              f"durable {s['durable_fraction']*100:.0f}% of settings  {s['verdict_counts']}")
    print(f"  {kill}")
    print(f"  wrote {out.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
