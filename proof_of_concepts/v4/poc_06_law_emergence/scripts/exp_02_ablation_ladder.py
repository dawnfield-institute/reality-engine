#!/usr/bin/env python3
"""Where along the operator stack does the engine start being generative?

exp_01 asked the full 16-operator pipeline what laws it has and got one, enforced.
The obvious follow-up is not to add a seventeenth operator — every additive proposal in
this POC round failed the same way — but to REMOVE them and watch what happens to the count.

Three outcomes, all worth having:

  flat at zero      the operators are not doing the work, and the axioms as implemented do
                    not generate laws
  rises on removal  the operators are SUPPRESSING emergence by pinning the dynamics — would
                    invert the engine's whole development history
  falls on removal  they are load-bearing, and the question becomes which ones

The comparison is RELATIVE, one instrument across every rung, which is what makes it robust
to the detector being blunt. exp_01's absolute verdict is only as good as the calibration;
the SHAPE of this curve survives a weak detector.

**The rung that matters most is `no_normalization`.** `NormalizationOperator` is what holds
`E + I + M` at 2.5e-15 with an explicit correction every tick — the single ENFORCED law. Take
it away and PAC either survives on its own, in which case it is emergent after all, or it does
not, in which case the engine's one conservation law was only ever bookkeeping. `enforce_pac`
is also swept separately, since that disables the correction while keeping the clamps.

SECTrackingOperator is kept in every rung. It is read-only and supplies the entropy metric;
removing it would change what can be measured rather than what the physics does.

    python .../exp_02_ablation_ladder.py [--ticks 3000] [--grid 64 64]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from law_detector import conservation_scan, entropy_trend  # noqa: E402
from src.v3.engine.config import SimulationConfig  # noqa: E402
from src.v3.engine.engine import Engine  # noqa: E402
from src.v3.engine.pipelines import CANONICAL  # noqa: E402
from src.v3.operators.actualization import ActualizationOperator  # noqa: E402
from src.v3.operators.adaptive import AdaptiveOperator  # noqa: E402
from src.v3.operators.charge_dynamics import ChargeDynamicsOperator  # noqa: E402
from src.v3.operators.confluence import ConfluenceOperator  # noqa: E402
from src.v3.operators.fusion import FusionOperator  # noqa: E402
from src.v3.operators.gravity import GravitationalCollapseOperator  # noqa: E402
from src.v3.operators.memory import MemoryOperator  # noqa: E402
from src.v3.operators.normalization import NormalizationOperator  # noqa: E402
from src.v3.operators.phi_cascade import PhiCascadeOperator  # noqa: E402
from src.v3.operators.protocol import Pipeline  # noqa: E402
from src.v3.operators.qbe import QBEOperator  # noqa: E402
from src.v3.operators.rbf import RBFOperator  # noqa: E402
from src.v3.operators.sec_tracking import SECTrackingOperator  # noqa: E402
from src.v3.operators.spin_statistics import SpinStatisticsOperator  # noqa: E402
from src.v3.operators.temperature import TemperatureOperator  # noqa: E402
from src.v3.operators.thermal_noise import ThermalNoiseOperator  # noqa: E402
from src.v3.operators.time_emergence import TimeEmergenceOperator  # noqa: E402

CORE = [RBFOperator, QBEOperator, ActualizationOperator, NormalizationOperator,
        SECTrackingOperator]

RUNGS = [
    ("16 canonical", list(CANONICAL), {}),
    ("12 no deep physics", [o for o in CANONICAL if o not in
                            (PhiCascadeOperator, SpinStatisticsOperator,
                             ChargeDynamicsOperator, FusionOperator)], {}),
    ("8 no forces/thermal", [o for o in CANONICAL if o not in
                             (PhiCascadeOperator, SpinStatisticsOperator,
                              ChargeDynamicsOperator, FusionOperator,
                              GravitationalCollapseOperator, TemperatureOperator,
                              ThermalNoiseOperator, AdaptiveOperator)], {}),
    ("6 core + memory + confluence",
     [RBFOperator, QBEOperator, ActualizationOperator, MemoryOperator,
      ConfluenceOperator, NormalizationOperator, SECTrackingOperator], {}),
    ("5 core + memory", CORE[:3] + [MemoryOperator] + CORE[3:], {}),
    ("4 core only", CORE, {}),
    ("4 core, enforce_pac off", CORE, {"enforce_pac": False}),
    ("3 NO normalization", [RBFOperator, QBEOperator, ActualizationOperator,
                            SECTrackingOperator], {}),
]


def quantities(s):
    E, I, M = s.E, s.I, s.M
    P = s.P if s.P is not None else torch.zeros_like(M)
    f = lambda t: float(t.sum().item())  # noqa: E731
    return {"E+I+M": f(E + I + M), "E+I+M+P": f(E + I + M + P),
            "E": f(E), "I": f(I), "M": f(M),
            "E^2+I^2": f(E.pow(2) + I.pow(2)),
            "E*I": f(E * I), "|E-I|": f((E - I).abs())}


def run(ops, cfg_kw, ticks, grid, seed):
    torch.manual_seed(seed)
    eng = Engine(config=SimulationConfig(nu=grid[0], nv=grid[1], noise_scale=0.0, **cfg_kw),
                 pipeline=Pipeline([o() for o in ops]))
    eng.initialize("big_bang", temperature=3.0)
    hist, ent, act, diverged = {}, [], [], None
    # What this pipeline actually enforces, decided by what is in it rather than by name.
    enforced = set()
    if NormalizationOperator in ops and cfg_kw.get("enforce_pac", True):
        enforced.add("E+I+M")
    if QBEOperator in ops:
        enforced.add("dI = -dE")
    for t in range(1, ticks + 1):
        eng.tick()
        s = eng.state
        # E and I too — a run where E blows up while M stays finite is still divergent, and
        # the first version only checked M, which let a non-finite ledger through as `nan`.
        if not (torch.isfinite(s.M).all() and torch.isfinite(s.E).all()
                and torch.isfinite(s.I).all()):
            diverged = t
            break
        if max(s.M.abs().max().item(), s.E.abs().max().item()) > 1e12:
            diverged = t
            break
        for k, v in quantities(s).items():
            hist.setdefault(k, []).append(v)
        e = s.metrics.get("field_entropy")
        if isinstance(e, (int, float)):
            ent.append(float(e))
        # Liveness: how much is still actualizing. The core-only rung decays from 66 events
        # per tick to 7 by t=3000, at which point everything looks conserved because nothing
        # is happening.
        act.append(float(s.metrics.get("actualization_count", 0)) / float(s.M.numel()))
    return hist, ent, act, diverged, enforced


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--grid", type=int, nargs=2, default=[64, 64])
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 7])
    args = ap.parse_args()

    print(f"  {args.grid[0]}x{args.grid[1]}, {args.ticks} ticks, seeds {args.seeds}")
    print(f"  exp_01 reference (16 operators, 128x128, 12000 ticks): "
          f"0 EMERGENT, 1 ENFORCED\n")
    hdr = (f"  {'rung':<30} {'ops':>4} {'EMERG':>6} {'QUIES':>6} {'ENF':>4} "
           f"{'PAC CV':>11} {'act ratio':>9} {'act/cell':>9}  emergent laws")
    print(hdr)
    print("  " + "-" * (len(hdr) + 18))

    rows = []
    for label, ops, kw in RUNGS:
        per_seed = []
        for sd in args.seeds:
            hist, ent, act, div, enf = run(ops, kw, args.ticks, tuple(args.grid), sd)
            if div:
                per_seed.append(("DIVERGED", div, None, None, None))
                continue
            laws = conservation_scan(hist, tol=1e-3, enforced=enf, activity=act)
            em = [l.name for l in laws if l.verdict == "EMERGENT"]
            en = [l.name for l in laws if l.verdict == "ENFORCED"]
            qu = [l.name for l in laws if l.verdict == "QUIESCENT"]
            pac = next((l.statistic for l in laws if l.name == "E+I+M"), float("nan"))
            slope = entropy_trend(ent)[0] if len(ent) > 10 else float("nan")
            ratio = (np.mean(act[len(act)//2:]) / (np.mean(act[:len(act)//2]) + 1e-30)
                     if len(act) >= 8 else float("nan"))
            per_seed.append(("OK", em, en, pac, slope, qu, ratio, act))

        if all(p[0] == "DIVERGED" for p in per_seed):
            print(f"  {label:<30} {len(ops):>4}  DIVERGED at t={per_seed[0][1]}")
            rows.append({"rung": label, "n_ops": len(ops), "diverged_at": per_seed[0][1]})
            continue
        ok = [p for p in per_seed if p[0] == "OK"]
        em_sets = [set(p[1]) for p in ok]
        common = set.intersection(*em_sets) if em_sets else set()
        en_n = len(ok[0][2])
        pac = float(np.mean([p[3] for p in ok]))
        slope = float(np.mean([p[4] for p in ok if p[4] == p[4]])) if any(
            p[4] == p[4] for p in ok) else float("nan")
        nq = len(set.intersection(*[set(p[5]) for p in ok])) if ok else 0
        ratio = float(np.mean([p[6] for p in ok if p[6] == p[6]])) if ok else float("nan")
        late = float(np.mean([np.mean(p[7][len(p[7])//2:]) for p in ok]))
        print(f"  {label:<30} {len(ops):>4} {len(common):>6} {nq:>6} {en_n:>4} "
              f"{pac:>11.2e} {ratio:>9.3f} {late:>9.5f}  "
              f"{', '.join(sorted(common)) if common else '-'}")
        rows.append({"rung": label, "n_ops": len(ops),
                     "emergent": sorted(common), "n_quiescent": nq, "n_enforced": en_n,
                     "pac_cv": pac, "entropy_slope": slope,
                     "activity_ratio": ratio,
                     "late_actualizing_fraction": late, "seeds": args.seeds})

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out / f"exp_02_ablation_{stamp}.json").write_text(json.dumps(
        {"ticks": args.ticks, "grid": args.grid, "seeds": args.seeds, "rungs": rows},
        indent=2), encoding="utf-8")
    print(f"\n  wrote results/exp_02_ablation_{stamp}.json")
    print("  NOTE: 'emergent' is the INTERSECTION across seeds — a law found on one seed "
          "and not another is not counted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
