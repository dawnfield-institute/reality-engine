#!/usr/bin/env python3
"""Does structure peak at Xi — and does connectivity peak where contrast does?

`gravity_from_maxwell_pac` exp_10 swept SEC balance from 0.3 to 1.3 and found:

    NO discrete phase transition -- SEC is CONTINUOUS control.
    Xi ~ 1.057 is not a phase transition point, but the OPTIMAL OPERATING POINT
    for maximum structural complexity.

That is a dimensionless prediction the corpus can be held to, in a substrate rebuilt here,
and it is one of the few places a DFT constant goes IN as a parameter rather than coming out
as a fit. Unlike Omega_Lambda in POC-08 -- which this substrate has no units to express --
`sec_balance` is directly comparable to Xi because exp_10 defined it that way.

**The new question.** exp_10 measured "structural complexity" as density CV. This POC has
percolation, which it did not. POC-05's central finding is that contrast and connectivity are
DECOUPLED in the field engine: CV drove to 3.9 while percolation never left the noise floor.
So do they peak at the same SEC balance here, or is the Xi optimum an optimum for contrast
only?

Three outcomes, all worth having:

  both peak near Xi          exp_10 replicates and extends to connectivity
  CV peaks at Xi, perc does  the Xi optimum is about contrast; connectivity wants something
    not, or peaks elsewhere    else, and the decoupling is substrate-independent
  neither peaks              the rebuild does not reproduce exp_10, and the difference
                             between the two implementations is the finding

Five seeds from the start. A single-seed optimum with a good story died in POC-08 and there is
no reason to repeat that.

    python .../exp_02_is_xi_the_optimum.py [--steps 1000]
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from particles import ParticleConfig, ParticleEngine  # noqa: E402
from structure import percolation, web_metrics  # noqa: E402

XI_ANALYTIC = 0.5772156649015329 + np.log((1 + 5 ** 0.5) / 2)   # gamma + ln(phi) = 1.05843
XI_DISCRETE = 1.0 + np.pi / 55                                   # 1 + pi/55  = 1.05712

# exp_10's own range, extended a little either side so a peak inside it is not an artifact
# of the endpoints. g is held at exp_09's 0.8, as exp_10 held it.
BALANCES = [0.3, 0.5, 0.7, 0.9, 1.0, 1.057, 1.15, 1.3, 1.6]
SEEDS = [42, 7, 99, 13, 71]


def run(sec_balance, steps, seed, n=4000):
    eng = ParticleEngine(ParticleConfig(n=n, box=120.0, r0=5.0, g=0.8,
                                        sec_balance=sec_balance, seed=seed))
    best_cv, best_perc = 0.0, 0.0
    for t in range(1, steps + 1):
        eng.tick()
        if t % 50 == 0:
            F = eng.density_field(128).cpu().numpy().astype(float)
            w = web_metrics(F)
            best_cv = max(best_cv, w["cv"])
            best_perc = max(best_perc, w["percolation"])
    return best_cv, best_perc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    print(f"  {args.n} particles, {args.steps} steps, g = 0.8, {len(SEEDS)} seeds")
    print(f"  exp_10: Xi ~ 1.057 is the optimal operating point for structural complexity")
    print(f"  Xi_analytic = {XI_ANALYTIC:.5f}   Xi_discrete = {XI_DISCRETE:.5f}\n")
    hdr = f"  {'sec_balance':>12} {'peak CV':>18} {'peak percolation':>22}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for b in BALANCES:
        res = [run(b, args.steps, s, args.n) for s in SEEDS]
        cv = np.array([r[0] for r in res])
        pc = np.array([r[1] for r in res])
        mark = "  <- Xi" if abs(b - 1.057) < 1e-6 else ""
        print(f"  {b:>12.3f} {cv.mean():>10.3f}+-{cv.std():<6.3f} "
              f"{pc.mean():>14.4f}+-{pc.std():<6.4f}{mark}")
        rows.append({"sec_balance": b, "cv_mean": float(cv.mean()),
                     "cv_std": float(cv.std()), "perc_mean": float(pc.mean()),
                     "perc_std": float(pc.std()), "seeds": SEEDS})

    b = np.array([r["sec_balance"] for r in rows])
    cvm = np.array([r["cv_mean"] for r in rows])
    cvs = np.array([r["cv_std"] for r in rows])
    pcm = np.array([r["perc_mean"] for r in rows])
    pcs = np.array([r["perc_std"] for r in rows])

    print(f"\n  peak CV          at sec_balance = {b[int(cvm.argmax())]:.3f}")
    print(f"  peak percolation at sec_balance = {b[int(pcm.argmax())]:.3f}")

    # Is either peak resolved against seed scatter? Compare the best arm to the Xi arm.
    xi_i = int(np.argmin(np.abs(b - 1.057)))
    for name, m, sd in (("CV", cvm, cvs), ("percolation", pcm, pcs)):
        k = int(m.argmax())
        if k == xi_i:
            print(f"  {name}: peak IS the Xi arm")
            continue
        se = np.sqrt(sd[k] ** 2 + sd[xi_i] ** 2) / np.sqrt(len(SEEDS))
        sig = (m[k] - m[xi_i]) / se if se > 0 else float("nan")
        print(f"  {name}: best arm ({b[k]:.3f}) beats the Xi arm by {sig:.2f} sigma")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, m, sd, lab in ((ax[0], cvm, cvs, "peak density CV"),
                          (ax[1], pcm, pcs, "peak percolation")):
        a.errorbar(b, m, yerr=sd / np.sqrt(len(SEEDS)), marker="o", lw=2, capsize=3)
        a.axvline(XI_ANALYTIC, color="crimson", ls="--", lw=1.2, label="Xi analytic")
        a.axvline(XI_DISCRETE, color="orange", ls=":", lw=1.2, label="Xi discrete")
        a.set_xlabel("sec_balance"); a.set_ylabel(lab); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Is Xi the optimal operating point — for contrast, for connectivity, or "
                 "neither?", fontsize=11)
    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "xi_optimum.png", dpi=115, bbox_inches="tight")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out / f"exp_02_xi_{stamp}.json").write_text(json.dumps(
        {"steps": args.steps, "n": args.n, "g": 0.8, "seeds": SEEDS,
         "xi_analytic": XI_ANALYTIC, "xi_discrete": XI_DISCRETE, "arms": rows},
        indent=2), encoding="utf-8")
    print(f"\n  wrote results/xi_optimum.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
