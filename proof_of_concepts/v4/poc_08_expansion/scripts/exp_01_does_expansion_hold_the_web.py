#!/usr/bin/env python3
"""Does expansion hold the web open — and does DFT's own number do it?

POC-07 found the web is TRANSIENT. Percolation peaks near 0.188 around t~600 and decays to
0.050 by t=4000 as clumps virialize and the connecting bridges drain into them. Without
something holding the structure open, gravity wins and you get isolated bound halos.

Real cosmology has that something: expansion. And DFT makes a prediction about it —
**Omega_Lambda = 1/phi = 0.6180**, from the PAC/SEC equilibrium (exp_25, which also puts the
universe crossing that equilibrium at z ~ 0.10). LCDM measures 0.685.

**What is being asked.** Not "is 0.618 better than 0.685" — they are 10% apart and this toy
almost certainly cannot resolve that. The question is whether the framework's number, dropped
in as a parameter, *does the job*: does the web persist where the static run drained?

That is the smallest case of a DFT prediction going in as a simulation parameter and the
simulation then doing something it was not tuned to do.

Arms span the range so the answer is a curve rather than a point:
    static           no expansion (POC-07's result, the control)
    Omega_L = 0.0    matter only, decelerating
    Omega_L = 0.618  DFT, 1/phi
    Omega_L = 0.685  LCDM
    Omega_L = 0.90   over-expanding — structure should be frozen out

    python proof_of_concepts/v4/poc_08_expansion/scripts/exp_01_does_expansion_hold_the_web.py
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
import torch  # noqa: E402

from particles import PHI, Cosmology, ParticleConfig, ParticleEngine  # noqa: E402
from structure import percolation, web_metrics  # noqa: E402


# First sweep varied Omega_Lambda at a fixed H0 = 0.02 and found expansion strictly HARMFUL,
# monotonically in a_final: static retained 42.5% of its peak, the DFT arm 9.5%, the
# over-expanding arm 6.3%. But `a` reached 25-45, so a fixed PHYSICAL interaction range of 5
# ended up inside a box of physical size ~3000 and particles simply could not reach each
# other. That measured dilution, not the competition between expansion and collapse.
#
# The competition is the real variable, and it is the coincidence problem in miniature: a web
# needs the expansion rate comparable to the collapse rate. So Omega_Lambda is now held at
# DFT's 1/phi and H0 is swept across three decades, asking whether ANY expansion rate beats
# the static control.
ARMS = [("static (control)", None)] + [
    (f"H0 {h:<7g} (Omega_L=1/phi)", (1.0 / PHI, h))
    for h in (0.0002, 0.0005, 0.001, 0.002, 0.005)
]


def run(spec, steps, n, h0, seed=42):
    if spec is None:
        cos = None
    elif isinstance(spec, tuple):
        ol, h = spec
        cos = Cosmology(h0=h, omega_lambda=ol)
    else:
        cos = Cosmology(h0=h0, omega_lambda=spec)
    cfg = ParticleConfig(n=n, box=120.0, r0=5.0, g=0.8, sec_balance=0.6,
                         seed=seed, cosmology=cos)
    eng = ParticleEngine(cfg)
    marks = sorted({steps // 8, steps // 4, steps // 2, steps})
    traj, snaps = [], {}
    for t in range(1, steps + 1):
        eng.tick()
        if t % 50 == 0 or t in marks:
            F = eng.density_field(128).cpu().numpy().astype(float)
            p = percolation(F)
            traj.append((t, p, cos.a if cos else 1.0))
            if t in marks:
                snaps[t] = (F, web_metrics(F))
    return traj, snaps, (cos.a if cos else 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--h0", type=float, default=0.02)
    args = ap.parse_args()

    print(f"  {args.n} particles, {args.steps} steps, H0={args.h0}")
    print(f"  POC-07 static reference: percolation peaks 0.188 at t~600, "
          f"decays to 0.050 by t=4000\n")
    hdr = (f"  {'arm':<28} {'peak perc':>10} {'at t':>7} {'final perc':>11} "
           f"{'retained':>9} {'a_final':>8} {'final cv':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows, curves, panels = [], {}, {}
    for label, ol in ARMS:
        traj, snaps, a_end = run(ol, args.steps, args.n, args.h0)
        ts = np.array([t for t, _, _ in traj])
        ps = np.array([p for _, p, _ in traj])
        k = int(np.argmax(ps))
        final = float(ps[-1])
        retained = final / float(ps[k]) if ps[k] > 0 else float("nan")
        w = snaps[max(snaps)][1]
        print(f"  {label:<28} {ps[k]:>10.3f} {ts[k]:>7d} {final:>11.3f} "
              f"{retained:>8.1%} {a_end:>8.2f} {w['cv']:>9.3f}")
        rows.append({"arm": label, "spec": (list(ol) if isinstance(ol, tuple) else ol),
                     "peak": float(ps[k]),
                     "peak_t": int(ts[k]), "final": final, "retained": retained,
                     "a_final": a_end, "final_metrics": {
                         kk: (bool(v) if isinstance(v, bool) else float(v))
                         for kk, v in w.items()}})
        curves[label] = (ts, ps)
        panels[label] = snaps[max(snaps)][0]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.2),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    for label, (ts, ps) in curves.items():
        ax.plot(ts, ps, lw=2, label=label,
                ls="--" if "static" in label else "-")
    ax.set_xlabel("step"); ax.set_ylabel("percolation (largest component / overdense set)")
    ax.set_title("Does expansion hold the web open?", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    best = max(curves, key=lambda k: curves[k][1][-1])
    F = panels[best]
    ax2.imshow(np.log1p(F).T, origin="lower", cmap="magma", aspect="equal",
               interpolation="nearest")
    ax2.set_title(f"final state — {best}", fontsize=10)
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.tight_layout()

    outdir = Path(__file__).resolve().parents[1] / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "expansion_holds_web.png", dpi=115, bbox_inches="tight")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (outdir / f"exp_01_{stamp}.json").write_text(json.dumps(
        {"steps": args.steps, "n": args.n, "h0": args.h0, "arms": rows}, indent=2),
        encoding="utf-8")
    print(f"\n  best final percolation: {best}")
    print(f"  wrote results/expansion_holds_web.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
