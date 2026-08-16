#!/usr/bin/env python3
"""Does the particle substrate build a web — and what laws does it have?

Two questions, one run, and the second is the one that makes the first trustworthy.

1. **Web.** Measured with exactly the instruments the field engine was measured with, on a
   density field binned from the particles at the same resolution, so the numbers are
   directly comparable. Field engine: percolation 0.012, ~1300 fragments, `is_web` False
   everywhere.

2. **Laws.** The law detector returned "0 EMERGENT, 1 ENFORCED" on the field engine, and I
   did not trust that null, because the detector had only been calibrated on clean N-body and
   diffusion — never on a weak law in a noisy many-body system. Here it gets real objects to
   track, in a system with a KNOWN interaction. If it recovers the force law from 4000
   interacting particles, the null on the field engine means something. If it cannot, the
   null was an artifact and says so.

Nothing here is corrected into conserving. `PACLedger` is read-only, so whatever holds, holds.

    python proof_of_concepts/v4/poc_07_particle_substrate/scripts/exp_01_does_it_web.py
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

from law_detector import conservation_scan, fit_force_law  # noqa: E402
from particles import ParticleConfig, ParticleEngine  # noqa: E402
from structure import (coherent_fraction, correlation_length,  # noqa: E402
                       percolation, web_metrics)


def measure_force_law(engine, steps=400, sample=4, pairs=600, seed=0):
    """Fit |a| vs r for many particle pairs — the detector, on real objects.

    Uses the acceleration of each particle projected onto the direction of its NEAREST
    neighbour, which for a short-range interaction is the dominant term. Sampled over many
    pairs rather than tracking one, because a many-body system has no clean two-body orbit.
    """
    c = engine.config
    dt = c.dt * sample
    hist = []
    for t in range(steps):
        engine.tick()
        if t % sample == 0:
            hist.append(engine.state.pos.detach().clone())
    if len(hist) < 3:
        return float("nan"), float("nan"), 0

    rng = np.random.default_rng(seed)
    seps, accs = [], []
    for k in range(1, len(hist) - 1):
        p0, p1, p2 = hist[k - 1], hist[k], hist[k + 1]
        box = engine.state.box

        def wrap(d):
            return d - box * torch.round(d / box)

        a = (wrap(p2 - p1) - wrap(p1 - p0)) / dt ** 2      # (N,2) acceleration
        d = wrap(p1.unsqueeze(1) - p1.unsqueeze(0))
        r = torch.sqrt((d ** 2).sum(-1) + 1e-8)
        r.fill_diagonal_(float("inf"))
        j = r.argmin(dim=1)
        idx = torch.from_numpy(rng.choice(p1.shape[0], size=min(pairs, p1.shape[0]),
                                          replace=False)).to(p1.device)
        rr = r[idx, j[idx]]
        unit = d[idx, j[idx]] / rr.unsqueeze(-1)
        a_rad = -(a[idx] * unit).sum(-1)                    # toward the neighbour = positive
        seps.append(rr.cpu().numpy())
        accs.append(a_rad.cpu().numpy())

    sep = np.concatenate(seps)
    acc = np.concatenate(accs)
    return fit_force_law(sep, acc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    cfg = ParticleConfig(n=args.n, box=120.0, r0=5.0, g=0.8, sec_balance=0.6)
    eng = ParticleEngine(cfg)
    print(f"  particle substrate: {cfg.n} particles, box {cfg.box}, r0 {cfg.r0}, "
          f"{args.steps} steps, device {eng.device}")

    ref = np.abs(np.random.default_rng(0).standard_normal((128, 128)))
    print(f"  white noise: percolation {percolation(ref):.3f}  cv {web_metrics(ref)['cv']:.3f}")
    print(f"  field engine (yesterday's best): percolation 0.012  cv 1.47  is_web False")
    print(f"  exp_09 target: void 0.50  filament 0.12  cv ~2.0\n")

    history: dict[str, list[float]] = {}
    snaps = {}
    marks = {args.steps // 6, args.steps // 2, args.steps}
    for t in range(1, args.steps + 1):
        s = eng.tick()
        for k, v in s.metrics.items():
            if isinstance(v, (int, float)):
                history.setdefault(k, []).append(float(v))
        if t in marks:
            F = eng.density_field(128).cpu().numpy().astype(float)
            w = web_metrics(F)
            snaps[t] = (F, w)
            print(f"  t={t:<5} percolation {w['percolation']:.3f}  void {w['void']:.3f}  "
                  f"cv {w['cv']:.3f}  xi {w['xi_u']:.3f}  filament {w['filament']:.3f}  "
                  f"is_web {w['is_web']}")

    # --- laws ---
    print("\n  conservation scan (nothing is corrected — read-only ledger):")
    # Particles start at rest, so momentum's initial value is ~0 and using it as the scale
    # gives CV ~1e10 for a perfectly fine quantity. Scale by a typical single-particle
    # momentum instead — the same trap the detector's own docstring warns about.
    typ = max(1e-12, float(np.mean(np.abs(history["kinetic"])) ** 0.5))
    scales = {"momentum_x": typ, "momentum_y": typ}
    for law in conservation_scan(history, tol=1e-3, scales=scales):
        print(f"   {law}")

    print("\n  force law from tracked particles:")
    n, r2, npts = measure_force_law(eng, steps=200)
    print(f"    |a| ~ r^{n:.3f}   R2 {r2:.4f}   {npts} samples")
    print(f"    (built with F ~ exp(-r/r0)/r — a power-law fit over the short-range part "
          f"should be steeper than -1)")

    # --- render ---
    fig, axes = plt.subplots(1, len(snaps), figsize=(4.6 * len(snaps), 4.4))
    for ax, (t, (F, w)) in zip(np.atleast_1d(axes), sorted(snaps.items())):
        ax.imshow(np.log1p(F).T, origin="lower", cmap="magma", aspect="equal",
                  interpolation="nearest")
        ax.set_title(f"t={t}  perc {w['percolation']:.2f}  void {w['void']:.2f}  "
                     f"cv {w['cv']:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Particle substrate — density field, log scale", fontsize=12)
    fig.tight_layout()
    outdir = Path(__file__).resolve().parents[1] / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "web_formation.png", dpi=115, bbox_inches="tight")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (outdir / f"exp_01_{stamp}.json").write_text(json.dumps({
        "config": vars(cfg),
        "snapshots": {str(t): {k: (bool(v) if isinstance(v, bool) else float(v))
                               for k, v in w.items()} for t, (_, w) in snaps.items()},
        "force_law": {"exponent": n, "r_squared": r2, "n_samples": npts},
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote results/web_formation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
