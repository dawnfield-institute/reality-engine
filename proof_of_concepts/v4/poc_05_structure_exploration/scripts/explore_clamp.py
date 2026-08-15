#!/usr/bin/env python3
"""Exploration: how far can the clamp be opened before the engine dies, and what appears?

The friction sweep found `field_scale` to be the largest single lever on large-scale
coherence — 20 -> 200 took the coherent fraction from 0.083 to 0.203 against a white-noise
floor of 0.072, and it is the only setting whose render shows extended bright and dark
domains rather than speckle. Removing normalization entirely instead collapses to a handful
of points and diverges at t~263, so `field_scale` is the controlled version of the same move:
it opens the clamp smoothly instead of deleting it.

Two questions, no thresholds:
  - where along the ladder does structure appear, and where does the engine stop surviving?
  - does loss (eta, the de-actualization rate) compound with an open clamp?

`saturated` is reported because it is the honest caveat on the headline number: a coherent
fraction can rise because big domains formed, or because big regions pinned themselves to a
higher ceiling. Those look the same in one statistic and different in a render.

    python proof_of_concepts/v4/explore_clamp.py [--ticks 3000]
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

from src.v3.engine.config import SimulationConfig  # noqa: E402
from src.v3.engine.engine import Engine  # noqa: E402
from src.v3.engine.pipelines import build_canonical_pipeline  # noqa: E402
from structure import coherent_fraction, correlation_length, web_metrics  # noqa: E402


def run(ticks, field_scale, eta, seed=42, noise=0.0):
    torch.manual_seed(seed)
    eng = Engine(
        config=SimulationConfig(nu=128, nv=32, noise_scale=noise,
                                field_scale=field_scale, deactualization_rate=eta),
        pipeline=build_canonical_pipeline())
    eng.initialize("big_bang", temperature=3.0)
    diverged = None
    for t in range(1, ticks + 1):
        eng.tick()
        if not torch.isfinite(eng.state.M).all() or eng.state.M.abs().max().item() > 1e12:
            diverged = t
            break
    M = eng.state.M.detach().cpu().numpy().astype(float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    # M is capped at field_scale/5 by NormalizationOperator; how much sits at the ceiling?
    cap = field_scale / 5.0
    sat = float((M >= cap * 0.99).mean())
    return M, diverged, sat


SCALES = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0]
ETAS = [0.025, 0.2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    args = ap.parse_args()

    ref = np.abs(np.random.default_rng(0).standard_normal((128, 32)))
    rw = web_metrics(ref)
    print(f"  128x32, {args.ticks} ticks, noise off")
    print(f"  white noise: coh {coherent_fraction(ref):.4f}  cv {rw['cv']:.3f}  "
          f"xi {rw['xi_u']:.3f}   |   exp_09 web: void 0.50  cv ~2.0\n")

    hdr = (f"  {'field_scale':>12} {'eta':>6} {'xi_u':>7} {'coh':>7} {'void':>7} "
           f"{'cv':>7} {'saturated':>10}  note")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows, panels = [], []
    for eta in ETAS:
        for fs in SCALES:
            M, diverged, sat = run(args.ticks, fs, eta)
            w = web_metrics(M)
            coh = coherent_fraction(M)
            note = f"DIVERGED t={diverged}" if diverged else ""
            print(f"  {fs:>12.0f} {eta:>6.3f} {w['xi_u']:>7.3f} {coh:>7.4f} "
                  f"{w['void']:>7.3f} {w['cv']:>7.3f} {sat:>10.3f}  {note}")
            rows.append({"field_scale": fs, "eta": eta, "diverged_at": diverged,
                         "saturated": sat, "coherent_fraction": coh, **w})
            panels.append((f"fs={fs:.0f} eta={eta}", M, w, sat))
        print()

    ncol = len(SCALES)
    fig, axes = plt.subplots(len(ETAS), ncol, figsize=(3.1 * ncol, 2.6 * len(ETAS)))
    axes = np.atleast_2d(axes)
    for i, (label, M, w, sat) in enumerate(panels):
        ax = axes[i // ncol, i % ncol]
        hi = np.percentile(M, 99.5)
        ax.imshow(M.T, origin="lower", cmap="magma", aspect="auto",
                  interpolation="nearest", vmin=M.min(),
                  vmax=hi if hi > M.min() else M.min() + 1.0)
        ax.set_title(f"{label}\ncv {w['cv']:.2f}  sat {sat:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Opening the clamp — mass field along the field_scale ladder", fontsize=11)
    fig.tight_layout()

    outdir = REPO / "proof_of_concepts" / "v4" / "poc_05_structure_exploration" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "clamp_ladder.png", dpi=110, bbox_inches="tight")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (outdir / f"clamp_ladder_{stamp}.json").write_text(
        json.dumps({"ticks": args.ticks, "rows": rows}, indent=2), encoding="utf-8")
    print(f"  wrote {(outdir / 'clamp_ladder.png').relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
