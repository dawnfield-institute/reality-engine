#!/usr/bin/env python3
"""Exploration: stop protecting the engine and see where it starts to live.

The Imperfection Engine, section 2: "Without noise, there's no information to collapse.
RBF models thrive on uncertainty; pruning requires alternatives." Section 3: "Collapse is
only meaningful when there is loss — entropy is required."

Every run in this POC so far had `noise_scale=0.0`, PAC enforced to 1e-12, a tanh clamp on
E and I, a hard cap on M, and a uniform correction applied every tick. The engine is built
to stay well-behaved, and a machine with no friction has nothing to collapse. So this sweep
turns the protections OFF and the noise UP, by orders of magnitude, and looks at what happens.

Divergence is a result here, not a failure — the tick it happens at is information about
where the stabilising term was doing real work. Nothing in this script passes or fails.

    python proof_of_concepts/v4/explore_friction.py [--ticks 2000]
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
from structure import coherent_fraction, web_metrics  # noqa: E402


def run(ticks: int, seed: int = 42, **cfg):
    torch.manual_seed(seed)
    eng = Engine(config=SimulationConfig(nu=128, nv=32, **cfg),
                 pipeline=build_canonical_pipeline())
    eng.initialize("big_bang", temperature=3.0)

    pac0 = None
    diverged = None
    for t in range(1, ticks + 1):
        eng.tick()
        s = eng.state
        if pac0 is None:
            pac0 = (s.E + s.I + s.M).sum().item()
        if not torch.isfinite(s.M).all() or s.M.abs().max().item() > 1e12:
            diverged = t
            break

    s = eng.state
    M = s.M.detach().cpu().numpy().astype(float)
    if not np.isfinite(M).all():
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    pac_now = float((s.E + s.I + s.M).sum().item())
    drift = abs(pac_now - pac0) / (abs(pac0) + 1e-12) if pac0 else float("nan")
    return M, diverged, drift


# Each row loosens something the engine uses to stay well-behaved.
CASES = [
    # label                          config
    ("baseline (all protections)",   dict(noise_scale=0.0)),
    ("noise 0.01 (default)",         dict(noise_scale=0.01)),
    ("noise 0.1",                    dict(noise_scale=0.1)),
    ("noise 0.5",                    dict(noise_scale=0.5)),
    ("noise 2.0",                    dict(noise_scale=2.0)),
    ("PAC unenforced",               dict(noise_scale=0.0, enforce_pac=False)),
    ("PAC unenforced + noise 0.5",   dict(noise_scale=0.5, enforce_pac=False)),
    ("NO normalization",             dict(noise_scale=0.0, enable_normalization=False)),
    ("NO normalization + noise 0.5", dict(noise_scale=0.5, enable_normalization=False)),
    ("field_scale 200 (loose clamp)", dict(noise_scale=0.1, field_scale=200.0)),
    ("eta 0.2 (fast forgetting)",    dict(noise_scale=0.1, deactualization_rate=0.2)),
    ("eta 0.0 (no forgetting)",      dict(noise_scale=0.1, deactualization_rate=0.0)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=2000)
    args = ap.parse_args()

    ref = web_metrics(np.abs(np.random.default_rng(0).standard_normal((128, 32))))
    print(f"  128x32, {args.ticks} ticks")
    print(f"  white noise |N(0,1)|:  void {ref['void']:.3f}  cv {ref['cv']:.3f}  "
          f"xi {ref['xi_u']:.3f}")
    print(f"  exp_09 particle web:   void 0.50   cv ~2.0\n")

    hdr = (f"  {'case':<30} {'xi_u':>7} {'coh':>7} {'void':>7} {'cv':>7} "
           f"{'ledger drift':>13} {'note':>16}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows, panels = [], []
    for label, cfg in CASES:
        M, diverged, drift = run(args.ticks, **cfg)
        w = web_metrics(M)
        note = f"DIVERGED t={diverged}" if diverged else ""
        print(f"  {label:<30} {w['xi_u']:>7.3f} {coherent_fraction(M):>7.4f} "
              f"{w['void']:>7.3f} {w['cv']:>7.3f} {drift:>13.4f} {note:>16}")
        rows.append({"case": label, "config": {k: v for k, v in cfg.items()},
                     "diverged_at": diverged, "ledger_drift": drift,
                     "coherent_fraction": coherent_fraction(M), **w})
        panels.append((label, M))

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.5 * nrow))
    for ax, (label, M) in zip(axes.ravel(), panels):
        span = np.percentile(M, 99.5) - M.min()
        ax.imshow(M.T, origin="lower", cmap="magma", aspect="auto",
                  interpolation="nearest", vmin=M.min(),
                  vmax=M.min() + (span if span > 0 else 1.0))
        w = web_metrics(M)
        ax.set_title(f"{label}\nvoid {w['void']:.2f}  cv {w['cv']:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Mass field with the protections loosened — friction sweep", fontsize=11)
    fig.tight_layout()

    outdir = REPO / "proof_of_concepts" / "v4" / "poc_05_structure_exploration" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "friction_sweep.png", dpi=110, bbox_inches="tight")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (outdir / f"friction_sweep_{stamp}.json").write_text(
        json.dumps({"ticks": args.ticks, "white_noise_reference": ref, "rows": rows},
                   indent=2), encoding="utf-8")
    print(f"\n  wrote {(outdir / 'friction_sweep.png').relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
