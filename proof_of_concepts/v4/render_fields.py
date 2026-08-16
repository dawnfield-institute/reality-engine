#!/usr/bin/env python3
"""Render the mass field. The instrument that was missing.

Hours were spent computing statistics on a 2D field without once looking at it. A broken
box-counting estimator returned D = 2.000 for a filament and went unnoticed; one rendered
frame beside the number would have exposed it immediately. Fractal dimension also cannot
distinguish a filament from scattered rubble at equal occupancy — the eye does that
instantly.

    python proof_of_concepts/v4/render_fields.py [--ticks 5000] [--grid 128 64]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.v3.engine.config import SimulationConfig  # noqa: E402
from src.v3.engine.engine import Engine  # noqa: E402
from src.v3.engine.pipelines import CANONICAL  # noqa: E402
from src.v3.operators.pac_balance import PACBalanceOperator  # noqa: E402
from src.v3.operators.protocol import Pipeline  # noqa: E402
from src.v3.operators.rbf import RBFOperator  # noqa: E402
from structure import box_dimension  # noqa: E402


def build(balance: str) -> Pipeline:
    ops = [PACBalanceOperator if (o is RBFOperator and balance == "pac") else o
           for o in CANONICAL]
    return Pipeline([o() for o in ops])


def run_capture(balance, ticks, grid, init, seed, marks):
    nu, nv = grid
    torch.manual_seed(seed)
    eng = Engine(config=SimulationConfig(nu=nu, nv=nv, noise_scale=0.0),
                 pipeline=build(balance))
    eng.initialize(init, temperature=3.0)
    frames = {}
    for t in range(1, ticks + 1):
        eng.tick()
        if t in marks:
            M = eng.state.M.detach().cpu().numpy().astype(float)
            frames[t] = (M, box_dimension(eng.state.M))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--grid", type=int, nargs=2, default=[128, 64])
    ap.add_argument("--init", default="big_bang")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    marks = [args.ticks // 10, args.ticks // 3, args.ticks]
    runs = {}
    for bal, label in (("rbf", "RBF (current)"), ("pac", "PACBalance (ADE-derived)")):
        print(f"  running {label} ...")
        runs[label] = run_capture(bal, args.ticks, tuple(args.grid),
                                  args.init, args.seed, marks)

    fig, axes = plt.subplots(len(runs), len(marks),
                             figsize=(4.2 * len(marks), 3.0 * len(runs)))
    if len(runs) == 1:
        axes = np.array([axes])

    for r, (label, frames) in enumerate(runs.items()):
        for c, t in enumerate(marks):
            M, D = frames[t]
            ax = axes[r, c]
            # percentile clip so a few hot cells don't wash the image out
            vmax = np.percentile(M, 99.5)
            ax.imshow(M.T, origin="lower", cmap="magma", vmin=M.min(), vmax=vmax,
                      aspect="auto", interpolation="nearest")
            ax.set_title(f"{label}\nt={t}   D={D:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Mass field M — {args.init}, seed {args.seed}, "
                 f"{args.grid[0]}x{args.grid[1]}", fontsize=11)
    fig.tight_layout()
    out = REPO / "proof_of_concepts" / "v4" / "results" / "mass_field_rbf_vs_pac.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  wrote {out.relative_to(REPO).as_posix()}")

    # Also dump a threshold view — what box_dimension actually measures.
    fig2, axes2 = plt.subplots(len(runs), len(marks),
                               figsize=(4.2 * len(marks), 3.0 * len(runs)))
    if len(runs) == 1:
        axes2 = np.array([axes2])
    for r, (label, frames) in enumerate(runs.items()):
        for c, t in enumerate(marks):
            M, D = frames[r] if False else frames[t]
            cut = np.quantile(M - M.min(), 0.90)
            occ = (M - M.min()) >= cut
            ax = axes2[r, c]
            ax.imshow(occ.T, origin="lower", cmap="gray", aspect="auto",
                      interpolation="nearest")
            ax.set_title(f"{label}\nt={t}  top-10%  D={D:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig2.suptitle("Thresholded set actually measured by box_dimension (top 10%)",
                  fontsize=11)
    fig2.tight_layout()
    out2 = REPO / "proof_of_concepts" / "v4" / "results" / "mass_field_thresholded.png"
    fig2.savefig(out2, dpi=110, bbox_inches="tight")
    print(f"  wrote {out2.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
