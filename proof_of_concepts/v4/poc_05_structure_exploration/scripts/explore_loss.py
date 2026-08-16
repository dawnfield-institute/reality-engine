#!/usr/bin/env python3
"""Exploration: push the loss channel. How much forgetting does structure want?

The clamp ladder produced a false lead worth recording: opening `field_scale` raised the
coherent fraction from 0.09 to 0.31, but the saturated fraction rose from 0.00 to 0.49 in
lockstep. Half the field pinned at a higher ceiling is a binary blobby field, and a binary
blobby field has lots of low-k power. Contrast got WORSE as the clamp opened (cv 0.825 ->
0.758) and voids shrank (0.264 -> 0.189). The clamp is not what was suppressing structure.

What did move everything the right way at once, with zero saturation, was **eta** — the
de-actualization rate, the engine's loss channel, memory fading where disequilibrium
resolves. At the default field_scale, eta 0.025 -> 0.2 gave coh 0.092 -> 0.120, void 0.264
-> 0.284, cv 0.825 -> 0.958.

"Collapse is only meaningful when there is loss — entropy is required."
(The Imperfection Engine, section 3.)

So: how far does that go? eta is swept past anything the engine has been run at. The default
0.025 came from spike 04 as the value that OPTIMISED COUPLING ACCURACY — a different
objective entirely, and no reason it should also be the value that builds geometry.

    python proof_of_concepts/v4/explore_loss.py [--ticks 3000]
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


def run(ticks, eta, seed=42, noise=0.0, field_scale=20.0):
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
    sat = float((M >= (field_scale / 5.0) * 0.99).mean())
    return M, diverged, sat, float(M.sum())


ETAS = [0.0, 0.025, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 7, 99])
    args = ap.parse_args()

    ref = np.abs(np.random.default_rng(0).standard_normal((128, 32)))
    rw = web_metrics(ref)
    print(f"  128x32, {args.ticks} ticks, noise off, field_scale 20, "
          f"{len(args.seeds)} seeds")
    print(f"  white noise: coh {coherent_fraction(ref):.4f}  void {rw['void']:.3f}  "
          f"cv {rw['cv']:.3f}   |   exp_09 web: void 0.50  cv ~2.0\n")

    hdr = (f"  {'eta':>6} {'coh':>16} {'void':>16} {'cv':>16} {'M_total':>10} "
           f"{'sat':>6}  note")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows, panels = [], []
    for eta in ETAS:
        res = [run(args.ticks, eta, seed=s) for s in args.seeds]
        if any(r[1] for r in res):
            first = next(r[1] for r in res if r[1])
            print(f"  {eta:>6.3f} {'DIVERGED t=' + str(first):>16}")
            continue
        cohs = [coherent_fraction(r[0]) for r in res]
        ws = [web_metrics(r[0]) for r in res]
        voids = [w["void"] for w in ws]
        cvs = [w["cv"] for w in ws]
        mt = float(np.mean([r[3] for r in res]))
        sat = float(np.mean([r[2] for r in res]))
        print(f"  {eta:>6.3f} {np.mean(cohs):>8.4f}+-{np.std(cohs):<6.4f} "
              f"{np.mean(voids):>8.3f}+-{np.std(voids):<6.3f} "
              f"{np.mean(cvs):>8.3f}+-{np.std(cvs):<6.3f} {mt:>10.1f} {sat:>6.3f}")
        rows.append({"eta": eta, "coh": np.mean(cohs), "coh_std": np.std(cohs),
                     "void": np.mean(voids), "void_std": np.std(voids),
                     "cv": np.mean(cvs), "cv_std": np.std(cvs),
                     "M_total": mt, "saturated": sat, "n_seeds": len(args.seeds)})
        panels.append((f"eta={eta}", res[0][0], ws[0]))

    if panels:
        ncol = min(4, len(panels))
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, (label, M, w) in zip(axes, panels):
            hi = np.percentile(M, 99.5)
            ax.imshow(M.T, origin="lower", cmap="magma", aspect="auto",
                      interpolation="nearest", vmin=M.min(),
                      vmax=hi if hi > M.min() else M.min() + 1.0)
            ax.set_title(f"{label}   void {w['void']:.2f}  cv {w['cv']:.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[len(panels):]:
            ax.axis("off")
        fig.suptitle("Pushing the loss channel — de-actualization rate eta", fontsize=11)
        fig.tight_layout()

        outdir = (REPO / "proof_of_concepts" / "v4" /
                  "poc_05_structure_exploration" / "results")
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "loss_ladder.png", dpi=110, bbox_inches="tight")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (outdir / f"loss_ladder_{stamp}.json").write_text(
            json.dumps({"ticks": args.ticks, "seeds": args.seeds, "rows": rows}, indent=2),
            encoding="utf-8")
        print(f"\n  wrote {(outdir / 'loss_ladder.png').relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
