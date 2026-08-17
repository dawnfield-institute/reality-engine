#!/usr/bin/env python3
"""Replicate exp_11's 3D web before asking anything new.

POC-07 rebuilt exp_09's 2D mechanism and reproduced its filament fraction (0.130 against a
measured 0.12), which is what made the 2D results trustworthy. This is the same gate in three
dimensions. If the rebuild does not land near exp_11's numbers, nothing downstream means
anything and the round stops here.

exp_11's config is notable for not being hand-tuned the way exp_09's was:

    n = 4000, box = 60, r0 = 10, g = 1.5, sec_balance = Xi / phi = 0.65334, 600 steps

`sec_balance` is a DERIVED value — Xi over phi — with the code comment "in 3D material spreads
more, need stronger gravity, lower SEC". exp_10 had found Xi itself optimal in 2D. That factor
of phi between the two dimensions is what exp_03 tests; this script only establishes that the
substrate reproduces the reference.

Measured targets (exp_11): void 0.89, density CV 2.94, clustering 0.50, filament 0.023.

**Filament fraction is NOT comparable.** exp_11 used the 75th percentile of occupied bins,
which is tautologically 0.25 on a continuous field; `web_metrics` uses an absolute overdensity
threshold instead. Void and CV are comparable; filament is reported for the record only.

    python .../exp_01_replicate_exp11.py
"""

from __future__ import annotations

import argparse
import json
import math
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

from particles import PHI, ParticleConfig, ParticleEngine  # noqa: E402
from structure import percolation, web_metrics  # noqa: E402

XI = 0.5772156649015329 + math.log(PHI)          # gamma + ln(phi) = 1.05843
SEC_EXP11 = XI / PHI                             # 0.65334 — exp_11's derived choice


def ratios(cfg) -> dict:
    """The dimensionless numbers. Absolute g, r0 and box are cosmological-scale INPUTS that
    the corpus does not derive (exp_31 Part B), so comparisons across runs and across
    dimension have to be made on these."""
    d = cfg.dims
    v_ball = (math.pi ** (d / 2) / math.gamma(d / 2 + 1)) * cfg.r0 ** d
    return {"neighbours_in_range": cfg.n * v_ball / cfg.box ** d,
            "sec_over_g": cfg.sec_balance / cfg.g,
            "r0_over_L": cfg.r0 / cfg.box,
            "dims": d}


def run(cfg, steps, res=32, marks=()):
    eng = ParticleEngine(cfg)
    snaps = {}
    for t in range(1, steps + 1):
        eng.tick()
        if t in marks:
            F = eng.density_field(res).cpu().numpy().astype(float)
            snaps[t] = (F, web_metrics(F))
    return eng, snaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    # exp_11 bins at 32^3. 64^3 puts 4000 particles on 262144 cells -- 0.015 per cell --
    # where the density field is empty by construction and ANY web reads as disconnected.
    # That default is what produced this POC's retracted "does not percolate" claim, and the
    # CV 15.4 its own meta.yaml records as the failed first attempt. See the 2026-08-17 journal.
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 7, 99])
    args = ap.parse_args()

    base = dict(n=4000, box=60.0, r0=10.0, g=1.5, sec_balance=SEC_EXP11, dims=3)
    print(f"  exp_11 config: n=4000 box=60 r0=10 g=1.5 sec_balance=Xi/phi={SEC_EXP11:.5f}, "
          f"{args.steps} steps, 3D")
    print(f"  dimensionless: {ratios(ParticleConfig(**base, seed=0))}")
    print(f"  exp_11 measured: void 0.89   CV 2.94   clustering 0.50   "
          f"(filament 0.023, not comparable)\n")

    rng = np.random.default_rng(0)
    ref = np.abs(rng.standard_normal((args.res,) * 3))
    print(f"  3D white-noise control: perc {percolation(ref):.4f}  "
          f"void {web_metrics(ref)['void']:.3f}  cv {web_metrics(ref)['cv']:.3f}\n")

    marks = {args.steps // 3, args.steps}
    rows, panels = [], None
    voids, cvs, percs = [], [], []
    for sd in args.seeds:
        eng, snaps = run(ParticleConfig(**base, seed=sd), args.steps, args.res, marks)
        F, w = snaps[args.steps]
        voids.append(w["void"]); cvs.append(w["cv"]); percs.append(w["percolation"])
        rows.append({"seed": sd, **{k: (bool(v) if isinstance(v, bool) else float(v))
                                    for k, v in w.items()}})
        if panels is None:
            panels = snaps

    v, c, pc = np.array(voids), np.array(cvs), np.array(percs)
    print(f"  {'':<14}{'measured':>22}{'exp_11':>10}")
    print(f"  {'void':<14}{v.mean():>13.3f} +-{v.std():<6.3f}{0.89:>10.2f}")
    print(f"  {'density CV':<14}{c.mean():>13.3f} +-{c.std():<6.3f}{2.94:>10.2f}")
    print(f"  {'percolation':<14}{pc.mean():>13.4f} +-{pc.std():<6.4f}{'n/a':>10}")

    ok_void = abs(v.mean() - 0.89) < 0.15
    ok_cv = 0.5 * 2.94 < c.mean() < 2.0 * 2.94
    print(f"\n  replication gate: void {'PASS' if ok_void else 'FAIL'}, "
          f"CV {'PASS' if ok_cv else 'FAIL'}")

    # Render orthogonal slices — no structural claim without a picture, and a 3D projection
    # hides exactly the connectivity this round is about.
    F = panels[args.steps][0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    mid = F.shape[0] // 2
    for a, (sl, lab) in zip(ax, [(F[mid], "x = mid"), (F[:, mid], "y = mid"),
                                 (F[:, :, mid], "z = mid")]):
        a.imshow(np.log1p(sl).T, origin="lower", cmap="magma", aspect="equal",
                 interpolation="nearest")
        a.set_title(f"slice {lab}", fontsize=9); a.set_xticks([]); a.set_yticks([])
    w = panels[args.steps][1]
    fig.suptitle(f"3D web, exp_11 config — void {w['void']:.2f}  cv {w['cv']:.2f}  "
                 f"perc {w['percolation']:.3f}", fontsize=11)
    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "exp11_replication_slices.png", dpi=115, bbox_inches="tight")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out / f"exp_01_{stamp}.json").write_text(json.dumps(
        {"config": base, "steps": args.steps, "res": args.res, "seeds": args.seeds,
         "ratios": ratios(ParticleConfig(**base, seed=0)),
         "exp_11_targets": {"void": 0.89, "cv": 2.94, "clustering": 0.50},
         "gate": {"void": bool(ok_void), "cv": bool(ok_cv)}, "runs": rows},
        indent=2), encoding="utf-8")
    print(f"  wrote results/exp11_replication_slices.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
