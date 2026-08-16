#!/usr/bin/env python3
"""POC-05 exp_01 — the structure panel, with calibrated per-direction metrics.

This exists because the panel that founded Milestone 16 was never committed. Its numbers —
correlation length 1.00 cell, spectral tilt 9.09x vs 2.23x, a preferred wavelength moving
23.2 -> 10.5 -> 2.1 with quantum pressure — lived only in journal prose, produced by
uncommitted code with no calibrated estimator behind it. After a box-counting estimator
returned D = 2.000 for a straight filament and survived a full day, that is not a state a
milestone can rest on.

Three things this run does that the original did not:

1. **Reports xi per direction.** The manifold is 128 (periodic circumference) x 32 (bounded
   strip width). A circular FFT along the bounded axis underestimates xi by ~2x on a field
   that is genuinely smooth there — measured, not assumed. An isotropic average over a 4:1
   grid buries whatever exists along `u`.
2. **Reports coherent_fraction beside xi.** A correlation length at the floor is consistent
   with a field that has no structure AND with one whose structure is real but buried under
   pointwise variance. A lambda=16 cosine under 3x noise reads xi = 0.66, the floor. Only
   the coherent fraction separates them.
3. **Carries a white-noise control row.** Every number is reported against what pure noise
   scores on the same grid, so "at the floor" is a comparison rather than an assertion.

    python proof_of_concepts/v4/poc_05_structure_exploration/scripts/exp_01_structure_panel.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from harness import assert_params_live  # noqa: E402
from src.v3.engine.config import SimulationConfig  # noqa: E402
from src.v3.engine.engine import Engine  # noqa: E402
from src.v3.engine.pipelines import CANONICAL  # noqa: E402
from src.v3.operators.gravity import GravitationalCollapseOperator  # noqa: E402
from src.v3.operators.pac_balance import PACBalanceOperator  # noqa: E402
from src.v3.operators.protocol import Pipeline  # noqa: E402
from src.v3.operators.rbf import RBFOperator  # noqa: E402
from structure import (box_dimension, coherent_fraction, contrast,  # noqa: E402
                       correlation_length, power_spectrum, spectral_tilt, web_metrics)


def build(balance: str = "rbf", gravity: bool = True) -> Pipeline:
    ops = []
    for o in CANONICAL:
        if o is RBFOperator and balance == "pac":
            ops.append(PACBalanceOperator)
            continue
        if o is GravitationalCollapseOperator and not gravity:
            continue
        ops.append(o)
    return Pipeline([o() for o in ops])


def measure(M) -> dict:
    """Every structural number for one field snapshot, per direction where it matters.

    `coherent_fraction` and the web metrics answer different questions and must both be
    read. Excess low-k power says the field has a large-scale component; the web criterion
    says whether that component has the GEOMETRY of a web — voids and dense filaments — or
    is merely smooth clumping. A field can pass the first and fail the second, and the
    engine does.
    """
    out = {
        "xi_u": correlation_length(M, axis=0, periodic=True),
        "xi_v": correlation_length(M, axis=1, periodic=False),
        "coherent_fraction": coherent_fraction(M),
        "D_box": box_dimension(M),
        "contrast": contrast(M),
    }
    out.update(web_metrics(M))
    return out


def reference_row(nu: int, nv: int, seed: int = 0) -> dict:
    """White noise on the same grid. The floor, measured rather than asserted.

    `|N(0,1)|`, not `N(0,1)`: M is non-negative, and the density CV of a zero-mean field is
    undefined (`std/mean` with mean ~ 0). A signed reference reported cv = nan and made the
    control column useless for exactly the comparison it existed to support.
    """
    noise = np.abs(np.random.default_rng(seed).standard_normal((nu, nv)))
    return measure(noise)


def run_variant(name: str, ticks: int, grid: tuple[int, int], *, init: str = "big_bang",
                seed: int = 42, balance: str = "rbf", gravity: bool = True,
                **cfg_kw) -> dict:
    nu, nv = grid
    torch.manual_seed(seed)
    cfg = SimulationConfig(nu=nu, nv=nv, noise_scale=cfg_kw.pop("noise_scale", 0.0),
                           **cfg_kw)
    eng = Engine(config=cfg, pipeline=build(balance=balance, gravity=gravity))
    eng.initialize(init, temperature=3.0)

    early_mark, late_mark = max(1, ticks // 10), ticks
    snaps, P_early = {}, None
    t0, diverged = time.time(), None

    for t in range(1, ticks + 1):
        eng.tick()
        if not torch.isfinite(eng.state.M).all():
            diverged = t
            break
        if t == early_mark:
            P_early = power_spectrum(eng.state.M, axis=0, periodic=True)[1]
            snaps[f"t{t}"] = measure(eng.state.M)
        elif t == late_mark:
            snaps[f"t{t}"] = measure(eng.state.M)

    out = {"name": name, "init": init, "balance": balance, "gravity": gravity,
           "seed": seed, "diverged_at": diverged, "wall_s": round(time.time() - t0, 1),
           "snaps": snaps, **cfg_kw}

    if diverged is None and P_early is not None:
        k, P_late = power_spectrum(eng.state.M, axis=0, periodic=True)
        ratio = np.divide(P_late, P_early, out=np.full_like(P_late, np.nan),
                          where=P_early > 0)
        out["spectral_tilt"] = spectral_tilt(P_early, P_late)
        # The "preferred scale" claim is about where GROWTH peaks, not where power sits.
        # Those are different fields and conflating them is how a preferred wavelength of
        # ~10 cells came to coexist with a correlation length of 1 cell in the same record.
        if np.isfinite(ratio).any():
            j = int(np.nanargmax(ratio))
            out["growth_peak_lambda"] = float(1.0 / k[j])
            out["growth_peak_ratio"] = float(ratio[j])
        out["power_peak_lambda"] = float(1.0 / k[int(np.argmax(P_late))])
    return out


VARIANTS: list[tuple[str, dict]] = [
    ("baseline RBF",            dict(balance="rbf")),
    ("PAC balance",             dict(balance="pac")),
    ("RBF, no gravity",         dict(balance="rbf", gravity=False)),
    ("PAC, no gravity",         dict(balance="pac", gravity=False)),
    # The quantum-pressure sweep: the journals claim this sets a preferred scale.
    ("RBF, qp=0.0",             dict(balance="rbf", quantum_pressure_coeff=0.0)),
    ("RBF, qp=0.30",            dict(balance="rbf", quantum_pressure_coeff=0.30)),
    ("PAC, qp=0.0",             dict(balance="pac", quantum_pressure_coeff=0.0)),
    ("PAC, qp=0.30",            dict(balance="pac", quantum_pressure_coeff=0.30)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--grid", type=int, nargs=2, default=[128, 32],
                    help="default is the manifold's own shape, not a square")
    args = ap.parse_args()
    grid = (args.grid[0], args.grid[1])

    # A dead config field silently inflates a panel: varying it produces bit-identical runs
    # that read as independent confirmations. mass_gen_coeff did exactly that to POC-02.
    live = assert_params_live(["quantum_pressure_coeff"], REPO)
    for n, ok in live.items():
        if not ok:
            print(f"  ABORT: '{n}' is read by no operator — sweeping it measures nothing.")
            return 1

    ref = reference_row(*grid)
    print(f"  grid {grid[0]}x{grid[1]}, {args.ticks} ticks")
    print(f"  white-noise reference: xi_u={ref['xi_u']:.3f} xi_v={ref['xi_v']:.3f} "
          f"coh={ref['coherent_fraction']:.4f} cv={ref['cv']:.3f}")
    print("  exp_09 particle web, for scale: void 0.50  cv ~2.0  -> is_web True\n")

    hdr = (f"  {'variant':<20} {'xi_u':>13} {'coherent':>15} {'void':>7} "
           f"{'cv':>7} {'web':>5} {'growth pk':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for name, kw in VARIANTS:
        r = run_variant(name, args.ticks, grid, **kw)
        rows.append(r)
        if r["diverged_at"]:
            print(f"  {name:<20} {'DIVERGED at t=' + str(r['diverged_at']):>13}")
            continue
        a, b = list(r["snaps"].values())
        print(f"  {name:<20} {a['xi_u']:>6.3f}->{b['xi_u']:>6.3f} "
              f"{a['coherent_fraction']:>7.4f}->{b['coherent_fraction']:>7.4f} "
              f"{b['void']:>7.3f} {b['cv']:>7.3f} "
              f"{str(b['is_web']):>5} "
              f"{r.get('growth_peak_lambda', float('nan')):>10.1f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = (Path(__file__).resolve().parents[1] / "results" /
           f"exp_01_structure_panel_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"grid": list(grid), "ticks": args.ticks,
         "white_noise_reference": ref, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
