#!/usr/bin/env python3
"""Structure character as a function of SCALE — where the process re-enters itself.

Every instrument in this directory so far reports ONE number for the whole box: percolation,
void fraction, density CV. That is the wrong shape of measurement for a system whose whole
claim is that the same process recurs at successive scales and behaves differently each time.

Gravity is the worked example. Locally it clumps — the standard N-body outcome, and what the
substrate does at t=400. At larger scale the same force makes filaments and sheets. Those
filaments then clump into nodes, which become the units the next scale organises. Collapse
runs sheet -> filament -> node (Zel'dovich), and because large scales collapse LATER, at any
one instant different scales sit at different points in that sequence. So "is it clumpy or
filamentary" has no single answer — only a profile across scale.

This measures that profile with the standard cosmic-web classifier. Smooth the density at
scale R, take the Hessian of the density field, and count how many of its eigenvalues exceed
a threshold at each point:

    3 positive  -> NODE      collapsing on all three axes
    2 positive  -> FILAMENT  collapsed on two, extended along one
    1 positive  -> SHEET     collapsed on one
    0 positive  -> VOID      expanding on all three

Sweeping R gives the character spectrum. A single dominant class at every scale means one
process at one scale. **Character that CHANGES with R is the recursion**, and where it changes
is where the transition sits.

Run it on the clock field too: if local time recurs the way matter does, tau's character
spectrum should show its own structure — and whether it tracks matter's at the same R or at a
shifted one is the question worth asking.

    python recursion.py [--steps 180] [--epochs 60 180 400] [--dims 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V4))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from particles import EXP11_TIME, ParticleConfig, ParticleEngine  # noqa: E402


OUT = HERE.parent / "results"
CLASSES = ["void", "sheet", "filament", "node"]
COLORS = {"void": "#4C6EF5", "sheet": "#12B886", "filament": "#F59F00", "node": "#E03131"}


def smooth(F, sigma):
    if sigma <= 0:
        return F
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(F, sigma=sigma, mode="wrap")
    except ImportError:                                   # spectral fallback, same kernel
        d = F.ndim
        k = np.meshgrid(*[np.fft.fftfreq(n) * 2 * np.pi for n in F.shape], indexing="ij")
        k2 = sum(x ** 2 for x in k)
        return np.fft.ifftn(np.fft.fftn(F) * np.exp(-0.5 * k2 * sigma ** 2)).real


def classify(F, sigma, thresh=0.0):
    """Hessian eigenvalue classification of a smoothed field.

    Returns the fraction of cells in each class. The Hessian is taken with periodic finite
    differences, matching the box. `thresh` shifts what counts as "collapsing"; 0 is the
    standard choice and is used throughout so scales stay comparable.
    """
    S = smooth(F, sigma)
    d = S.ndim
    # H[i][j] = d2 S / dx_i dx_j, periodic
    g = np.gradient(S, edge_order=2)
    H = np.empty(S.shape + (d, d))
    for i in range(d):
        gi = np.gradient(g[i], edge_order=2)
        for j in range(d):
            H[..., i, j] = gi[j]
    H = 0.5 * (H + np.swapaxes(H, -1, -2))                # symmetrise against FD asymmetry
    ev = np.linalg.eigvalsh(H)
    # Density Hessian: NEGATIVE eigenvalues mark a local maximum (collapse), so count those.
    ncoll = (ev < -thresh).sum(axis=-1)
    return {CLASSES[k]: float((ncoll == k).mean()) for k in range(min(d, 3) + 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=3, choices=(2, 3))
    ap.add_argument("-n", type=int, default=8000)
    ap.add_argument("--box", type=float, default=60.0)
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=1.5)
    ap.add_argument("--sec-balance", dest="sec_balance", type=float, default=0.65334)
    ap.add_argument("--epochs", type=int, nargs="+", default=[60, 180, 400])
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.6, 1.0, 1.6, 2.4, 3.4, 4.6, 6.0])
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-mode", dest="time_mode", default="potential")
    ap.add_argument("--viscosity", type=float, default=0.3)
    a = ap.parse_args()

    cfg = ParticleConfig(n=a.n, box=a.box, r0=a.r0, g=a.g, sec_balance=a.sec_balance,
                         dims=a.dims, seed=a.seed, entropy_init=0.1,
                         time_mode=a.time_mode, time_viscosity=a.viscosity)
    eng = ParticleEngine(cfg, pipeline=EXP11_TIME)
    cell = a.box / a.res
    print(f"  {a.n} particles, {a.dims}D, grid {a.res}^{a.dims}, cell {cell:.2f}, "
          f"r0 {a.r0} = {a.r0/cell:.1f} cells")
    print(f"  smoothing scales R (cells): {a.scales}")

    out, done = {}, 0
    for ep in a.epochs:
        for _ in range(ep - done):
            eng.tick()
        done = ep
        rho = eng.density_field(a.res).cpu().numpy().astype(float)
        tau = eng.field_of(eng.state.tau, a.res).cpu().numpy() if eng.state.tau is not None else None
        if tau is not None:
            tau = np.nan_to_num(tau, nan=float(np.nanmean(tau)))

        print(f"\n  --- epoch t = {ep} ---")
        print(f"  {'R(cells)':>9} {'R/r0':>6} | " + " ".join(f"{c:>9}" for c in CLASSES)
              + " |  tau: " + " ".join(f"{c:>8}" for c in CLASSES))
        rows = []
        for R in a.scales:
            m = classify(rho, R)
            t = classify(tau, R) if tau is not None else {}
            rows.append({"R_cells": R, "R_over_r0": R * cell / a.r0, "matter": m, "tau": t})
            print(f"  {R:>9.1f} {R*cell/a.r0:>6.2f} | "
                  + " ".join(f"{m.get(c,0)*100:>8.1f}%" for c in CLASSES)
                  + " |        " + " ".join(f"{t.get(c,0)*100:>7.1f}%" for c in CLASSES))
        out[ep] = rows

        fil = [r["matter"].get("filament", 0) for r in rows]
        nod = [r["matter"].get("node", 0) for r in rows]
        kf, kn = int(np.argmax(fil)), int(np.argmax(nod))
        print(f"  filament fraction peaks at R = {a.scales[kf]:.1f} cells "
              f"({a.scales[kf]*cell/a.r0:.2f} r0);  node fraction peaks at "
              f"R = {a.scales[kn]:.1f} ({a.scales[kn]*cell/a.r0:.2f} r0)")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(2, len(a.epochs), figsize=(5.2 * len(a.epochs), 8.2),
                             squeeze=False, sharex=True)
    xs = [R * cell / a.r0 for R in a.scales]
    for j, ep in enumerate(a.epochs):
        for i, key in enumerate(("matter", "tau")):
            ax = axes[i][j]
            ys = {c: [r[key].get(c, 0) for r in out[ep]] for c in CLASSES}
            ax.stackplot(xs, [ys[c] for c in CLASSES],
                         labels=CLASSES, colors=[COLORS[c] for c in CLASSES], alpha=0.9)
            ax.set_xscale("log"); ax.set_xlim(min(xs), max(xs)); ax.set_ylim(0, 1)
            ax.set_title(f"{'matter' if i == 0 else 'clock field tau'}  —  t = {ep}",
                         fontsize=10)
            if i == 1:
                ax.set_xlabel("smoothing scale  R / r0")
            if j == 0:
                ax.set_ylabel("fraction of cells")
            if i == 0 and j == len(a.epochs) - 1:
                ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.suptitle("structure character across scale and epoch — the recursion, if it is there",
                 fontsize=12)
    fig.tight_layout()
    p = OUT / f"recursion_{a.dims}d_{stamp}.png"
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    (OUT / f"recursion_{stamp}.json").write_text(
        json.dumps({"config": vars(a), "cell": cell,
                    "epochs": {str(k): v for k, v in out.items()}}, indent=2), encoding="utf-8")
    print(f"\n  wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
