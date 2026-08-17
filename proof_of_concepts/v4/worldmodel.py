#!/usr/bin/env python3
"""The world model — run it, watch structure form.

    python proof_of_concepts/v4/worldmodel.py run                    # 2D, renders an animation
    python proof_of_concepts/v4/worldmodel.py run --dims 3           # 3D, orthogonal slices
    python proof_of_concepts/v4/worldmodel.py run --steps 2000 -n 8000
    python proof_of_concepts/v4/worldmodel.py sweep --param sec_balance --values 0.3 0.6 1.0
    python proof_of_concepts/v4/worldmodel.py sweep --param g --values 0.4 0.8 1.6 --dims 3

Writes an animated GIF of the density field plus a metrics trace, and prints the calibrated
structure numbers as it goes. Sweeps render one panel per value so parameter effects are
visible side by side rather than inferred from a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from particles import CANONICAL, EXP11, ParticleConfig, ParticleEngine  # noqa: E402
from structure import web_metrics  # noqa: E402

OUT = HERE / "worldmodel_out"


def slice_of(F: np.ndarray) -> np.ndarray:
    """A 2D view: the field itself in 2D, the mid-plane in 3D."""
    return F if F.ndim == 2 else F[F.shape[0] // 2]


def matched_res(n: int, dims: int) -> int:
    """Grid resolution giving ~1 particle per cell.

    This is not a cosmetic default. Connectivity statistics are read off a thresholded
    density field, and if the grid is finer than the particle count can fill, the field is
    empty by construction: the overdense set shatters into singletons and a genuinely
    connected web reads as disconnected. It fails in the direction that looks like rigour,
    which is why it survived so long.

    Measured on one 3D run, identical physics, binning alone:

        res 16 (0.98 particles/cell) -> percolation 0.472, is_web True
        res 32 (0.12)                -> 0.385, True
        res 64 (0.015)               -> 0.062, False      <- the artifact

    So `n/res**d ~ 1`. Report `occupancy` beside any percolation value and treat an occupancy
    far below the relevant site threshold (0.593 in 2D, 0.312 in 3D) as a sampling warning
    rather than a physical result.
    """
    return max(8, int(round(n ** (1.0 / dims))))


def build(dims, n, box, r0, g, sec, seed, convention, ic="zeldovich",
          ic_index=-1.727, ic_amplitude=2.0):
    pipe = EXP11 if convention == "exp11" else CANONICAL
    ei = 0.1 if convention == "exp11" else 0.0
    cfg = ParticleConfig(n=n, box=box, r0=r0, g=g, sec_balance=sec, dims=dims,
                         seed=seed, entropy_init=ei, ic=ic, ic_index=ic_index,
                         ic_amplitude=ic_amplitude)
    return ParticleEngine(cfg, pipeline=pipe), cfg


def render_field(eng, res, dims, slab=0.25):
    """A picture, at a resolution chosen for looking rather than for measuring.

    Measurement and rendering pull in opposite directions. `matched_res` wants ~1 particle
    per cell so connectivity statistics mean something; an image wants MANY particles per
    pixel or it is just shot noise. Sharing one grid between them forces a bad compromise,
    so they get separate ones — and the metrics reported alongside always come from the
    matched grid, never from this.

    In 3D a single mid-plane slice is one cell thick and throws away almost every particle.
    Projecting a slab (summing over a fraction `slab` of the depth) is what real cosmic-web
    figures do: it accumulates enough particles to show the filaments and keeps the topology
    legible, at the cost of some superposition along the line of sight.

    Raising the resolution alone does NOT buy detail — the same sampling limit that produced
    the percolation artifact applies to pixels. Past ~1 particle per pixel a finer histogram
    just resolves shot noise. So this smooths on deposit: a Gaussian kernel turns the
    histogram into a density ESTIMATE, which is what makes a smooth image from a modest
    particle count. Standard practice (CIC/TSC/SPH deposit); it is a rendering choice and
    never touches the measured numbers.
    """
    F = eng.density_field(res).cpu().numpy().astype(float)
    if F.ndim > 2:
        depth = max(1, int(res * slab))
        lo = (res - depth) // 2
        F = F[lo:lo + depth].sum(axis=0)

    # Kernel width set by how far the sampling falls short of one particle per pixel.
    n = int(eng.config.n * (slab if eng.state.pos.shape[1] > 2 else 1.0))
    per_px = max(n / max(F.size, 1), 1e-9)
    sigma = float(np.clip((1.0 / per_px) ** 0.5 * 0.6, 0.0, 4.0))
    if sigma > 0.3:
        try:
            from scipy.ndimage import gaussian_filter
            F = gaussian_filter(F, sigma=sigma, mode="wrap")
        except ImportError:
            pass
    return F


def evolve(eng, steps, res, every, render_res=None, dims=2):
    """Yield (tick, render_image, metrics). Metrics from the matched grid, image from its own."""
    for t in range(1, steps + 1):
        eng.tick()
        if t % every == 0 or t == steps:
            F = eng.density_field(res).cpu().numpy().astype(float)
            img = render_field(eng, render_res, dims) if render_res else slice_of(F)
            yield t, img, web_metrics(F)


def cmd_run(a):
    eng, cfg = build(a.dims, a.n, a.box, a.r0, a.g, a.sec_balance, a.seed, a.convention,
                     a.ic, a.ic_index, a.ic_amplitude)
    res = a.res or matched_res(a.n, a.dims)
    print(f"  {a.n} particles, {a.dims}D, box {a.box}, r0 {a.r0}, g {a.g}, "
          f"sec {a.sec_balance}, {a.convention} convention, {a.ic} IC, device {eng.device}")
    rres = a.render_res or (res * 4 if a.dims == 2 else res * 3)
    print(f"  metrics grid {res}^{a.dims} = {res**a.dims} cells, {a.n/res**a.dims:.2f} particles/cell   |   render grid {rres}^{a.dims}")
    print(f"  {'tick':>7} {'void':>7} {'cv':>7} {'perc':>7} {'occ':>7} {'web':>6}")

    frames, trace = [], []
    for t, F, w in evolve(eng, a.steps, res, a.every, rres, a.dims):
        frames.append(F)
        trace.append({"tick": t, **{k: (bool(v) if isinstance(v, bool) else float(v))
                                    for k, v in w.items()}})
        print(f"  {t:>7} {w['void']:>7.3f} {w['cv']:>7.3f} {w['percolation']:>7.3f} "
              f"{w['occupancy']:>7.3f} {str(w['is_web']):>6}")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Display OVERDENSITY rho/rho_bar, not raw counts. The render kernel changes absolute
    # values (it spreads mass), so a colour scale keyed to raw density silently rescales
    # whenever smoothing changes. Overdensity is invariant to that and is the physically
    # meaningful quantity anyway.
    def od(F):
        m = float(np.mean(F))
        return np.log1p(F / m) if m > 0 else np.zeros_like(F)

    frames = [od(F) for F in frames]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                  gridspec_kw={"width_ratios": [1, 1.05]})
    vmax = float(np.percentile(frames[-1], 99.0)) or 1.0
    im = ax.imshow(frames[0].T, origin="lower", cmap="magma",
                   interpolation="bilinear", vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ttl = ax.set_title("")

    ts = [r["tick"] for r in trace]
    ax2.plot(ts, [r["cv"] for r in trace], lw=2, label="density CV (contrast)")
    ax2.plot(ts, [r["percolation"] for r in trace], lw=2, label="percolation (connectivity)")
    ax2.plot(ts, [r["void"] for r in trace], lw=2, ls="--", label="void fraction")
    ax2.set_xlabel("tick"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    ax2.set_title("structure metrics", fontsize=10)
    mark = ax2.axvline(ts[0], color="k", lw=1)

    def draw(i):
        im.set_data(frames[i].T)
        w = trace[i]
        ttl.set_text(f"t={w['tick']}   void {w['void']:.2f}   cv {w['cv']:.2f}   "
                     f"perc {w['percolation']:.3f}")
        mark.set_xdata([w["tick"], w["tick"]])
        return im, ttl, mark

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=120, blit=False)
    gif = OUT / f"worldmodel_{a.dims}d_{stamp}.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=8))
    draw(len(frames) - 1)                       # a still of the END state, not frame one
    fig.savefig(OUT / f"worldmodel_{a.dims}d_{stamp}_final.png", dpi=115, bbox_inches="tight")
    plt.close(fig)

    (OUT / f"worldmodel_{a.dims}d_{stamp}.json").write_text(json.dumps(
        {"config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")
                    and not hasattr(v, "__dict__")},
         "convention": a.convention, "res": res, "trace": trace}, indent=2), encoding="utf-8")
    print(f"\n  wrote {gif.relative_to(HERE.parents[1]).as_posix()}")
    return 0


def cmd_sweep(a):
    res = a.res or matched_res(a.n, a.dims)
    print(f"  sweeping {a.param} over {a.values}   ({a.dims}D, {a.n} particles, "
          f"{a.steps} steps, {a.convention})")
    print(f"  {a.param:>12} {'void':>7} {'cv':>7} {'perc':>7} {'occ':>7} {'web':>6}")

    panels, rows = [], []
    for v in a.values:
        kw = dict(dims=a.dims, n=a.n, box=a.box, r0=a.r0, g=a.g,
                  sec=a.sec_balance, seed=a.seed, convention=a.convention,
                  ic=a.ic, ic_index=a.ic_index, ic_amplitude=a.ic_amplitude)
        kw["sec" if a.param == "sec_balance" else a.param] = v
        eng, _ = build(**kw)
        F, w = None, None
        rres = a.render_res or (res * 4 if a.dims == 2 else res * 3)
        for _, F, w in evolve(eng, a.steps, res, a.steps, rres, a.dims):
            pass
        panels.append((v, F, w))
        rows.append({a.param: v, **{k: (bool(x) if isinstance(x, bool) else float(x))
                                    for k, x in w.items()}})
        print(f"  {v:>12.4g} {w['void']:>7.3f} {w['cv']:>7.3f} {w['percolation']:>7.3f} "
              f"{w['occupancy']:>7.3f} {str(w['is_web']):>6}")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.3))
    for ax, (v, F, w) in zip(np.atleast_1d(axes), panels):
        m = float(np.mean(F))
        ax.imshow((np.log1p(F / m) if m > 0 else F).T, origin="lower", cmap="magma",
                  interpolation="bilinear", aspect="equal")
        ax.set_title(f"{a.param} = {v:g}\nperc {w['percolation']:.3f}  cv {w['cv']:.2f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{a.param} sweep — {a.dims}D, {a.steps} steps", fontsize=11)
    fig.tight_layout()
    png = OUT / f"sweep_{a.param}_{stamp}.png"
    fig.savefig(png, dpi=115, bbox_inches="tight")
    (OUT / f"sweep_{a.param}_{stamp}.json").write_text(
        json.dumps({"param": a.param, "values": a.values, "dims": a.dims,
                    "steps": a.steps, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote {png.relative_to(HERE.parents[1]).as_posix()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the world model and watch structure form.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dims", type=int, default=2, choices=(2, 3))
        p.add_argument("-n", type=int, default=4000)
        p.add_argument("--box", type=float, default=120.0)
        p.add_argument("--r0", type=float, default=5.0)
        p.add_argument("--g", type=float, default=0.8)
        p.add_argument("--sec-balance", dest="sec_balance", type=float, default=0.6)
        p.add_argument("--steps", type=int, default=800)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--res", type=int, default=None,
                       help="metrics grid; default ~1 particle/cell (see matched_res)")
        p.add_argument("--render-res", dest="render_res", type=int, default=None,
                       help="image grid; independent of the metrics grid")
        p.add_argument("--convention", choices=("attractive", "exp11"), default="attractive")
        p.add_argument("--ic", choices=("zeldovich", "lattice"), default="zeldovich",
                       help="zeldovich = correlated (gives a web); lattice = exp_09/exp_11")
        p.add_argument("--ic-index", dest="ic_index", type=float, default=-1.727)
        p.add_argument("--ic-amplitude", dest="ic_amplitude", type=float, default=2.0)

    r = sub.add_parser("run", help="evolve and render an animation")
    common(r); r.add_argument("--every", type=int, default=25)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("sweep", help="one panel per parameter value")
    common(s)
    s.add_argument("--param", required=True,
                   choices=("sec_balance", "g", "r0", "n", "box", "ic_index", "ic_amplitude"))
    s.add_argument("--values", type=float, nargs="+", required=True)
    s.set_defaults(fn=cmd_sweep)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
