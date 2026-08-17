#!/usr/bin/env python3
"""Emergent local time on the web — does the clock field trace the filaments?

The premise under test: a collapse event is a tick, so the local framerate is set by the
neighbourhood's remaining collapse budget. Mass is spent potential, so mass-dense regions
tick slowly and gravitational time dilation falls out of PAC bookkeeping instead of being
inserted. Nothing in the substrate knows about GR.

Two candidate framerate sources make DIFFERENT topological predictions, which is why both
are implemented rather than one being argued for:

    potential   tau ~ 1/(1 + kappa*delta)         slow-time set traces FILAMENTS AND NODES
    rate        tau ~ 1/(1 + kappa*|d delta/dt|)  slow-time set traces INFALL FRONTS

So the discriminator is not "is there dilation" — both give that — but *where the slow
surfaces sit relative to the mass*. Reported as the tau-delta correlation and, more sharply,
as how much of the slow-time set sits ON the overdense set versus on its boundary.

    python time_flow.py                       # 3D, exp_11 config, both modes
    python time_flow.py --mode potential --steps 240
    python time_flow.py --dims 2 -n 4000

Writes a four-panel figure per mode: the web, the clock field, the flow, and the correlation.
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
import torch  # noqa: E402

from particles import CANONICAL_TIME, EXP11_TIME, ParticleConfig, ParticleEngine  # noqa: E402
from structure import web_metrics  # noqa: E402
from worldmodel import matched_res  # noqa: E402

OUT = HERE.parent / "results"


def velocity_field(eng, res):
    """Mean velocity per cell, one component at a time."""
    s = eng.state
    return [eng.field_of(s.vel[:, ax].contiguous(), res).cpu().numpy() for ax in range(s.pos.shape[1])]


def flatten(F, slab=0.3):
    """3D -> 2D by averaging a central slab, ignoring empty cells."""
    if F.ndim == 2:
        return F
    depth = max(1, int(F.shape[0] * slab))
    lo = (F.shape[0] - depth) // 2
    sub = F[lo:lo + depth]
    with np.errstate(invalid="ignore"):
        return np.nanmean(sub, axis=0)


def boundary_of(mask):
    """Cells outside `mask` that touch it — the overdense set's boundary shell."""
    edge = np.zeros_like(mask)
    for ax in range(mask.ndim):
        for sh in (1, -1):
            edge |= np.roll(mask, sh, axis=ax)
    return edge & ~mask


def analyse(eng, res):
    """Where does the slow-time set sit relative to the mass?"""
    rho = eng.density_field(res).cpu().numpy().astype(float)
    tau = eng.field_of(eng.state.tau, res).cpu().numpy()
    ok = np.isfinite(tau)
    if ok.sum() < 10:
        return None
    delta = rho / rho.mean() - 1.0

    corr = float(np.corrcoef(delta[ok], tau[ok])[0, 1])
    dense = (rho > 2.0 * rho.mean())
    shell = boundary_of(dense) & ok
    slow = np.zeros_like(ok)
    slow[ok] = tau[ok] < np.nanpercentile(tau[ok], 25)      # the slowest quartile of clocks

    on_dense = float((slow & dense).sum() / max(slow.sum(), 1))
    on_shell = float((slow & shell).sum() / max(slow.sum(), 1))
    return {
        "tau_delta_corr": corr,
        "slow_on_filaments": on_dense,
        "slow_on_boundary": on_shell,
        "tau_in_dense": float(np.nanmean(tau[dense & ok])) if (dense & ok).any() else float("nan"),
        "tau_in_void": float(np.nanmean(tau[~dense & ok])),
        "tau_dispersion": float(eng.state.metrics.get("tau_dispersion", 0.0)),
        "proper_time_spread": float(eng.state.metrics.get("proper_time_spread", 0.0)),
    }


def figure(eng, res, rres, mode, stats, web, stamp, dims):
    rho_r = eng.density_field(rres).cpu().numpy().astype(float)
    tau_r = eng.field_of(eng.state.tau, rres).cpu().numpy()
    vfs = velocity_field(eng, rres)

    rho2, tau2 = flatten(rho_r), flatten(tau_r)
    vx, vy = flatten(vfs[-2]), flatten(vfs[-1])
    vx, vy = np.nan_to_num(vx), np.nan_to_num(vy)

    try:
        from scipy.ndimage import gaussian_filter
        sm = lambda A, s=1.4: gaussian_filter(np.nan_to_num(A, nan=float(np.nanmean(A))),
                                              s, mode="wrap")
    except ImportError:
        sm = lambda A, s=1.4: np.nan_to_num(A, nan=float(np.nanmean(A)))

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.0))

    # 1 — the web
    od = np.log1p(rho2 / max(rho2.mean(), 1e-9))
    axes[0].imshow(sm(od).T, origin="lower", cmap="magma", interpolation="bilinear",
                   vmin=0, vmax=np.percentile(od, 99))
    axes[0].set_title(f"matter — the web\nperc {web['percolation']:.3f}   void {web['void']:.2f}",
                      fontsize=10)

    # 2 — the clock field. Diverging around 1: blue slow, red fast.
    t2 = sm(tau2)
    lim = max(abs(np.nanpercentile(t2, 2) - 1), abs(np.nanpercentile(t2, 98) - 1), 1e-3)
    im = axes[1].imshow(t2.T, origin="lower", cmap="coolwarm", interpolation="bilinear",
                        vmin=1 - lim, vmax=1 + lim)
    axes[1].set_title(f"local time tau  ({mode})\nblue = slow clocks    "
                      f"dispersion {stats['tau_dispersion']:.3f}", fontsize=10)
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 3 — the flow, over the web, streamlines coloured by speed
    axes[2].imshow(sm(od).T, origin="lower", cmap="gray_r", interpolation="bilinear",
                   vmin=0, vmax=np.percentile(od, 99), alpha=0.55)
    g = np.arange(rres)
    spd = np.hypot(sm(vx, 1.8), sm(vy, 1.8))
    try:
        axes[2].streamplot(g, g, sm(vx, 1.8).T, sm(vy, 1.8).T, color=spd.T,
                           cmap="viridis", density=1.5, linewidth=1.0, arrowsize=0.8)
    except Exception:
        step = max(1, rres // 28)
        axes[2].quiver(g[::step], g[::step], sm(vx, 1.8).T[::step, ::step],
                       sm(vy, 1.8).T[::step, ::step], spd.T[::step, ::step], cmap="viridis")
    axes[2].set_xlim(0, rres - 1); axes[2].set_ylim(0, rres - 1)
    axes[2].set_title("flow — infall along the filaments", fontsize=10)

    for ax in axes[:3]:
        ax.set_xticks([]); ax.set_yticks([])

    # 4 — the discriminator
    rho_m = eng.density_field(res).cpu().numpy().astype(float)
    tau_m = eng.field_of(eng.state.tau, res).cpu().numpy()
    ok = np.isfinite(tau_m)
    d_m = (rho_m / rho_m.mean() - 1.0)[ok]
    axes[3].scatter(d_m, tau_m[ok], s=6, alpha=0.25, edgecolors="none")
    axes[3].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[3].set_xlabel("local overdensity  delta"); axes[3].set_ylabel("clock rate  tau")
    axes[3].set_xscale("symlog", linthresh=1.0)
    axes[3].grid(alpha=0.3)
    axes[3].set_title(
        f"corr(tau, delta) = {stats['tau_delta_corr']:+.3f}\n"
        f"slow clocks: {stats['slow_on_filaments']*100:.0f}% on filaments, "
        f"{stats['slow_on_boundary']*100:.0f}% on boundary", fontsize=10)

    fig.suptitle(f"emergent local time on the cosmic web — mode '{mode}', {dims}D", fontsize=12)
    fig.tight_layout()
    path = OUT / f"time_flow_{mode}_{dims}d_{stamp}.png"
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return path


def run_mode(a, mode):
    pipe = EXP11_TIME if a.convention == "exp11" else CANONICAL_TIME
    cfg = ParticleConfig(n=a.n, box=a.box, r0=a.r0, g=a.g, sec_balance=a.sec_balance,
                         dims=a.dims, seed=a.seed, entropy_init=0.1, ic=a.ic,
                         time_mode=mode, time_kappa=a.kappa, time_viscosity=a.viscosity)
    eng = ParticleEngine(cfg, pipeline=pipe)
    res = a.res or matched_res(a.n, a.dims)
    for _ in range(a.steps):
        eng.tick()

    rho = eng.density_field(res).cpu().numpy().astype(float)
    web = web_metrics(rho)
    if mode == "global":
        print(f"  {mode:>10}  perc {web['percolation']:.3f}  web {web['is_web']}   "
              f"(control: one clock everywhere)")
        return {"mode": mode, "web": {k: (bool(v) if isinstance(v, bool) else float(v))
                                      for k, v in web.items()}}
    st = analyse(eng, res)
    print(f"  {mode:>10}  perc {web['percolation']:.3f}  web {str(web['is_web']):>5}   "
          f"corr {st['tau_delta_corr']:+.3f}   tau dense {st['tau_in_dense']:.3f} vs "
          f"void {st['tau_in_void']:.3f}   slow: {st['slow_on_filaments']*100:.0f}% fil / "
          f"{st['slow_on_boundary']*100:.0f}% bnd")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    p = figure(eng, res, a.render_res or res * 3, mode, st, web, stamp, a.dims)
    print(f"              -> {p.name}")
    return {"mode": mode, "stats": st,
            "web": {k: (bool(v) if isinstance(v, bool) else float(v)) for k, v in web.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("global", "potential", "rate", "all"), default="all")
    ap.add_argument("--dims", type=int, default=3, choices=(2, 3))
    ap.add_argument("-n", type=int, default=8000)
    ap.add_argument("--box", type=float, default=60.0)
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=1.5)
    ap.add_argument("--sec-balance", dest="sec_balance", type=float, default=0.65334)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--render-res", dest="render_res", type=int, default=None)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--viscosity", type=float, default=0.3)
    ap.add_argument("--ic", choices=("lattice", "zeldovich"), default="lattice")
    ap.add_argument("--convention", choices=("attractive", "exp11"), default="exp11")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    modes = ["global", "potential", "rate"] if a.mode == "all" else [a.mode]
    print(f"  {a.n} particles, {a.dims}D, {a.steps} steps, kappa {a.kappa}, "
          f"viscosity {a.viscosity}, {a.ic} IC")
    rows = [run_mode(a, m) for m in modes]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (OUT / f"time_flow_{stamp}.json").write_text(json.dumps(
        {"config": vars(a), "results": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
