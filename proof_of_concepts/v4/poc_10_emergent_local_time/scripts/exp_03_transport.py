#!/usr/bin/env python3
"""Does time flow THROUGH the filaments?

The claim: the clock field propagates along the cosmic web the way a current runs through a
conductor — filaments channel it, voids resist it. That is a transport statement, distinct
from anything measured so far. `dilation.py` showed the clock field carries non-local
structure and that viscosity is the channel; it did not show the transport is STRUCTURED.

Design, kept deliberately narrow so the answer means one thing:

  1. Evolve to the web epoch, then FREEZE the positions. Nothing moves after this, so what
     is measured is transport and not advection or further collapse.
  2. Set tau = 1 everywhere, then raise it at a single source particle in a dense node.
  3. Iterate ONLY the viscosity step — the neighbour diffusion from LocalTime.
  4. At fixed EUCLIDEAN distance from the source, ask whether the perturbation is larger
     where the local density is higher.

That last step is the whole experiment. Distance is held fixed, so a plain "it spreads"
cannot produce the signal; only a preference for dense paths can.

    corr(perturbation, density | distance shell) > 0   ->  filaments channel the clock field
                                              ~ 0   ->  transport is blind to the web

**The control is a uniform particle distribution** at the same n, box and r0 — same diffusion
operator, no web. Clustering raises the mean neighbour count, so the web diffuses faster in
absolute terms whatever happens; that is why the observable is a WITHIN-SHELL correlation and
not a raw speed. The control must return ~0, and if it does not, the measurement is bad
rather than the physics interesting.

    python time_transport.py [--steps 180] [--diffuse 40] [--viscosity 0.3]
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

from particles import EXP11_TIME, ParticleConfig, ParticleEngine, pairwise  # noqa: E402
from structure import web_metrics  # noqa: E402

OUT = HERE.parent / "results"


def neighbour_matrix(state, r0):
    """Boolean adjacency inside r0, and each particle's local density (neighbour count)."""
    r, _, _ = pairwise(state, 1.0)
    near = (r < r0)
    near.fill_diagonal_(False)
    return near, near.sum(dim=1).float()


def diffuse(pert, near, nu, steps, thresh=1e-6):
    """Iterate the LocalTime viscosity step on a perturbation, positions frozen.

    Returns (final amplitude, ARRIVAL TIME per particle).

    Amplitude alone is the wrong observable and using it produced a confident wrong answer
    first time round. The relaxation shares a particle's value among its neighbours, so in a
    dense filament the same signal is divided over many more particles and the per-particle
    amplitude FALLS with density — regardless of whether filaments channel anything. That is
    degree dilution, and it fakes exactly the negative correlation it is meant to test for.

    Arrival time is dilution-insensitive: it asks WHEN the front first reaches a particle,
    not how much is left after sharing. If filaments channel the clock field, particles at
    the same Euclidean distance should be reached SOONER along dense paths — so the
    signature is a NEGATIVE correlation between arrival time and local density.
    """
    w = near.float()
    deg = w.sum(dim=1).clamp(min=1.0)
    p = pert.clone()
    arrival = torch.full_like(p, float("nan"))
    for t in range(1, steps + 1):
        p = p + nu * ((w @ p) / deg - p)
        newly = torch.isnan(arrival) & (p > thresh)
        arrival[newly] = float(t)
    return p, arrival


def shell_correlations(pert, dist, dens, r0, nshell=6):
    """corr(perturbation, density) inside fixed-distance shells — the actual observable."""
    out = []
    edges = np.linspace(r0 * 0.5, r0 * 4.0, nshell + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < 30:
            continue
        p, d = pert[m], dens[m]
        if p.std() < 1e-12 or d.std() < 1e-12:
            continue
        out.append({"r_lo": float(lo), "r_hi": float(hi), "n": int(m.sum()),
                    "corr": float(np.corrcoef(p, d)[0, 1])})
    return out


def run(kind, a, seed):
    """kind: 'web' (evolved, clustered) or 'uniform' (control, unstructured)."""
    cfg = ParticleConfig(n=a.n, box=a.box, r0=a.r0, g=a.g, sec_balance=a.sec_balance,
                         dims=a.dims, seed=seed, entropy_init=0.1,
                         time_mode="potential", time_viscosity=a.viscosity)
    eng = ParticleEngine(cfg, pipeline=EXP11_TIME)
    if kind == "web":
        for _ in range(a.steps):
            eng.tick()
    else:
        # Same n, box, r0 — positions uniform, so the operator is identical and only the
        # STRUCTURE differs. This is the control that must come back ~0.
        g = torch.Generator(device="cpu").manual_seed(seed + 5150)
        pos = (torch.rand(a.n, a.dims, generator=g) * a.box).to(eng.state.pos.device)
        eng.state = eng.state.replace(pos=pos, vel=torch.zeros_like(pos))

    res = max(8, int(round(a.n ** (1.0 / a.dims))))
    rho = eng.density_field(res).cpu().numpy().astype(float)
    web = web_metrics(rho)

    # The CLOCK coupling radius is separate from the FORCE radius. Tying them together
    # (both r0) makes the neighbour graph ~2 hops wide at exp_11's r0/box = 1/6, so the
    # clock field equilibrates instantly and transport cannot exist. Local time only has
    # propagation structure when its coupling is short-ranged against the web.
    rc = a.diffuse_radius or a.r0
    near, _ = neighbour_matrix(eng.state, rc)
    _, dens = neighbour_matrix(eng.state, a.r0)          # density always at the force scale
    dens_np = dens.cpu().numpy()

    # Source: the densest particle — a node of the web where one exists.
    src = int(torch.argmax(dens).item())
    pert = torch.zeros(a.n, device=eng.state.pos.device)
    pert[src] = 1.0

    p_t, arr_t = diffuse(pert, near, a.viscosity, a.diffuse)
    p, arr = p_t.cpu().numpy(), arr_t.cpu().numpy()
    p[src] = np.nan; arr[src] = np.nan                 # the source itself is not transport

    d = eng.state.pos - eng.state.pos[src]
    d = (d + a.box / 2) % a.box - a.box / 2            # minimum image
    dist = d.norm(dim=-1).cpu().numpy()

    ok = np.isfinite(arr) & (dist > 0)
    # NEGATIVE corr(arrival, density) == dense paths reached sooner == channelling.
    # Sign-flipped on report so that POSITIVE always means "filaments channel".
    shells = shell_correlations(-arr[ok], dist[ok], dens_np[ok], a.r0)
    overall = float(np.mean([s["corr"] for s in shells])) if shells else float("nan")
    amp_ok = np.isfinite(p) & (dist > 0)
    amp_shells = shell_correlations(p[amp_ok], dist[amp_ok], dens_np[amp_ok], a.r0)
    amp_overall = float(np.mean([s["corr"] for s in amp_shells])) if amp_shells else float("nan")
    reach = float(np.percentile(dist[ok][p[ok] > p[ok].max() * 0.01], 95)) if (p[ok] > 0).any() \
        else float("nan")
    return {"kind": kind, "seed": seed, "shells": shells, "mean_shell_corr": overall,
            "amplitude_shell_corr": amp_overall, "arrived_frac": float(np.isfinite(arr).mean()),
            "reach_95": reach, "percolation": float(web["percolation"]),
            "is_web": bool(web["is_web"]), "mean_neighbours": float(dens_np.mean()),
            "_plot": (dist[ok], arr[ok], dens_np[ok])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=3, choices=(2, 3))
    ap.add_argument("-n", type=int, default=6000)
    ap.add_argument("--box", type=float, default=60.0)
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=1.5)
    ap.add_argument("--sec-balance", dest="sec_balance", type=float, default=0.65334)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--diffuse", type=int, default=40)
    ap.add_argument("--viscosity", type=float, default=0.3)
    ap.add_argument("--diffuse-radius", dest="diffuse_radius", type=float,
                    default=None, help="clock coupling radius; default = r0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 101])
    a = ap.parse_args()

    print(f"  {a.n} particles, {a.dims}D, web epoch {a.steps} steps, then positions FROZEN")
    print(f"  {a.diffuse} diffusion iterations at nu = {a.viscosity}, "
          f"clock radius {a.diffuse_radius or a.r0} vs force radius {a.r0}")
    print(f"\n  {'kind':>8} {'seed':>5} {'perc':>6} {'<nbrs>':>7} {'reach':>7} "
          f"{'MEAN SHELL CORR':>16}")

    rows = []
    for seed in a.seeds:
        for kind in ("web", "uniform"):
            r = run(kind, a, seed)
            rows.append(r)
            print(f"  {kind:>8} {seed:>5} {r['percolation']:>6.3f} "
                  f"{r['mean_neighbours']:>7.1f} {r['arrived_frac']:>8.2f} "
                  f"{r['amplitude_shell_corr']:>+16.3f} {r['mean_shell_corr']:>+13.3f}")

    web = np.array([r["mean_shell_corr"] for r in rows if r["kind"] == "web"])
    uni = np.array([r["mean_shell_corr"] for r in rows if r["kind"] == "uniform"])
    se = np.sqrt(web.var(ddof=1) / len(web) + uni.var(ddof=1) / len(uni)) if len(web) > 1 else 0
    print(f"\n  web      {web.mean():+.3f} +/- {web.std():.3f}")
    print(f"  uniform  {uni.mean():+.3f} +/- {uni.std():.3f}   <- CONTROL, must be ~0")
    print(f"  difference {web.mean()-uni.mean():+.3f}"
          + (f"   ({(web.mean()-uni.mean())/se:+.2f} sigma)" if se > 0 else ""))
    print(f"\n  Reading: web >> uniform => the filaments CHANNEL the clock field.")
    print(f"           web ~ uniform  => transport is blind to the web.")

    # Per-shell detail: channeling should persist out to several r0, not just the first shell.
    print(f"\n  per-shell correlation (web, seed {a.seeds[0]}):")
    for s in rows[0]["shells"]:
        print(f"     r/r0 {s['r_lo']/a.r0:.1f}-{s['r_hi']/a.r0:.1f}  n {s['n']:>5}  "
              f"corr {s['corr']:+.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for ax, kind in zip(axes[:2], ("web", "uniform")):
        r = next(x for x in rows if x["kind"] == kind)
        dist, p, dens = r["_plot"]
        m = np.isfinite(p)
        sc = ax.scatter(dist[m] / a.r0, p[m], c=dens[m], s=6, alpha=0.35,
                        cmap="viridis", edgecolors="none")
        ax.set_xlabel("distance from source  (r/r0)")
        ax.set_ylabel("arrival time  (diffusion steps)")
        ax.set_title(f"{kind}  —  mean shell corr {r['mean_shell_corr']:+.3f}", fontsize=10)
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, fraction=0.046, label="local density")

    for kind, c in (("web", "tab:red"), ("uniform", "tab:blue")):
        rs = [r for r in rows if r["kind"] == kind]
        xs = [s["r_lo"] / a.r0 for s in rs[0]["shells"]]
        ys = np.array([[s["corr"] for s in r["shells"]] for r in rs
                       if len(r["shells"]) == len(rs[0]["shells"])])
        if len(ys):
            axes[2].errorbar(xs, ys.mean(0), yerr=ys.std(0), marker="o", color=c, label=kind,
                             capsize=3)
    axes[2].axhline(0, color="k", lw=0.8, ls=":")
    axes[2].set_xlabel("shell inner radius  (r/r0)")
    axes[2].set_ylabel("corr(perturbation, density)")
    axes[2].set_title("does the clock field prefer dense paths?", fontsize=10)
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle("time transport through the cosmic web — positions frozen, "
                 "viscosity only", fontsize=12)
    fig.tight_layout()
    path = OUT / f"time_transport_{a.dims}d_{stamp}.png"
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)

    for r in rows:
        r.pop("_plot", None)
    (OUT / f"time_transport_{stamp}.json").write_text(
        json.dumps({"config": vars(a), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
