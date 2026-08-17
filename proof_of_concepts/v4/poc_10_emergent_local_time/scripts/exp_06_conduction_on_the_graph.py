#!/usr/bin/env python3
"""Does the clock field conduct along the web? — asked on a graph, not a ball.

exp_03 answered "no" and the test could not have answered otherwise. It coupled particles by
DISTANCE BALL, which cannot see connectivity by construction: a ball around a point in a dense
blob contains more particles than a ball on a filament, so "how much matter is nearby" and "how
it is connected" are entangled in the very definition of the operator. Worse, more neighbours
means the relaxation divides a signal among more partners, so amplitude falls with density
whether or not filaments channel anything. The null was structural, not physical.

A k-NEAREST-NEIGHBOUR graph separates them. Every particle gets exactly k links whether it sits
in a node, on a filament or alone in a void, so a dense region earns NO extra conductance for
being dense. Degree dilution is removed by construction rather than corrected for, and the web
and the uniform control have the same degree distribution. Any difference that survives is
topology.

Two measurements, the first needing no dynamics at all:

  A. GEODESIC RATIO — graph hops vs Euclidean distance. If filaments are connection paths, two
     particles on the same filament should be close in graph distance relative to their
     separation, while two separated by a void should be far. Pure connectivity, no diffusion,
     no threshold, nothing to confound.

  B. CONDUCTION — the unnormalised (conduction) Laplacian on that graph, arrival time measured
     at fixed EUCLIDEAN distance from a source. On a k-NN graph this is the question exp_03
     meant to ask.

Both against a uniform control at the same n, box and k. The control must come back flat; if it
does not, the measurement is bad rather than the physics interesting.

    python exp_06_conduction_on_the_graph.py [--k 6] [--steps 180] [--seeds 42 7 101]
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


def knn_graph(state, k):
    """Symmetric k-nearest-neighbour adjacency, minimum-image.

    Symmetrising (A or A^T) lets degree exceed k slightly — a particle chosen by many others
    keeps those links — so the residual degree spread is reported rather than assumed away.
    The point is that it no longer scales with local density the way a ball does.
    """
    r, _, _ = pairwise(state, 1.0)
    r.fill_diagonal_(float("inf"))
    idx = r.topk(k, dim=1, largest=False).indices
    n = state.n
    A = torch.zeros(n, n, dtype=torch.bool, device=r.device)
    A.scatter_(1, idx, True)
    return A | A.T


def local_density(state, r0):
    r, _, _ = pairwise(state, 1.0)
    near = (r < r0)
    near.fill_diagonal_(False)
    return near.sum(dim=1).float()


def geodesic_length(A, state, src, box):
    """Shortest PHYSICAL path length along the graph — not hop count.

    Hop count is confounded on a k-NN graph and using it produced a confident wrong answer
    first time round. Edge length scales with local spacing: in a dense filament the k
    nearest neighbours are physically close, so a hop covers little ground, while in a void
    hops are long. Crossing a fixed Euclidean distance through dense material therefore takes
    MORE hops for purely geometric reasons, which fakes exactly the anti-channelling signal
    it was meant to test for.

    Summing physical edge lengths removes it. The geodesic RATIO (path length / Euclidean
    separation) is then dimensionless and scale-free: ~1 means the graph offers a near-direct
    route, >1 means the route detours. If filaments are connection paths, particles on one
    should have ratios near 1 while particles across a void should not.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    d = state.pos.unsqueeze(1) - state.pos.unsqueeze(0)
    d = (d + box / 2) % box - box / 2
    L = d.norm(dim=-1)
    W = torch.where(A, L, torch.zeros_like(L)).cpu().numpy()
    return dijkstra(csr_matrix(W), directed=False, indices=src), L


def conductance_graph(A, L, weighted=True):
    """Edge conductance for the resistor network.

    weighted=True  -> conductance ~ 1/length, the physical model (resistors in series).
    weighted=False -> every edge conducts equally: TOPOLOGY ONLY.

    The unweighted case is the decisive control. With 1/length weights a dense region gets
    high conductance automatically because its edges are short, which is the same geometric
    artifact that made hop-counting fail. Setting every edge to 1 removes length from the
    problem entirely on a graph whose degree is already bounded, so anything that survives is
    connectivity: more parallel routes between the same points.
    """
    if not weighted:
        return A.float()
    return torch.where(A, 1.0 / L.clamp(min=1e-6), torch.zeros_like(L))


def conduct(W, src, steps=None, nu=None, thresh=None):
    """Conduction potential from a unit current injected at src — EFFECTIVE RESISTANCE.

    Explicit time-stepping was the wrong solver and returned all-NaN. Conductance weights
    (~1/length) make dense-region degrees enormous, so the stable step nu/deg_max collapses
    and nothing propagates within any reasonable number of iterations. The network is stiff
    precisely BECAUSE it has the structure being measured.

    Solving the steady state instead removes the stiffness, the arrival threshold and the
    step-count parameter in one go. Inject 1 A at the source, drain it uniformly, solve
    L x = b. The potential x is the effective-resistance distance from the source: LOW where
    the network conducts well. It is the exact quantity the time-stepping was approximating.

    L is singular (constant nullspace), so one node is grounded and the reduced system solved.
    """
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import spsolve
    Wn = W.cpu().numpy()
    n = Wn.shape[0]
    L = csr_matrix(diags(Wn.sum(1)) - Wn)
    b = np.full(n, -1.0 / n)
    b[src] += 1.0
    keep = np.arange(n) != 0                      # ground node 0
    x = np.zeros(n)
    x[keep] = spsolve(L[keep][:, keep].tocsc(), b[keep])
    return torch.from_numpy(x - x[src])           # 0 at source, rises with resistance


def binned(x, y, edges):
    """Mean of y in bins of x — returns (centres, means, counts)."""
    c, m, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        if s.sum() < 20:
            continue
        c.append(0.5 * (lo + hi)); m.append(float(np.mean(y[s]))); n.append(int(s.sum()))
    return np.array(c), np.array(m), np.array(n)


def build(kind, a, seed):
    cfg = ParticleConfig(n=a.n, box=a.box, r0=a.r0, g=a.g, sec_balance=a.sec_balance,
                         dims=a.dims, seed=seed, entropy_init=0.1,
                         time_mode="potential", time_viscosity=a.viscosity)
    eng = ParticleEngine(cfg, pipeline=EXP11_TIME)
    if kind == "web":
        for _ in range(a.steps):
            eng.tick()
    else:
        g = torch.Generator(device="cpu").manual_seed(seed + 5150)
        pos = (torch.rand(a.n, a.dims, generator=g) * a.box).to(eng.state.pos.device)
        eng.state = eng.state.replace(pos=pos, vel=torch.zeros_like(pos))
    return eng


def run(kind, a, seed):
    eng = build(kind, a, seed)
    res = max(8, int(round(a.n ** (1.0 / a.dims))))
    web = web_metrics(eng.density_field(res).cpu().numpy().astype(float))

    A = knn_graph(eng.state, a.k)
    deg = A.sum(1).float()
    dens = local_density(eng.state, a.r0)
    src = int(torch.argmax(dens).item())

    d = eng.state.pos - eng.state.pos[src]
    d = (d + a.box / 2) % a.box - a.box / 2
    euc = d.norm(dim=-1).cpu().numpy()

    glen, L = geodesic_length(A, eng.state, src, a.box)
    Wc = conductance_graph(A, L, weighted=not a.unweighted)
    arr = conduct(Wc, src).cpu().numpy()          # effective-resistance distance
    hops = glen                                    # PHYSICAL path length, not hop count
    dn = dens.cpu().numpy()
    ok = np.isfinite(hops) & (euc > 0)
    arr = np.where(np.isfinite(arr), arr, np.nan)

    # A. Geodesic ratio: PATH LENGTH per unit Euclidean distance, dimensionless. ~1 means a
    #    near-direct route through the graph; >1 means a detour. If filaments are paths, dense
    #    particles should have LOWER ratios at fixed Euclidean separation.
    ratio = hops[ok] / np.maximum(euc[ok], 1e-9)

    # Correlate against density WITHIN fixed-distance shells, so "further away needs more hops"
    # cannot produce the signal.
    edges = np.linspace(a.r0 * 0.5, a.r0 * 4.0, 7)
    geo_c, arr_c = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (euc[ok] >= lo) & (euc[ok] < hi)
        if s.sum() < 30:
            continue
        dd = dn[ok][s]
        if dd.std() < 1e-9:
            continue
        if np.std(hops[ok][s]) > 1e-9:
            geo_c.append(float(np.corrcoef(hops[ok][s], dd)[0, 1]))
        av = arr[ok][s]
        fin = np.isfinite(av)
        if fin.sum() > 20 and np.std(av[fin]) > 1e-9 and np.std(dd[fin]) > 1e-9:
            arr_c.append(float(np.corrcoef(av[fin], dd[fin])[0, 1]))

    return {
        "kind": kind, "seed": seed,
        "percolation": float(web["percolation"]), "is_web": bool(web["is_web"]),
        "degree_mean": float(deg.mean()), "degree_max": float(deg.max()),
        "degree_cv": float((deg.std() / deg.mean()).item()),
        # Sign-flipped so POSITIVE always means "filaments channel".
        "geodesic_vs_density": -float(np.mean(geo_c)) if geo_c else float("nan"),
        "arrival_vs_density": -float(np.mean(arr_c)) if arr_c else float("nan"),
        "reached_frac": float(np.isfinite(hops).mean()),
        "_plot": (euc[ok], hops[ok], arr[ok], dn[ok]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=3, choices=(2, 3))
    ap.add_argument("-n", type=int, default=6000)
    ap.add_argument("--box", type=float, default=60.0)
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=1.5)
    ap.add_argument("--sec-balance", dest="sec_balance", type=float, default=0.65334)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--k", type=int, default=6, help="neighbours per particle; bounded degree")
    ap.add_argument("--unweighted", action="store_true",
                    help="every edge conducts equally: topology only, no edge length")
    ap.add_argument("--diffuse", type=int, default=400)
    ap.add_argument("--viscosity", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 101])
    a = ap.parse_args()

    print(f"  {a.n} particles, {a.dims}D, web epoch {a.steps} steps, k-NN graph k={a.k}")
    print(f"  bounded degree removes the density/connectivity entanglement by construction\n")
    print(f"  {'kind':>8} {'seed':>5} {'perc':>6} {'<deg>':>6} {'degCV':>6} {'reach':>6} | "
          f"{'GEODESIC':>9} {'RESISTANCE':>11}")

    rows = []
    for seed in a.seeds:
        for kind in ("web", "uniform"):
            r = run(kind, a, seed)
            rows.append(r)
            print(f"  {kind:>8} {seed:>5} {r['percolation']:>6.3f} {r['degree_mean']:>6.2f} "
                  f"{r['degree_cv']:>6.3f} {r['reached_frac']:>6.2f} | "
                  f"{r['geodesic_vs_density']:>+9.3f} {r['arrival_vs_density']:>+11.3f}")

    def arr_of(kind, key):
        return np.array([r[key] for r in rows if r["kind"] == kind])

    print()
    for key, label in (("geodesic_vs_density", "GEODESIC (hops vs density)"),
                       ("arrival_vs_density", "RESISTANCE (eff. resistance vs density)")):
        w, u = arr_of("web", key), arr_of("uniform", key)
        se = np.sqrt(w.var(ddof=1) / len(w) + u.var(ddof=1) / len(u)) if len(w) > 1 else 0
        sig = (w.mean() - u.mean()) / se if se > 0 else 0
        print(f"  {label:<32} web {w.mean():+.3f} +-{w.std():.3f}   "
              f"uniform {u.mean():+.3f} +-{u.std():.3f}   diff {w.mean()-u.mean():+.3f} "
              f"({sig:+.2f} sigma)")

    print(f"\n  POSITIVE = dense particles are BETTER connected than their distance implies")
    print(f"           = filaments are conduction paths.")
    print(f"  Degree CV is reported because the whole point is that it does NOT track density;")
    print(f"  compare against exp_03's distance ball, where <nbrs> was 798 in the web vs 116")
    print(f"  in the control at the same n and box.")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for ax, kind in zip(axes[:2], ("web", "uniform")):
        r = next(x for x in rows if x["kind"] == kind)
        euc, hops, arr, dn = r["_plot"]
        sc = ax.scatter(euc / a.r0, hops, c=np.log10(np.maximum(dn, 1)), s=5, alpha=0.3,
                        cmap="viridis", edgecolors="none")
        ax.set_xlabel("Euclidean distance from source (r/r0)")
        ax.set_ylabel("geodesic path length")
        ax.set_title(f"{kind} — geodesic {r['geodesic_vs_density']:+.3f}", fontsize=10)
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, fraction=0.046, label="log10 local density")
    for kind, c in (("web", "tab:red"), ("uniform", "tab:blue")):
        r = next(x for x in rows if x["kind"] == kind)
        euc, hops, arr, dn = r["_plot"]
        e = np.linspace(0, euc.max(), 15)
        cx, cy, _ = binned(euc, hops, e)
        axes[2].plot(cx / a.r0, cy, marker="o", color=c, label=kind)
    axes[2].set_xlabel("Euclidean distance (r/r0)"); axes[2].set_ylabel("mean geodesic length")
    axes[2].set_title("how far in the graph, per unit space", fontsize=10)
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle(f"conduction on a k-NN graph (k={a.k}) — bounded degree, so any difference "
                 f"is topology", fontsize=12)
    fig.tight_layout()
    p = OUT / f"exp_06_conduction_graph_{a.dims}d_{stamp}.png"
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    for r in rows:
        r.pop("_plot", None)
    (OUT / f"exp_06_conduction_graph_{stamp}.json").write_text(
        json.dumps({"config": vars(a), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
