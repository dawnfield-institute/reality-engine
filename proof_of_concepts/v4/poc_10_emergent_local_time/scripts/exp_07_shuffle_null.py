#!/usr/bin/env python3
"""Does the CLOCK FIELD follow the web's topology? — answered with a shuffle null.

Two attempts at this question produced large, confident, meaningless numbers:

  ball vs k-NN coupling      confounded by KERNEL SIZE. tau's non-local content is a monotone,
                             roughly logarithmic function of coupling degree (7.6 -> +0.085,
                             58.8 -> +0.275, 237 -> +0.399), so comparing a ball to a graph
                             compares two smoothing radii, not two topologies.
  web vs uniform control     confounded by CONTROL-VARIABLE VARIANCE. Partial correlation is
                             not comparable across systems whose control variable has wildly
                             different spread: in a uniform box delta barely varies, so
                             regressing it out removes nothing, while in a web it dominates
                             tau and regressing it out strips the signal. The control scored
                             HIGHER at -9.31 sigma, which is an artifact of the statistic.

The method that works is the corpus's own, from `euclidean_distance_validation`
experiment_25 (the "R^2 = 1.0 controversy"), which faced the mirror-image accusation --
"your metrics correlate because they are redundant, not because of conservation" -- and
answered it by SHUFFLING: r fell 0.79 -> -0.29 and xi fell 0.87 -> 0.17 once structure was
removed, with a 100-trial random baseline and a z-score against it.

A DEGREE-PRESERVING EDGE SHUFFLE holds everything fixed except the wiring. Same particles,
same positions, same density field, same delta and its variance, same degree per node, same
edge count, same diffusion kernel size. Only WHICH particles are connected changes. So both
confounds above are impossible by construction, and no partial correlation is needed --
the shuffle IS the control.

    H0: tau's agreement with the long-range potential is unchanged when the web's topology is
        destroyed but every other property is preserved.

Rejecting H0 means the clock field is following the structure, not merely the matter.

    python exp_07_shuffle_null.py [--k 6] [--rewires 30] [--seeds 42 7 101]
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

from exp_02_dilation import potential  # noqa: E402
from particles import EXP11_TIME, ParticleConfig, ParticleEngine, pairwise  # noqa: E402
from structure import web_metrics  # noqa: E402

OUT = HERE.parent / "results"


def knn_edges(state, k):
    """Undirected edge list of the symmetric k-NN graph."""
    r, _, _ = pairwise(state, 1.0)
    r.fill_diagonal_(float("inf"))
    idx = r.topk(k, dim=1, largest=False).indices.cpu().numpy()
    n = idx.shape[0]
    e = {(min(i, j), max(i, j)) for i in range(n) for j in idx[i]}
    return np.array(sorted(e), dtype=np.int64)


def degree_preserving_shuffle(edges, n, rng, passes=8):
    """Double-edge swap: (a,b),(c,d) -> (a,d),(c,b). Degree is exactly preserved.

    This is the null model. It destroys WHICH particles are connected while leaving every
    node's number of connections untouched, so anything that depends on degree -- kernel
    size, diffusion rate, the variance of any per-particle quantity -- is identical between
    the real graph and the shuffled one.
    """
    E = edges.copy()
    m = len(E)
    seen = {(int(a), int(b)) for a, b in E}
    for _ in range(passes * m):
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if len({a, b, c, d}) < 4:
            continue
        na, nb = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if na in seen or nb in seen:
            continue
        seen.discard((min(a, b), max(a, b)))
        seen.discard((min(c, d), max(c, d)))
        seen.add(na); seen.add(nb)
        E[i] = na; E[j] = nb
    return E


def adjacency(edges, n, device):
    A = torch.zeros(n, n, dtype=torch.bool, device=device)
    i = torch.from_numpy(edges[:, 0]).to(device)
    j = torch.from_numpy(edges[:, 1]).to(device)
    A[i, j] = True
    A[j, i] = True
    return A


def diffuse_tau(A, delta, kappa, nu, steps, floor=0.05):
    """Run LocalTime's tau update on a FIXED graph, positions frozen.

    Same arithmetic as the operator: tau from the local collapse budget, then nu * grad^2
    relaxation toward the neighbour mean, then renormalise to mean 1.
    """
    w = A.float()
    deg = w.sum(1).clamp(min=1.0)
    tau = 1.0 / (1.0 + kappa * delta.clamp(min=0.0))
    for _ in range(steps):
        tau = tau + nu * ((w @ tau) / deg - tau)
        tau = tau.clamp(min=floor)
        tau = tau / tau.mean().clamp(min=1e-9)
    return tau


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=3, choices=(2, 3))
    ap.add_argument("-n", type=int, default=4000)
    ap.add_argument("--box", type=float, default=60.0)
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=1.5)
    ap.add_argument("--sec-balance", dest="sec_balance", type=float, default=0.65334)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--diffuse", type=int, default=120)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--viscosity", type=float, default=0.3)
    ap.add_argument("--rewires", type=int, default=30, help="shuffled graphs per seed")
    ap.add_argument("--res", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 101])
    ap.add_argument("--compare-uniform", dest="compare_uniform", action="store_true",
                    help="also run the null inside a UNIFORM box and compare the GAPS")
    a = ap.parse_args()

    print(f"  {a.n} particles, {a.dims}D, web epoch {a.steps}, k-NN k={a.k}, "
          f"{a.diffuse} tau-diffusion steps")
    print(f"  null = degree-preserving edge shuffle, {a.rewires} rewires/seed")
    print(f"  everything but the WIRING is held identical between arms\n")
    print(f"  {'kind':>8} {'seed':>5} {'perc':>6} {'deg ok':>7} | {'REAL':>8} "
          f"{'shuffled':>18} {'GAP':>7} {'z':>7}")

    kinds = ["web", "uniform"] if a.compare_uniform else ["web"]
    rows = []
    for kind in kinds:
      for sd in a.seeds:
        cfg = ParticleConfig(n=a.n, box=a.box, r0=a.r0, g=a.g, sec_balance=a.sec_balance,
                             dims=a.dims, seed=sd, entropy_init=0.1, time_mode="potential",
                             time_viscosity=a.viscosity, time_coupling="knn", time_k=a.k)
        eng = ParticleEngine(cfg, pipeline=EXP11_TIME)
        if kind == "web":
            for _ in range(a.steps):
                eng.tick()
        else:
            # Spatially embedded but structureless. The shuffle destroys the web AND spatial
            # locality together, so a web-only null cannot tell them apart. Running the SAME
            # null inside a uniform box gives the baseline for generic spatial embedding, and
            # each arm is compared against a shuffle of ITSELF -- so the GAPS are comparable
            # even though the raw correlations are not.
            g = torch.Generator(device="cpu").manual_seed(sd + 5150)
            pos = (torch.rand(a.n, a.dims, generator=g) * a.box).to(eng.state.pos.device)
            eng.state = eng.state.replace(pos=pos, vel=torch.zeros_like(pos))

        dev = eng.state.pos.device
        edges = knn_edges(eng.state, a.k)
        A = adjacency(edges, a.n, dev)
        deg_real = A.sum(1).cpu().numpy()

        r, _, _ = pairwise(eng.state, 1.0)
        near = (r < a.r0); near.fill_diagonal_(False)
        count = near.sum(dim=1).float()
        delta = count / count.mean().clamp(min=1e-9) - 1.0

        rho = eng.density_field(a.res).cpu().numpy().astype(float)
        perc = float(web_metrics(rho)["percolation"])
        phi = potential(rho, a.box, screening=None, g=a.g)

        def score(Adj):
            tau = diffuse_tau(Adj, delta, a.kappa, a.viscosity, a.diffuse)
            t = eng.field_of(tau, a.res).cpu().numpy()
            ok = np.isfinite(t)
            return float(np.corrcoef(t[ok], phi[ok])[0, 1])

        real = score(A)

        rng = np.random.default_rng(sd + 909)
        nulls, degs_ok = [], True
        for _ in range(a.rewires):
            sh = degree_preserving_shuffle(edges, a.n, rng)
            Ash = adjacency(sh, a.n, dev)
            if not np.array_equal(np.sort(Ash.sum(1).cpu().numpy()), np.sort(deg_real)):
                degs_ok = False
            nulls.append(score(Ash))
        nulls = np.array(nulls)
        z = (real - nulls.mean()) / nulls.std() if nulls.std() > 1e-12 else float("nan")
        print(f"  {kind:>8} {sd:>5} {perc:>6.3f} {str(degs_ok):>7} | {real:>+8.3f} "
              f"{nulls.mean():>+9.3f} +-{nulls.std():.3f} {real-nulls.mean():>+7.3f} {z:>+7.2f}")
        rows.append({"kind": kind, "gap": float(real - nulls.mean()),
                     "seed": sd, "percolation": perc, "degree_preserved": degs_ok,
                     "real": real, "null_mean": float(nulls.mean()),
                     "null_std": float(nulls.std()), "z": float(z),
                     "nulls": nulls.tolist()})

    def sel(k, key):
        return np.array([r[key] for r in rows if r["kind"] == k])

    for k in kinds:
        print(f"\n  {k:>8}: real {sel(k,'real').mean():+.3f}  shuffled "
              f"{sel(k,'null_mean').mean():+.3f}  GAP {sel(k,'gap').mean():+.3f} "
              f"+-{sel(k,'gap').std():.3f}  (mean z {sel(k,'z').mean():+.1f})")

    if a.compare_uniform:
        wg, ug = sel("web", "gap"), sel("uniform", "gap")
        se = np.sqrt(wg.var(ddof=1) / len(wg) + ug.var(ddof=1) / len(ug))
        exc = wg.mean() - ug.mean()
        print(f"\n  EXCESS (web gap - uniform gap) = {exc:+.3f}  "
              f"({exc/se if se > 0 else 0:+.2f} sigma)")
        print(f"  > 0 => the WEB's topology matters beyond generic spatial embedding.")
        print(f"  ~ 0 => tau follows spatial locality and the web adds nothing on top.")
    else:
        print(f"\n  H0: tau's agreement with the long-range potential is unchanged when the")
        print(f"      wiring is destroyed but every other property is preserved.")
        print(f"  NOTE: a shuffle destroys the web AND spatial locality together. Use")
        print(f"        --compare-uniform to separate them; without it this cannot attribute")
        print(f"        the effect to the WEB specifically.")
    if not all(r["degree_preserved"] for r in rows):
        print(f"\nWARNING: a shuffle did not preserve the degree sequence. Null invalid.")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(1, len(rows), figsize=(4.4 * len(rows), 4.2), squeeze=False)
    for k, r in enumerate(rows):
        axi = ax[0][k]
        axi.hist(r["nulls"], bins=12, alpha=0.75, color="tab:blue",
                 label=f"shuffled (n={len(r['nulls'])})")
        axi.axvline(r["real"], color="tab:red", lw=2.5, label="real graph")
        axi.set_title(f"{r['kind']} seed {r['seed']} — gap {r['gap']:+.3f}", fontsize=10)
        axi.set_xlabel("corr(tau, Phi_newton)"); axi.grid(alpha=0.3); axi.legend(fontsize=8)
    fig.suptitle("does the clock field follow the web? — degree-preserving shuffle null",
                 fontsize=12)
    fig.tight_layout()
    p = OUT / f"exp_07_shuffle_null_{a.dims}d_{stamp}.png"
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    (OUT / f"exp_07_shuffle_null_{stamp}.json").write_text(
        json.dumps({"config": vars(a), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
