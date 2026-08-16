#!/usr/bin/env python3
"""Exploration: does the discarded curl make the connections?

`charge_dynamics.py` builds a vector force from the charge potential and then adds only its
DIVERGENCE to dE_dt. Measured at 128x128 after 3000 ticks, the discarded curl is 89% the
magnitude of the kept divergence. Every other force in the engine is likewise a gradient of
a scalar potential, and a gradient flow has point attractors — it makes blobs. Filaments are
line-like and live in the divergence-free part.

The engine's measured failure is exactly that shape: contrast yes (void 0.65, cv 1.47),
connectivity no (percolation 0.013 against a web's 1.000, ~1300 fragments).

So: let mass be TRANSPORTED along the charge force and split the force by Helmholtz to see
which half does what. Four arms:

    baseline      no extra transport (what the engine does today)
    gradient      advect along the curl-free part only
    curl          advect along the divergence-free part only
    full          advect along both

Advection is a flux divergence, dM = -div(M*v)*dt, so it moves mass without creating or
destroying it — the PAC ledger is untouched by construction, which is the point: a purely
rotational flow rearranges without changing any total.

    python proof_of_concepts/v4/explore_curl.py [--ticks 3000]
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
from structure import coherent_fraction, percolation, web_metrics  # noqa: E402


def d(x, dim):
    return (torch.roll(x, -1, dim) - torch.roll(x, 1, dim)) / 2.0


def charge_force(E, I):
    """The vector force `charge_dynamics.py` builds before it takes the divergence."""
    Q = d(E, 0) - d(I, 1)
    diseq = E - I
    S = d(diseq, 0) - d(diseq, 1)
    e_loc = Q.pow(2) / (Q.pow(2) + S.pow(2) + 1e-12)
    n = E.shape[0]
    dev = E.device
    ku = (torch.fft.fftfreq(E.shape[0], device=dev) * 2 * np.pi).view(-1, 1)
    kv = (torch.fft.fftfreq(E.shape[1], device=dev) * 2 * np.pi).view(1, -1)
    k2 = (ku ** 2 + kv ** 2).clone()
    k2[0, 0] = 1.0
    phi = torch.fft.ifft2(torch.fft.fft2(Q) / (-k2)).real
    return e_loc * Q * d(phi, 0), e_loc * Q * d(phi, 1)


def helmholtz(Fu, Fv):
    """Split F into curl-free (gradient) and divergence-free (curl) parts."""
    dev = Fu.device
    ku = (torch.fft.fftfreq(Fu.shape[0], device=dev) * 2 * np.pi).view(-1, 1)
    kv = (torch.fft.fftfreq(Fu.shape[1], device=dev) * 2 * np.pi).view(1, -1)
    k2 = (ku ** 2 + kv ** 2).clone()
    k2[0, 0] = 1.0
    div = d(Fu, 0) + d(Fv, 1)
    phi = torch.fft.ifft2(torch.fft.fft2(div) / (-k2)).real
    Gu, Gv = d(phi, 0), d(phi, 1)          # curl-free part
    return (Gu, Gv), (Fu - Gu, Fv - Gv)    # gradient, curl


def advect(M, vu, vv, dt):
    """dM = -div(M v) dt by donor-cell upwind. Conservative and stable under CFL.

    The obvious thing — centred differences on M*v with forward Euler — is
    UNCONDITIONALLY UNSTABLE for advection, at any timestep. It is fine for diffusion,
    which is why it looks reasonable. The first version of this function used it: at low
    transport it grew slowly enough to survive 3000 ticks and read as a null result, and at
    50x transport it diverged in every arm. Neither run tested the physics.

    Fluxes here are computed at cell faces from the upwind cell and then differenced, so
    what leaves one cell enters its neighbour exactly and total M is conserved to rounding.
    """
    fu = torch.where(vu > 0, M * vu, torch.roll(M, -1, 0) * vu)
    fv = torch.where(vv > 0, M * vv, torch.roll(M, -1, 1) * vv)
    return M - dt * ((fu - torch.roll(fu, 1, 0)) + (fv - torch.roll(fv, 1, 1)))


def run(mode, ticks, nu=128, nv=128, seed=42, strength=0.5, eta=0.025):
    torch.manual_seed(seed)
    eng = Engine(config=SimulationConfig(nu=nu, nv=nv, noise_scale=0.0,
                                         deactualization_rate=eta),
                 pipeline=build_canonical_pipeline())
    eng.initialize("big_bang", temperature=3.0)
    for _ in range(ticks):
        eng.tick()
        if mode == "baseline":
            continue
        s = eng.state
        Fu, Fv = charge_force(s.E, s.I)
        (Gu, Gv), (Cu, Cv) = helmholtz(Fu, Fv)
        if mode == "gradient":
            vu, vv = Gu, Gv
        elif mode == "curl":
            vu, vv = Cu, Cv
        else:
            vu, vv = Fu, Fv
        # normalise so the arms are compared at equal transport magnitude, not equal
        # coefficient — otherwise "curl does less" could just mean "curl is smaller".
        mag = (vu.pow(2) + vv.pow(2)).mean().sqrt().clamp(min=1e-12)
        M_new = advect(s.M, strength * vu / mag, strength * vv / mag, eng.config.dt)
        eng.state = s.replace(M=torch.clamp(M_new, min=0.0))
        if not torch.isfinite(eng.state.M).all():
            return None
    return eng.state.M.detach().cpu().numpy().astype(float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--eta", type=float, default=0.8)
    args = ap.parse_args()

    web = np.full((128, 128), 0.01)
    for u in range(0, 128, 16):
        web[u:u + 3, :] = 3.0
    for v in range(0, 128, 16):
        web[:, v:v + 3] = 3.0
    print(f"  128x128, {args.ticks} ticks, eta={args.eta}")
    print(f"  synthetic web: percolation {percolation(web):.3f}   "
          f"white noise: {percolation(np.abs(np.random.default_rng(0).standard_normal((128,128)))):.3f}\n")

    hdr = f"  {'arm':<12} {'percolation':>12} {'void':>7} {'cv':>7} {'xi_u':>7} {'coh':>8}  web"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows, panels = [], []
    for mode in ("baseline", "gradient", "curl", "full"):
        M = run(mode, args.ticks, eta=args.eta)
        if M is None:
            print(f"  {mode:<12} DIVERGED")
            continue
        w = web_metrics(M)
        print(f"  {mode:<12} {w['percolation']:>12.3f} {w['void']:>7.3f} {w['cv']:>7.3f} "
              f"{w['xi_u']:>7.3f} {coherent_fraction(M):>8.4f}  {w['is_web']}")
        rows.append({"arm": mode, "coh": coherent_fraction(M), **w})
        panels.append((mode, M, w))

    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4))
        for ax, (mode, M, w) in zip(np.atleast_1d(axes), panels):
            ax.imshow(M.T, origin="lower", cmap="magma", aspect="equal",
                      interpolation="nearest", vmin=M.min(), vmax=np.percentile(M, 99.5))
            ax.set_title(f"{mode}\nperc {w['percolation']:.3f}  cv {w['cv']:.2f}",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Transporting mass along the charge force — Helmholtz split",
                     fontsize=12)
        fig.tight_layout()
        outdir = (REPO / "proof_of_concepts" / "v4" /
                  "poc_05_structure_exploration" / "results")
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "curl_split.png", dpi=115, bbox_inches="tight")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (outdir / f"curl_split_{stamp}.json").write_text(
            json.dumps({"ticks": args.ticks, "eta": args.eta, "rows": rows}, indent=2),
            encoding="utf-8")
        print(f"\n  wrote {(outdir / 'curl_split.png').relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
