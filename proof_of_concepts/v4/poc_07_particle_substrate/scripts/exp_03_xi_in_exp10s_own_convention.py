#!/usr/bin/env python3
"""Test exp_10's Xi claim properly: its own convention, past its endpoint, with seeds.

exp_10 reports two things:

    "NO discrete phase transition -- SEC is CONTINUOUS control."   <- supported by its data
    "Xi ~ 1.057 is ... the OPTIMAL OPERATING POINT for maximum      <- not supported
     structural complexity."

Checking exp_10's own numbers, none of its five metrics peaks at Xi and Xi is not a local
maximum of any of them. Two of the five maxima sit on the boundaries of its swept range
(0.30-1.30), so the optimum was never bracketed: `density_cv` simply rises monotonically to the
upper endpoint, 1.793 -> 2.157.

exp_02 swept the v4 substrate and found CV falling monotonically instead, but that does NOT
test exp_10 -- the v4 substrate uses attractive gravity where exp_09/10/11 use the repulsive
convention, and POC-09 established that the repulsive form is what reproduces the corpus's
published numbers to three significant figures.

So this is the fair test. Three changes from exp_02:

  1. **exp_10's convention**, transcribed literally: force_dir = +deltas for gravity and
     -deltas for pressure, exp_10's relative-deviation SEC rule, its config
     (n=2000, box=100, r0=6, g=0.8, 600 steps).
  2. **Past the endpoint** -- swept to 2.5 so an optimum near 1.3 would be bracketed rather
     than sitting on the boundary.
  3. **Five seeds**, with sigma reported against the Xi arm. exp_10 ran one.

Percolation is measured alongside density_cv, since exp_10 had no connectivity metric and
POC-09 showed its published web does not percolate.

    python .../exp_03_xi_in_exp10s_own_convention.py
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
import torch  # noqa: E402

from structure import percolation, web_metrics  # noqa: E402

PHI = (1 + math.sqrt(5)) / 2
XI_ANALYTIC = 0.5772156649015329 + math.log(PHI)      # 1.05843
XI_DISCRETE = 1.0 + math.pi / 55                       # 1.05712

# exp_10's config, from its own SweepConfig and results JSON.
N, BOX, R0, G, STEPS, DECAY = 2000, 100.0, 6.0, 0.8, 600, 0.95

# exp_10 swept 0.30-1.30. Extended to 2.5 so a maximum near the old endpoint is bracketed.
BALANCES = [0.3, 0.5, 0.7, 0.9, 1.057, 1.3, 1.6, 2.0, 2.5]
SEEDS = [42, 7, 99, 13, 71]


def run_exp10_convention(sec_balance, seed, steps=STEPS, res=100):
    """exp_10 transcribed: repulsive `gravity`, attractive `pressure`, relative SEC rule."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    per = int(math.ceil(N ** 0.5))
    sp = BOX / per
    g = torch.arange(per, device=dev, dtype=torch.float32) * sp + sp / 2
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    pos = torch.stack([xx.flatten()[:N], yy.flatten()[:N]], dim=1)
    pos = (pos + torch.randn_like(pos) * sp * 0.1) % BOX
    vel = torch.zeros_like(pos)
    mass = 1.0 + 0.1 * torch.randn(N, device=dev)
    ent = torch.zeros(N, device=dev)
    dt = 0.05

    best_cv, best_perc = 0.0, 0.0
    for t in range(1, steps + 1):
        de = pos.unsqueeze(1) - pos.unsqueeze(0)
        de = de - BOX * torch.round(de / BOX)
        dist = torch.sqrt((de ** 2).sum(2) + 1e-8)
        dist.fill_diagonal_(float("inf"))
        mp = mass.unsqueeze(1) * mass.unsqueeze(0)
        fm = torch.where(dist < 3 * R0, G * mp * torch.exp(-dist / R0) / (dist + 0.1),
                         torch.zeros_like(dist))
        fd = de / (dist.unsqueeze(2) + 0.1)          # points AWAY from j, as written
        grav = (fm.unsqueeze(2) * fd).sum(1)
        ed = ent.unsqueeze(1) - ent.unsqueeze(0)
        pm = torch.where(dist < 2 * R0, sec_balance * ed * torch.exp(-dist / R0),
                         torch.zeros_like(dist))
        press = (pm.unsqueeze(2) * (-fd)).sum(1)
        vel = 0.99 * vel + (grav + press) * dt / mass.unsqueeze(1)
        sp_ = vel.norm(dim=1, keepdim=True)
        vel = torch.where(sp_ > 2.0, vel * 2.0 / sp_, vel)
        pos = (pos + vel * dt) % BOX
        lc = (dist < R0).sum(1).float()
        mc = lc.mean()
        ent = torch.clamp(ent + sec_balance * ((lc - mc) / (mc + 1)), min=0.0)

        if t % 50 == 0:
            ix = (pos[:, 0] / BOX * res).long() % res
            iy = (pos[:, 1] / BOX * res).long() % res
            f = torch.zeros(res * res, device=dev)
            f.scatter_add_(0, iy * res + ix, mass)
            F = f.view(res, res).cpu().numpy().astype(float)
            w = web_metrics(F)
            best_cv = max(best_cv, w["cv"])
            best_perc = max(best_perc, w["percolation"])
    return best_cv, best_perc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    print(f"  exp_10 convention: n={N} box={BOX} r0={R0} g={G}, {args.steps} steps, "
          f"{len(SEEDS)} seeds, 100^2 binning")
    print(f"  exp_10 swept 0.30-1.30 on ONE seed and its density_cv rose monotonically to the "
          f"upper endpoint (1.793 -> 2.157)")
    print(f"  Xi_analytic = {XI_ANALYTIC:.5f}\n")
    hdr = f"  {'sec_balance':>12} {'peak CV':>18} {'peak percolation':>22}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for b in BALANCES:
        res = [run_exp10_convention(b, s, args.steps) for s in SEEDS]
        cv = np.array([r[0] for r in res])
        pc = np.array([r[1] for r in res])
        mark = "  <- Xi" if abs(b - 1.057) < 1e-6 else ""
        print(f"  {b:>12.3f} {cv.mean():>10.3f}+-{cv.std():<6.3f} "
              f"{pc.mean():>14.4f}+-{pc.std():<6.4f}{mark}")
        rows.append({"sec_balance": b, "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
                     "perc_mean": float(pc.mean()), "perc_std": float(pc.std())})

    b = np.array([r["sec_balance"] for r in rows])
    cvm = np.array([r["cv_mean"] for r in rows]); cvs = np.array([r["cv_std"] for r in rows])
    pcm = np.array([r["perc_mean"] for r in rows]); pcs = np.array([r["perc_std"] for r in rows])
    xi_i = int(np.argmin(np.abs(b - 1.057)))

    print(f"\n  peak CV          at sec_balance = {b[int(cvm.argmax())]:.3f}"
          f"{'  (BRACKETED)' if 0 < cvm.argmax() < len(b)-1 else '  (still at an endpoint)'}")
    print(f"  peak percolation at sec_balance = {b[int(pcm.argmax())]:.3f}"
          f"{'  (BRACKETED)' if 0 < pcm.argmax() < len(b)-1 else '  (still at an endpoint)'}")
    for name, m, sd in (("CV", cvm, cvs), ("percolation", pcm, pcs)):
        k = int(m.argmax())
        if k == xi_i:
            print(f"  {name}: peak IS the Xi arm")
            continue
        se = np.sqrt(sd[k] ** 2 + sd[xi_i] ** 2) / np.sqrt(len(SEEDS))
        print(f"  {name}: best arm ({b[k]:.3f}) beats Xi by "
              f"{(m[k]-m[xi_i])/se if se > 0 else float('nan'):.2f} sigma")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, m, sd, lab in ((ax[0], cvm, cvs, "peak density CV"),
                          (ax[1], pcm, pcs, "peak percolation")):
        a.errorbar(b, m, yerr=sd / np.sqrt(len(SEEDS)), marker="o", lw=2, capsize=3)
        a.axvline(XI_ANALYTIC, color="crimson", ls="--", lw=1.2, label="Xi")
        a.axvspan(0.3, 1.3, color="grey", alpha=0.12, label="exp_10's swept range")
        a.set_xlabel("sec_balance"); a.set_ylabel(lab); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("exp_10's convention, swept past its endpoint, 5 seeds", fontsize=11)
    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "xi_exp10_convention.png", dpi=115, bbox_inches="tight")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out / f"exp_03_xi_exp10_{stamp}.json").write_text(json.dumps(
        {"config": {"n": N, "box": BOX, "r0": R0, "g": G, "steps": args.steps},
         "seeds": SEEDS, "xi_analytic": XI_ANALYTIC, "arms": rows}, indent=2), encoding="utf-8")
    print(f"\n  wrote results/xi_exp10_convention.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
