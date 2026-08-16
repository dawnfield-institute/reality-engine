#!/usr/bin/env python3
"""v4 exploration panel — does this engine build structure, under any regime tried?

Exploratory, not predictive. The point is to find a regime where geometry DEVELOPS, and to
record honestly which levers did nothing.

Metrics come from structure.py, which is calibrated against known geometries (filament 1.0,
plane 2.0, points 0.08). The previous estimator was not, and returned 2.000 for filament,
blob and scattered points alike — every structure claim made with it was void.

    python proof_of_concepts/v4/explore_structure.py [--ticks 4000] [--grid 128 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))

import torch  # noqa: E402

from src.v3.engine.config import SimulationConfig  # noqa: E402
from src.v3.engine.engine import Engine  # noqa: E402
from src.v3.engine.pipelines import CANONICAL  # noqa: E402
from src.v3.operators.gravity import GravitationalCollapseOperator  # noqa: E402
from src.v3.operators.pac_balance import PACBalanceOperator  # noqa: E402
from src.v3.operators.protocol import Pipeline  # noqa: E402
from src.v3.operators.rbf import RBFOperator  # noqa: E402
from structure import box_dimension, contrast, occupied_fraction, shannon_entropy  # noqa: E402


def build(balance="rbf", gravity=True, poisson_power=None):
    ops = []
    for o in CANONICAL:
        if o is RBFOperator and balance == "pac":
            ops.append(PACBalanceOperator); continue
        if o is GravitationalCollapseOperator:
            if not gravity:
                continue
            if poisson_power is not None:
                p = poisson_power

                class G(GravitationalCollapseOperator):
                    def __call__(self, state, config, bus=None):
                        real = torch.sqrt

                        def fake(x):
                            return (x.pow(p) if x.shape == state.M.shape
                                    and bool((x >= 0).all()) else real(x))
                        torch.sqrt = fake
                        try:
                            return super().__call__(state, config, bus)
                        finally:
                            torch.sqrt = real
                ops.append(G); continue
        ops.append(o)
    return Pipeline([o() for o in ops])


def run_variant(name, ticks, grid, init="big_bang", seed=42, **kw):
    nu, nv = grid
    cfg_kw = {k: v for k, v in kw.items() if k in
              ("noise_scale", "mass_diffusion_coeff", "gamma_damping", "enforce_pac")}
    build_kw = {k: v for k, v in kw.items() if k in ("balance", "gravity", "poisson_power")}
    torch.manual_seed(seed)
    cfg = SimulationConfig(nu=nu, nv=nv, noise_scale=cfg_kw.pop("noise_scale", 0.0), **cfg_kw)
    eng = Engine(config=cfg, pipeline=build(**build_kw))
    eng.initialize(init, temperature=3.0)

    marks = [ticks // 8, ticks // 2, ticks]
    snaps, t0 = [], time.time()
    diverged = None
    for t in range(1, ticks + 1):
        eng.tick()
        if not torch.isfinite(eng.state.M).all():
            diverged = t
            break
        if t in marks:
            s = eng.state
            snaps.append({
                "tick": t,
                "D_M": box_dimension(s.M),
                "D_diseq": box_dimension((s.E - s.I).abs()),
                "H_M": shannon_entropy(s.M),
                "contrast_M": contrast(s.M),
                "occ_M": occupied_fraction(s.M),
                "M_total": float(s.M.sum().item()),
            })
    return {"name": name, "init": init, "diverged_at": diverged,
            "wall_s": round(time.time() - t0, 1), "snaps": snaps, **kw}


VARIANTS = [
    ("baseline RBF",             dict(balance="rbf")),
    ("PAC balance",              dict(balance="pac")),
    ("RBF, no gravity",          dict(balance="rbf", gravity=False)),
    ("PAC, no gravity",          dict(balance="pac", gravity=False)),
    ("RBF, poisson M^0.8",       dict(balance="rbf", poisson_power=0.8)),
    ("RBF, high viscosity",      dict(balance="rbf", mass_diffusion_coeff=0.02)),
    ("RBF, low damping",         dict(balance="rbf", gamma_damping=0.001)),
    ("RBF, unenforced PAC",      dict(balance="rbf", enforce_pac=False)),
    ("RBF, thermal noise on",    dict(balance="rbf", noise_scale=0.01)),
]
INITS = ["big_bang", "entropy_dominated", "info_dominated"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--grid", type=int, nargs=2, default=[128, 64])
    args = ap.parse_args()

    rows = []
    print(f"  grid {args.grid[0]}x{args.grid[1]}, {args.ticks} ticks\n")
    hdr = f"  {'variant':<26} {'init':<18} {'D(M) start->end':>20} {'dD':>7} {'contrast':>16} {'H(M)':>16}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name, kw in VARIANTS:
        for init in (INITS if name in ("baseline RBF", "PAC balance") else ["big_bang"]):
            r = run_variant(name, args.ticks, tuple(args.grid), init=init, **kw)
            rows.append(r)
            if r["diverged_at"]:
                print(f"  {name:<26} {init:<18} {'DIVERGED at t=' + str(r['diverged_at']):>20}")
                continue
            a, b = r["snaps"][0], r["snaps"][-1]
            dd = b["D_M"] - a["D_M"]
            print(f"  {name:<26} {init:<18} {a['D_M']:>9.3f} ->{b['D_M']:>9.3f} {dd:>+7.3f} "
                  f"{a['contrast_M']:>7.4f}->{b['contrast_M']:>7.4f} "
                  f"{a['H_M']:>7.3f}->{b['H_M']:>7.3f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = REPO / "proof_of_concepts" / "v4" / "results" / f"explore_structure_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"grid": args.grid, "ticks": args.ticks, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
