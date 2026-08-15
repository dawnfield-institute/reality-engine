#!/usr/bin/env python3
"""Is the web transient because of dissipation rather than because of expansion?

exp_02 killed the expansion hypothesis: across five seeds no expansion rate beat the static
control by more than 1.2 sigma, and seed scatter swamped every between-arm difference.

The alternative was flagged before that result came in. `damping = 0.99` multiplies every
velocity every tick, and `max_speed` clips the tail. Together they bleed energy continuously
and nothing replaces it. A system losing energy virializes into isolated bound clumps whatever
the box is doing — which is exactly the decay POC-07 measured.

That is a substrate property chosen without registering it as a physics commitment: the same
failure mode as the field engine, where the representation was making claims nobody was
reading.

One run answers it. Turn the dissipation down and see whether the web still decays.

    python .../exp_03_is_it_the_dissipation.py
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))
import numpy as np, torch  # noqa: E402
from particles import ParticleConfig, ParticleEngine  # noqa: E402
from structure import percolation  # noqa: E402

ARMS = [
    ("damping 0.99 (POC-07)", 0.99, 2.0),
    ("damping 0.999",         0.999, 2.0),
    ("damping 1.0 (none)",    1.0, 2.0),
    ("damping 1.0, cap 20",   1.0, 20.0),
]
SEEDS = [42, 7, 99]
STEPS, N = 4000, 4000

def run(damping, cap, seed):
    eng = ParticleEngine(ParticleConfig(n=N, box=120.0, r0=5.0, g=0.8, sec_balance=0.6,
                                        damping=damping, max_speed=cap, seed=seed))
    ps, ke = [], []
    for t in range(1, STEPS + 1):
        s = eng.tick()
        if t % 50 == 0:
            ps.append(percolation(eng.density_field(128).cpu().numpy().astype(float)))
            ke.append(s.metrics.get("kinetic", float("nan")))
            if not torch.isfinite(s.vel).all():
                return float("nan"), float("nan"), float("nan")
    ps = np.array(ps)
    return float(ps.max()), float(ps[-5:].mean()), float(ke[-1] / (ke[0] + 1e-12))

rows = []
print(f"  {N} particles, {STEPS} steps, static box, {len(SEEDS)} seeds")
print(f"  {'arm':<24} {'peak':>15} {'final':>15} {'retained':>14} {'KE end/start':>13}")
print("  " + "-" * 86)
for label, d, cap in ARMS:
    res = [run(d, cap, s) for s in SEEDS]
    pk = np.array([r[0] for r in res]); fi = np.array([r[1] for r in res])
    kr = np.array([r[2] for r in res])
    if not np.isfinite(pk).all():
        print(f"  {label:<24} DIVERGED"); continue
    ret = fi / pk
    print(f"  {label:<24} {pk.mean():>7.3f}+-{pk.std():<6.3f} {fi.mean():>7.3f}+-{fi.std():<6.3f} "
          f"{100*ret.mean():>8.1f}%+-{100*ret.std():<4.1f} {kr.mean():>13.3g}")
    rows.append({"arm": label, "damping": d, "cap": cap,
                 "peak_mean": pk.mean(), "peak_std": pk.std(),
                 "final_mean": fi.mean(), "final_std": fi.std(),
                 "retained_mean": float(ret.mean()), "retained_std": float(ret.std()),
                 "ke_ratio_mean": float(kr.mean()), "seeds": SEEDS})

out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
(out / f"exp_03_dissipation_{stamp}.json").write_text(
    json.dumps({"steps": STEPS, "n": N, "seeds": SEEDS, "arms": rows}, indent=2), encoding="utf-8")
print(f"\n  wrote results/exp_03_dissipation_{stamp}.json")
