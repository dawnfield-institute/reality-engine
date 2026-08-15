#!/usr/bin/env python3
"""Does the expansion window survive replication?

exp_01 found an interior optimum: H0 = 0.001 at Omega_L = 1/phi retained 53.4% of its peak
percolation against the static control's 42.5%, and beat it on final percolation 0.093 to
0.068. Too little expansion looked like static; too much was worse than static.

On ONE seed. The peak column bounced non-monotonically in a way consistent with noise, and a
37% difference on a single realization is the kind of result that evaporates on replication.
So: three arms, five seeds, error bars.

    python .../exp_02_replicate_the_window.py
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "proof_of_concepts" / "v4"))
import numpy as np  # noqa: E402
from particles import PHI, Cosmology, ParticleConfig, ParticleEngine  # noqa: E402
from structure import percolation  # noqa: E402

ARMS = [("static", None), ("H0 0.0005", 0.0005), ("H0 0.001", 0.001), ("H0 0.002", 0.002)]
SEEDS = [42, 7, 99, 13, 71]
STEPS, N = 4000, 4000

def run(h0, seed):
    cos = None if h0 is None else Cosmology(h0=h0, omega_lambda=1.0 / PHI)
    eng = ParticleEngine(ParticleConfig(n=N, box=120.0, r0=5.0, g=0.8,
                                        sec_balance=0.6, seed=seed, cosmology=cos))
    ps = []
    for t in range(1, STEPS + 1):
        eng.tick()
        if t % 50 == 0:
            ps.append(percolation(eng.density_field(128).cpu().numpy().astype(float)))
    ps = np.array(ps)
    return float(ps.max()), float(ps[-5:].mean())   # tail mean, not a single noisy sample

rows = []
print(f"  {N} particles, {STEPS} steps, Omega_L = 1/phi, {len(SEEDS)} seeds")
print(f"  {'arm':<12} {'peak':>16} {'final (tail mean)':>22} {'retained':>16}")
print("  " + "-" * 70)
for label, h0 in ARMS:
    pk, fi = zip(*[run(h0, s) for s in SEEDS])
    pk, fi = np.array(pk), np.array(fi)
    ret = fi / pk
    print(f"  {label:<12} {pk.mean():>8.3f}+-{pk.std():<6.3f} "
          f"{fi.mean():>13.3f}+-{fi.std():<6.3f} {100*ret.mean():>10.1f}%+-{100*ret.std():<4.1f}")
    rows.append({"arm": label, "h0": h0, "peak_mean": pk.mean(), "peak_std": pk.std(),
                 "final_mean": fi.mean(), "final_std": fi.std(),
                 "retained_mean": float(ret.mean()), "retained_std": float(ret.std()),
                 "seeds": SEEDS})

out = Path(__file__).resolve().parents[1] / "results"
out.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
(out / f"exp_02_replication_{stamp}.json").write_text(
    json.dumps({"steps": STEPS, "n": N, "seeds": SEEDS, "arms": rows}, indent=2), encoding="utf-8")
print(f"\n  wrote results/exp_02_replication_{stamp}.json")
