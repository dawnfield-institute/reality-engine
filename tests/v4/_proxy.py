"""The proxy run shared by the v4 tests, and the import shim for proof_of_concepts/v4.

Kept out of conftest.py because a conftest is a pytest plugin, not an importable module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "proof_of_concepts" / "v4"
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

import particles as P  # noqa: E402  (after sys.path)

# exp_11's 3D config (n=4000, box=60, r0=10, g=1.5) at the same number density, n=500:
# box = 60 * (500/4000)^(1/3) = 30.0. sec_balance = Xi/phi, the diagnostic's middle arm.
# On the unfixed code this pins every particle at the cap by tick ~75 and runs entropy up
# monotonically — the n=4000 shape, at 4 ms per tick on CPU.
PROXY = dict(n=500, box=30.0, r0=10.0, g=1.5, dims=3, seed=1, sec_balance=0.6541)

# Sampling ticks: mid-saturation (50: gravity alone has pinned ~64%), fully pinned (100+),
# and well into the entropy runaway (300 = the diagnostic's own horizon at n=4000).
MARKS = (25, 50, 100, 150, 200, 300)


def run_marks(cfg: "P.ParticleConfig", pipeline=None, marks=MARKS, min_sim_time: float = 15.0):
    """Run an engine; return (engine, [(tick, sim_time, metrics-copy), ...]).

    Stops at the last mark or when `sim_time` reaches `min_sim_time`, whichever is later, so
    a repaired integrator that shrinks dt still covers the same span of simulated time as the
    unfixed one (300 ticks x 0.05). Falls back to tick * dt when the `sim_time` metric is
    absent — the fixture has to run on the unfixed code too.
    """
    eng = P.ParticleEngine(cfg, pipeline=pipeline, device=torch.device("cpu"))
    out, t = [], 0
    last_mark = max(marks)
    while True:
        s = eng.tick()
        t += 1
        sim_time = float(s.metrics.get("sim_time", t * cfg.dt))
        if t in marks:
            out.append((t, sim_time, dict(s.metrics)))
        if t >= last_mark and sim_time >= min_sim_time:
            if t not in marks:
                out.append((t, sim_time, dict(s.metrics)))
            break
        if t > 20_000:  # a floored dt cannot make this run forever
            out.append((t, sim_time, dict(s.metrics)))
            break
    return eng, out
