"""v4 experiment harness — parameter sweeps and twin-difference durability.

Two primitives, both extracted from POC-01 rather than designed up front:

    sweep(...)        run a parameter grid, collect per-run metric trajectories
    durability(...)   displace an observable, measure whether it returns

`durability` is the one that matters. It differences against an identically-seeded twin,
so the engine's own baseline behaviour cancels exactly and the measurement needs no
position on whether that baseline is physical or numerical. POC-01 v1 failed by trying to
characterise the baseline; v2 succeeded by differencing it away.

Deliberately small. This is a harness, not a framework: no run registry, no persistence
layer, no config system. Those exist off the shelf and should be adopted rather than
written here if they are ever needed.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from src.v3.engine.pipelines import build_canonical_pipeline
from src.v3.engine.config import SimulationConfig
from src.v3.engine.engine import Engine

# Bands are shared with POC-01 so results across POCs are directly comparable.
DURABLE_MAX = 0.5
NEUTRAL_LO, NEUTRAL_HI = 0.9, 1.1
UNSTABLE_MIN = 1.5


def make_engine(seed: int, **cfg_kwargs) -> Engine:
    """A fresh engine on the CANONICAL 16-operator pipeline.

    POC-01..03 used the dashboard's reduced 12, which has no ActualizationOperator
    and therefore no live Delta buffer. Any durability result measured that way was
    measuring reduced physics.
    """
    torch.manual_seed(seed)
    cfg = SimulationConfig(**cfg_kwargs)
    eng = Engine(config=cfg, pipeline=build_canonical_pipeline())
    eng.initialize(mode=cfg_kwargs.pop("mode", "big_bang"))
    return eng


def classify(R: float) -> str:
    """Registered bands. Returns 'ambiguous' rather than rounding toward a hypothesis."""
    if R != R:
        return "invalid"
    if R < DURABLE_MAX:
        return "durable"
    if NEUTRAL_LO <= R <= NEUTRAL_HI:
        return "neutral"
    if R > UNSTABLE_MIN:
        return "unstable"
    return "ambiguous"


@dataclass
class DurabilityResult:
    observable: str
    D0: float
    D_end: float
    R: float
    verdict: str
    trajectory: list[float]
    pre_impulse_gap: float
    # The reference twin's own values, unperturbed. Lets a caller measure a property of
    # the baseline dynamics from the SAME run that produced R, so the two cannot differ
    # by run conditions.
    reference_trajectory: list[float] | None = None


def durability(
    observables: dict[str, Callable],
    perturb: Callable,
    *,
    seed: int,
    settle: int,
    observe: int,
    tail_fraction: float = 0.2,
    **cfg_kwargs,
) -> dict[str, DurabilityResult]:
    """Displace the system, measure whether each observable returns.

    observables: name -> fn(engine) -> float, read once per tick
    perturb:     fn(engine) -> None, applied once after settling, to the twin only

    Both copies are built with the same seed and stepped in lockstep, so the difference
    between them is the perturbation and nothing else. `pre_impulse_gap` must be ~0 or the
    measurement is meaningless — check it before trusting R.
    """
    ref = make_engine(seed, **cfg_kwargs)
    per = make_engine(seed, **cfg_kwargs)

    for _ in range(settle):
        ref.tick()
        per.tick()

    gaps = [abs(fn(per) - fn(ref)) for fn in observables.values()]
    pre_gap = max((g for g in gaps if g == g), default=0.0)

    perturb(per)

    # D0 is taken after ONE tick, not immediately after the impulse.
    #
    # Observables read from state.metrics lag by a tick: the metrics dict still holds
    # values computed during the previous tick, so an impulse applied to the fields is
    # invisible to them until the pipeline runs again. Measuring D0 before that tick gives
    # D0 = 0 for every metric-based observable and R = nan. Field-computed observables do
    # not have this lag, so the bug is silent for them — which is exactly how it was
    # missed. One tick makes the definition uniform across both kinds.
    ref.tick()
    per.tick()
    d0 = {k: fn(per) - fn(ref) for k, fn in observables.items()}

    traj: dict[str, list[float]] = {k: [] for k in observables}
    ref_traj: dict[str, list[float]] = {k: [fn(ref)] for k, fn in observables.items()}
    for _ in range(observe - 1):
        ref.tick()
        per.tick()
        for k, fn in observables.items():
            traj[k].append(fn(per) - fn(ref))
            ref_traj[k].append(fn(ref))

    out = {}
    for k in observables:
        series = traj[k]
        tail = max(1, int(len(series) * tail_fraction))
        d_end = statistics.median(series[-tail:])
        R = abs(d_end) / abs(d0[k]) if d0[k] else float("nan")
        out[k] = DurabilityResult(
            observable=k, D0=d0[k], D_end=d_end, R=R,
            verdict=classify(R), trajectory=series, pre_impulse_gap=pre_gap,
            reference_trajectory=ref_traj[k],
        )
    return out


def sweep(grid: dict[str, Sequence], base: dict | None = None) -> Iterable[dict]:
    """Yield one config dict per point of the cartesian product of `grid`, over `base`."""
    base = dict(base or {})
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        yield cfg


def assert_params_live(names: Iterable[str], repo_root) -> dict[str, bool]:
    """Check each parameter is actually read by an operator before sweeping it.

    A dead config field silently inflates a robustness claim: varying it produces
    bit-identical runs that look like independent confirmations. `mass_gen_coeff` is
    declared in config.py and read by ZERO operators — memory.py computes
    gamma_local = diseq2/total_field2 directly — and it was swept in POC-02 and POC-03
    before anyone noticed the runs were duplicates.
    """
    import subprocess
    live = {}
    for n in names:
        r = subprocess.run(["git", "grep", "-l", n, "--", "src/v3/operators"],
                           cwd=repo_root, capture_output=True, text=True)
        live[n] = bool(r.stdout.strip())
    return live


def one_at_a_time(base: dict, factors: dict[str, Sequence[float]]) -> Iterable[dict]:
    """Vary one parameter at a time from `base`, by multiplicative factors.

    Preferred over a full cartesian product when the question is basin WIDTH: a full grid
    conflates 'which parameter matters' with 'how far can each move', and costs
    exponentially more runs to answer a question that is per-parameter.
    """
    yield dict(base) | {"_varied": "none", "_factor": 1.0}
    for name, facs in factors.items():
        for f in facs:
            if f == 1.0:
                continue
            cfg = dict(base)
            cfg[name] = base[name] * f
            cfg["_varied"] = name
            cfg["_factor"] = f
            yield cfg
