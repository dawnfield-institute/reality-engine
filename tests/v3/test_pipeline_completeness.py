"""The description-vs-behaviour gate.

Every problem of consequence found in this repo was a system reporting success about
something other than what it was doing: pytest.ini gating the wrong engine generation, a
PAC audit that could not fail, a scorecard measuring an engine missing six operators, and
CLAUDE.md documenting a 16-operator pipeline while 12 ran.

Tests check behaviour against behaviour. Docs state intent. Nothing bound them, so drift
was silent by construction and every instance was found by accident.

This binds them for the one case that matters most here: an operator cannot be implemented
and then silently left out of the loop.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import src.v3.operators as operators_pkg
from src.v3.engine.pipelines import (
    DEFAULT_OPERATORS,
    EXCLUDED,
    build_default_pipeline,
    build_full_pipeline,
)


def _implemented_operators() -> dict[str, type]:
    """Every Operator/Integrator class defined under src/v3/operators."""
    found: dict[str, type] = {}
    for mod_info in pkgutil.iter_modules(operators_pkg.__path__):
        if mod_info.name in ("protocol", "__init__"):
            continue
        mod = importlib.import_module(f"src.v3.operators.{mod_info.name}")
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != mod.__name__:
                continue
            if name.endswith(("Operator", "Integrator")):
                found[name] = obj
    return found


def test_every_operator_is_accounted_for():
    """No operator may be implemented and silently absent from the pipeline.

    It is either in DEFAULT_OPERATORS or in EXCLUDED with a stated reason. This is the
    check that would have caught six missing operators — including ActualizationOperator,
    without which the engine has no Delta buffer and cannot express P + A + Delta = C.
    """
    implemented = _implemented_operators()
    accounted = {c.__name__ for c in DEFAULT_OPERATORS} | {c.__name__ for c in EXCLUDED}
    unaccounted = sorted(set(implemented) - accounted)
    assert not unaccounted, (
        f"{len(unaccounted)} operator(s) implemented but neither in DEFAULT_OPERATORS nor "
        f"EXCLUDED: {unaccounted}. Add to the pipeline, or to EXCLUDED with the reason and "
        f"the measured effect."
    )


def test_exclusions_state_a_reason():
    """An exclusion without a reason is indistinguishable from an accident."""
    for cls, reason in EXCLUDED.items():
        assert reason and len(reason) > 40, (
            f"{cls.__name__} is excluded without a substantive reason. State why, and what "
            f"was measured."
        )


def test_default_and_excluded_are_disjoint():
    overlap = {c.__name__ for c in DEFAULT_OPERATORS} & {c.__name__ for c in EXCLUDED}
    assert not overlap, f"operators both default and excluded: {sorted(overlap)}"


def test_default_pipeline_matches_declaration():
    """The built pipeline must be exactly what DEFAULT_OPERATORS declares, in order."""
    built = [type(op).__name__ for op in build_default_pipeline()]
    declared = [c.__name__ for c in DEFAULT_OPERATORS]
    assert built == declared


def test_full_pipeline_runs_and_activates_the_delta_buffer():
    """The full pipeline must run, and must make state.P live.

    P is the unactualized potential buffer — the engine's Delta term. Under the default
    pipeline it is all zeros, because ActualizationOperator is not wired in. This asserts
    the full pipeline actually restores it, so a future refactor cannot quietly return the
    engine to a state where Delta does not exist.
    """
    import torch

    from src.v3.engine.config import SimulationConfig
    from src.v3.engine.engine import Engine

    torch.manual_seed(7)
    cfg = SimulationConfig(nu=16, nv=8, noise_scale=0.0)
    eng = Engine(config=cfg, pipeline=build_full_pipeline())
    eng.initialize(mode="big_bang")
    for _ in range(50):
        eng.tick()

    assert torch.isfinite(eng.state.E).all()
    assert torch.isfinite(eng.state.I).all()
    assert (eng.state.P != 0).any(), "Delta buffer (state.P) is inert under the full pipeline"
