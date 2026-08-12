"""The description-vs-behaviour gate.

25 sites outside `archive/` construct their own `Pipeline([...])`, from 8 to 16 operators.
Nothing declares which is canonical, so results are not comparable across scripts and
spikes — two can disagree because they ran different physics, and nothing says so.

These tests bind the declaration to the code: every implemented operator must be accounted
for, and the difference between the canonical pipeline and the dashboard's reduced one
must be exactly the documented set.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import src.v3.operators as operators_pkg
from src.v3.engine.pipelines import (
    CANONICAL,
    DASHBOARD_REDUCED,
    REDUCED_OMITS,
    UNUSED,
    build_canonical_pipeline,
    build_dashboard_pipeline,
)


def _implemented_operators() -> dict[str, type]:
    found: dict[str, type] = {}
    for mod_info in pkgutil.iter_modules(operators_pkg.__path__):
        if mod_info.name in ("protocol", "__init__"):
            continue
        mod = importlib.import_module(f"src.v3.operators.{mod_info.name}")
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__ and name.endswith(("Operator", "Integrator")):
                found[name] = obj
    return found


def test_every_operator_is_accounted_for():
    """No operator may be implemented and silently unreferenced."""
    implemented = set(_implemented_operators())
    accounted = ({c.__name__ for c in CANONICAL}
                 | {c.__name__ for c in DASHBOARD_REDUCED}
                 | {c.__name__ for c in UNUSED})
    missing = sorted(implemented - accounted)
    assert not missing, (
        f"{len(missing)} operator(s) implemented but in no declared pipeline and not in "
        f"UNUSED: {missing}. Add to CANONICAL, or to UNUSED with a reason."
    )


def test_reduced_omissions_are_declared_and_explained():
    """The reduced pipeline may differ from canonical only in documented ways.

    This is the check that would have caught the original drift: the dashboard ran 12
    operators under the name `build_default_pipeline` while __main__ and the scorecard ran
    16, and nothing recorded the difference.
    """
    canonical = {c.__name__ for c in CANONICAL}
    reduced = {c.__name__ for c in DASHBOARD_REDUCED}
    declared = {c.__name__ for c in REDUCED_OMITS}

    actual_omits = canonical - reduced - {"EulerIntegrator"}
    assert actual_omits == declared, (
        f"reduced pipeline omits {sorted(actual_omits)} but REDUCED_OMITS declares "
        f"{sorted(declared)}. Every difference must be stated with its measured effect."
    )
    for cls, reason in REDUCED_OMITS.items():
        assert reason and len(reason) > 40, f"{cls.__name__} omitted without a real reason"


def test_pipelines_match_their_declarations():
    assert [type(o).__name__ for o in build_canonical_pipeline()] == \
           [c.__name__ for c in CANONICAL]
    assert [type(o).__name__ for o in build_dashboard_pipeline()] == \
           [c.__name__ for c in DASHBOARD_REDUCED]


def test_canonical_activates_the_delta_buffer():
    """Under CANONICAL, state.P must be live; under the reduced pipeline it is inert.

    P is the unactualized potential buffer — the engine's Delta term. This pins the
    difference that matters, so a refactor cannot quietly return the canonical pipeline to
    a state where Delta does not exist.
    """
    import torch

    from src.v3.engine.config import SimulationConfig
    from src.v3.engine.engine import Engine

    def run(pipeline):
        torch.manual_seed(7)
        cfg = SimulationConfig(nu=16, nv=8, noise_scale=0.0)
        eng = Engine(config=cfg, pipeline=pipeline)
        eng.initialize(mode="big_bang")
        for _ in range(50):
            eng.tick()
        return eng.state

    canon = run(build_canonical_pipeline())
    assert torch.isfinite(canon.E).all() and torch.isfinite(canon.I).all()
    assert (canon.P != 0).any(), "Delta buffer inert under the CANONICAL pipeline"

    reduced = run(build_dashboard_pipeline())
    assert (reduced.P == 0).all(), (
        "reduced pipeline unexpectedly populates P — it has no ActualizationOperator, so "
        "this assumption about the difference between the two is no longer true"
    )
