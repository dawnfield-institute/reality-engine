"""Fixtures for the v4 particle substrate (proof_of_concepts/v4).

The substrate had zero test coverage until 2026-09-05; pytest.ini collected tests/v3 only. These
tests exist because a numerical bound bound silently for fifteen runs (see
proof_of_concepts/v4/poc_07_particle_substrate/journals/2026-08-28_the-clamp-is-the-equation-of-motion.md).
Everything runs on CPU at a size that reproduces the defect in seconds; see `_proxy.py`.
"""

from __future__ import annotations

import pytest

from ._proxy import P, PROXY, run_marks


@pytest.fixture(scope="module")
def proxy_run():
    """One exp_11-density run on the CANONICAL pipeline, shared by the module's tests."""
    return run_marks(P.ParticleConfig(**PROXY))


@pytest.fixture
def proxy_config():
    return P.ParticleConfig(**PROXY)
