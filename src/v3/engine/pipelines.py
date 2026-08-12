"""Canonical pipeline definitions.

The default pipeline previously lived in `dashboard/server.py` — a view module — and had
drifted from `CLAUDE.md`, which documented a 16-operator order. The real default was 12
operators plus an Euler integrator, and **six implemented operators were silently absent**.
Nothing detected that, because nothing compared the described physics to the running
physics.

Every operator must now appear in exactly one of `DEFAULT_OPERATORS` or `EXCLUDED` —
`tests/v3/test_pipeline_completeness.py` fails otherwise. An operator cannot be written
and then quietly left out of the loop again.
"""

from __future__ import annotations

from ..operators.actualization import ActualizationOperator
from ..operators.adaptive import AdaptiveOperator
from ..operators.charge_dynamics import ChargeDynamicsOperator
from ..operators.confluence import ConfluenceOperator
from ..operators.fusion import FusionOperator
from ..operators.gravity import GravitationalCollapseOperator
from ..operators.integrator import EulerIntegrator
from ..operators.memory import MemoryOperator
from ..operators.normalization import NormalizationOperator
from ..operators.phi_cascade import PhiCascadeOperator
from ..operators.protocol import Pipeline
from ..operators.qbe import QBEOperator
from ..operators.rbf import RBFOperator
from ..operators.sec_tracking import SECTrackingOperator
from ..operators.spin_statistics import SpinStatisticsOperator
from ..operators.temperature import TemperatureOperator
from ..operators.thermal_noise import ThermalNoiseOperator
from ..operators.time_emergence import TimeEmergenceOperator
from ..operators.unified_force import UnifiedForceOperator

# --- the pipeline that actually runs -------------------------------------------------
DEFAULT_OPERATORS = [
    RBFOperator,
    QBEOperator,
    EulerIntegrator,
    MemoryOperator,
    GravitationalCollapseOperator,
    FusionOperator,
    ConfluenceOperator,
    TemperatureOperator,
    ThermalNoiseOperator,
    NormalizationOperator,
    AdaptiveOperator,
    TimeEmergenceOperator,
]

# --- implemented, deliberately not in the default ------------------------------------
#
# Each entry states WHY, and the measured effect. Measured 2026-08-11 at 32x16, 1500
# ticks, noise off, seed 7 — every one of these runs stably: no crash, no non-finite
# fields, |max field| within 0.05% of baseline. **None was excluded because it is
# broken.** The omission was drift, not a decision, and these notes record what is known
# so the next choice is informed.
EXCLUDED: dict[type, str] = {
    ActualizationOperator:
        "MAR-gated integration; REPLACES EulerIntegrator rather than adding to it "
        "(it contains its own Euler fallback). Wiring it in makes state.P — the "
        "unactualized potential buffer, the engine's Delta term — live: |P| rises to "
        "~12.4 over 475/512 cells and saturates, and including P halves apparent ledger "
        "drift (0.181 -> 0.082 at t=3000). M_total +8.9%. This is the single most "
        "consequential exclusion: without it the engine has no Delta and no "
        "reconciliation, so the validated P+A+Delta=C model cannot be expressed.",
    PhiCascadeOperator:
        "Fibonacci two-step memory for phi-spaced mass levels. Runs stably; M_total "
        "+0.5%. Theoretically load-bearing — phi structure is the point of the framework.",
    SECTrackingOperator:
        "Read-only SEC metrics: entropy, info_fraction, cascade depth. Zero effect on "
        "dynamics (M_total identical). Its absence is why info_fraction is missing from "
        "metrics, which experiments have worked around by recomputing it from fields.",
    SpinStatisticsOperator:
        "Emergent Pauli exclusion from information cost. Runs stably; M_total -6.4%.",
    ChargeDynamicsOperator:
        "EM-like forces from charge field Q. Runs stably but has NO measured effect "
        "(M_total identical) — FieldState carries no Q field, so it is likely inert as "
        "wired. Investigate before including.",
    UnifiedForceOperator:
        "Combined gravity + EM. Superseded in the default by the separate "
        "GravitationalCollapseOperator; including both would double-count gravity.",
}


def build_default_pipeline() -> Pipeline:
    """The pipeline the engine actually runs."""
    return Pipeline([op() for op in DEFAULT_OPERATORS])


def build_full_pipeline() -> Pipeline:
    """Every operator that alters dynamics, including those excluded by default.

    Actualization replaces Euler; UnifiedForce is left out to avoid double-counting
    gravity. Provided so the excluded physics can be measured rather than argued about.
    """
    ops = [ActualizationOperator if op is EulerIntegrator else op
           for op in DEFAULT_OPERATORS]
    insert_at = ops.index(MemoryOperator)
    for extra in (PhiCascadeOperator, SpinStatisticsOperator):
        ops.insert(insert_at + 1, extra)
    ops.append(SECTrackingOperator)
    return Pipeline([op() for op in ops])
