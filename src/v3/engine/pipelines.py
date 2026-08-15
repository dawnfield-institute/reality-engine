"""Canonical pipeline definitions.

**The problem this solves is pipeline proliferation, not a missing operator.**

25 sites outside `archive/` construct their own `Pipeline([...])`, ranging from 8 to 16
operators: 7 sites use 16, 8 use 15, 5 use 14, and the rest fewer. Scripts and spikes each
hand-roll one, so **results are not comparable across them** — two spikes can disagree
because they ran different physics, and nothing says so.

`CANONICAL` is the 16-operator pipeline used by `src/v3/__main__.py` (the engine entry
point), `scripts/physics_scorecard.py`, and `spikes/theory_integration/harness.py`. It is
what `CLAUDE.md` documents. New code should use `build_canonical_pipeline()` rather than
assembling another variant.

`DASHBOARD_REDUCED` is the 12-operator pipeline that `dashboard/server.py` used under the
misleading name `build_default_pipeline`. It omits six operators, including
`ActualizationOperator` — so under it `state.P`, the unactualized potential buffer and the
engine's Δ term, is **all zeros**. Retained because the dashboard depends on it and its
performance characteristics may be why it exists, but it is NOT the default physics.

`tests/v3/test_pipeline_completeness.py` asserts every implemented operator is accounted
for, and that the two variants differ only by the documented set.
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
from ..operators.pac_balance import PACBalanceOperator
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

# --- the canonical 16 — __main__, scorecard, theory_integration, and CLAUDE.md ---------
CANONICAL = [
    RBFOperator,
    QBEOperator,
    ActualizationOperator,      # replaces EulerIntegrator — MAR-gated integration
    MemoryOperator,
    PhiCascadeOperator,
    GravitationalCollapseOperator,
    SpinStatisticsOperator,
    ChargeDynamicsOperator,
    FusionOperator,
    ConfluenceOperator,
    TemperatureOperator,
    ThermalNoiseOperator,
    NormalizationOperator,
    SECTrackingOperator,
    AdaptiveOperator,
    TimeEmergenceOperator,
]

# --- the dashboard's reduced 12 -------------------------------------------------------
DASHBOARD_REDUCED = [
    RBFOperator,
    QBEOperator,
    EulerIntegrator,            # NOT ActualizationOperator — state.P stays zero
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

# Operators in CANONICAL but not in DASHBOARD_REDUCED, with the measured cost of dropping
# them (32x16, 1500 ticks, noise off, seed 7 — all run stably, none is broken):
REDUCED_OMITS: dict[type, str] = {
    ActualizationOperator:
        "MAR-gated integration. Its absence leaves state.P — the unactualized potential "
        "buffer, the engine's Delta term — all zeros, so P + A + Delta = C cannot be "
        "expressed. Under CANONICAL, |P| rises to ~12.4 across 475/512 cells and "
        "saturates, and including P halves apparent ledger drift (0.181 -> 0.082 at "
        "t=3000). M_total +8.9%.",
    PhiCascadeOperator:
        "Fibonacci two-step memory producing phi-spaced mass levels. Runs stably; "
        "M_total +0.5%. Theoretically load-bearing even though the bulk effect is small "
        "— phi structure is the framework's central claim.",
    SpinStatisticsOperator:
        "Emergent Pauli exclusion from information cost. Runs stably; M_total -6.4%, the "
        "largest dynamical effect of any omitted operator. Its absence means the reduced "
        "pipeline has no exclusion principle at all.",
    SECTrackingOperator:
        "Read-only SEC metrics — entropy, info_fraction, cascade depth. No dynamical "
        "effect, but its absence is why info_fraction is missing from metrics under the "
        "reduced pipeline.",
    ChargeDynamicsOperator:
        "EM-like forces from charge field Q. No measured effect on bulk mass, but the "
        "earlier note here — that it is inert because FieldState carries no Q field — was "
        "WRONG. It computes Q itself from the cross-gradient dE/du - dI/dv, solves a "
        "Poisson potential, and builds a genuine VECTOR force (force_u, force_v). It then "
        "takes the DIVERGENCE of that force and adds the scalar to dE_dt, discarding the "
        "curl. Measured at 128x128 after 3000 ticks: |curl| rms 0.714 against |div| rms "
        "0.799, a ratio of 0.894 — roughly half the EM force thrown away every tick. This "
        "matters because the corpus derives EM as the ANTISYMMETRIC (curl) projection and "
        "gravity as the symmetric (divergence) one, so projecting the charge force onto "
        "its divergence routes EM through the gravity channel and drops the part that is "
        "EM. Every retained force in this engine is the gradient of a scalar potential, "
        "and gradient flows have point attractors — they make blobs, not filaments.",
}

# Implemented but in neither variant.
UNUSED: dict[type, str] = {
    PACBalanceOperator:
        "EXPERIMENTAL replacement for RBFOperator, from exp_30's ADE reconstruction: "
        "B = lambda * P_anti[(E-I)/(1+alpha*|M|)]. Not in CANONICAL — it is under "
        "evaluation, not adopted. Measured against RBF: spectral growth tilt 2.23x vs "
        "9.09x (RBF's laplacian damps as k^2, piling power into the box mode), and mass "
        "declines where RBF grows it. It does NOT produce spatial structure — correlation "
        "length is 1.0 cell for both, i.e. neighbouring cells are uncorrelated. Adopting "
        "it changes the spectrum, not the absence of structure.",
    UnifiedForceOperator:
        "Combined gravity + EM. Not in CANONICAL — including it alongside "
        "GravitationalCollapseOperator would double-count gravity. Used only by "
        "scripts/validate_unified_force.py.",
}


def build_canonical_pipeline() -> Pipeline:
    """The 16-operator pipeline. Use this unless you have a stated reason not to."""
    return Pipeline([op() for op in CANONICAL])


def build_dashboard_pipeline() -> Pipeline:
    """The dashboard's reduced 12. Runs without MAR actualization — state.P stays zero."""
    return Pipeline([op() for op in DASHBOARD_REDUCED])


# Back-compat: `build_default_pipeline` was the dashboard's reduced pipeline under a name
# that implied it was the engine's default. It was not — __main__ and the scorecard both
# use the canonical 16. Kept as an alias so existing imports do not break, but new code
# should name which pipeline it wants.
build_default_pipeline = build_dashboard_pipeline
