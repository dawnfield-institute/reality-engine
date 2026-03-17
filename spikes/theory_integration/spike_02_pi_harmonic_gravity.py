"""Spike 02: Pi-harmonic modulation in gravity operator.

Theory source: lambda_Higgs * 4*pi = phi (M5 exp_07).
    Circular phase space (two full revolutions) resolves to golden-ratio
    stability. The Higgs boson is harmonic resonance in the scalar potential.
    Gravity should be phase-locked to the same harmonic structure.

Hypothesis: Gravity coupling G_local should include pi-harmonic modulation
    from the E-I phase angle. This creates phase coherence between gravity
    and the actualization cycle (which already has pi/2 modulation).

    The key identity: 4*pi * lambda = phi suggests gravity at depth F_183
    shares the same phase structure as the Higgs at depth F_7. The
    pi-harmonic factor enforces this cross-depth coherence.

Variants:
    A) G_local *= (1 + cos(2*phase)) / 2 — period-pi modulation
    B) G_local *= cos²(phase) — sharper, zero at balance points
    C) xi_mod raised to pi/4 power — softer harmonic coupling
"""

import math
import torch
from harness import (
    default_pipeline, default_config, run_and_score,
    print_result, print_comparison,
)
from src.v3.operators.gravity import GravitationalCollapseOperator

_EPS = 1e-12
_PHI = (1 + math.sqrt(5)) / 2


class PiHarmonicGravityOperator(GravitationalCollapseOperator):
    """Gravity with pi-harmonic phase modulation."""

    def __init__(self, mode="cos2", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        # Compute phase before parent modifies state
        phase = torch.atan2(state.I, state.E + _EPS)

        if self.mode == "cos2":
            pi_mod = torch.cos(phase).pow(2)
        elif self.mode == "half":
            pi_mod = (1.0 + torch.cos(2.0 * phase)) / 2.0
        elif self.mode == "xi_pi":
            # Softer: raise existing xi_mod to pi/4 power
            # This doesn't need phase, it modifies the xi_mod exponent
            pi_mod = None
        else:
            pi_mod = torch.ones_like(phase)

        # Run parent gravity
        result = super().__call__(state, config, bus)

        if pi_mod is not None:
            # Modulate the gravity-induced mass change
            dM = result.M - state.M
            dM_modulated = dM * pi_mod
            M_new = state.M + dM_modulated
            M_new = torch.clamp(M_new, min=0.0)

            # PAC correction
            dM_actual = M_new - state.M
            # Original drain was for unmodulated dM
            orig_dM_actual = result.M - state.M
            drain_diff = (dM_actual - orig_dM_actual) * 0.5
            E_new = result.E - drain_diff
            I_new = result.I - drain_diff

            metrics = dict(result.metrics)
            metrics["grav_pi_mod_mean"] = pi_mod.mean().item()
            return result.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)

        return result


def make_pipeline_with_gravity(grav_op):
    from src.v3.operators.protocol import Pipeline
    from src.v3.operators.rbf import RBFOperator
    from src.v3.operators.qbe import QBEOperator
    from src.v3.operators.actualization import ActualizationOperator
    from src.v3.operators.memory import MemoryOperator
    from src.v3.operators.phi_cascade import PhiCascadeOperator
    from src.v3.operators.spin_statistics import SpinStatisticsOperator
    from src.v3.operators.charge_dynamics import ChargeDynamicsOperator
    from src.v3.operators.fusion import FusionOperator
    from src.v3.operators.confluence import ConfluenceOperator
    from src.v3.operators.temperature import TemperatureOperator
    from src.v3.operators.thermal_noise import ThermalNoiseOperator
    from src.v3.operators.normalization import NormalizationOperator
    from src.v3.operators.sec_tracking import SECTrackingOperator
    from src.v3.operators.adaptive import AdaptiveOperator
    from src.v3.operators.time_emergence import TimeEmergenceOperator

    return Pipeline([
        RBFOperator(), QBEOperator(), ActualizationOperator(),
        MemoryOperator(), PhiCascadeOperator(),
        grav_op,
        SpinStatisticsOperator(), ChargeDynamicsOperator(),
        FusionOperator(),
        ConfluenceOperator(), TemperatureOperator(), ThermalNoiseOperator(),
        NormalizationOperator(), SECTrackingOperator(),
        AdaptiveOperator(), TimeEmergenceOperator(),
    ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = default_config(device=device)
    ticks = 10000

    print("=" * 80)
    print("  SPIKE 02: Pi-Harmonic Modulation in Gravity Operator")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks")
    print("=" * 80)

    results = []

    # Baseline
    print("\n  Running baseline...", flush=True)
    r = run_and_score("baseline", default_pipeline(), config, ticks=ticks)
    results.append(r)
    print_result(r)

    # Variant A: cos²(phase)
    print("\n  Running cos²(phase) gravity...", flush=True)
    p = make_pipeline_with_gravity(PiHarmonicGravityOperator(mode="cos2"))
    r = run_and_score("grav_cos2", p, config, ticks=ticks)
    results.append(r)
    print_result(r)

    # Variant B: (1+cos(2*phase))/2
    print("\n  Running half-cos gravity...", flush=True)
    p = make_pipeline_with_gravity(PiHarmonicGravityOperator(mode="half"))
    r = run_and_score("grav_half_cos", p, config, ticks=ticks)
    results.append(r)
    print_result(r)

    print("\n" + "=" * 80)
    print("  COMPARISON")
    print("=" * 80)
    print_comparison(results)

    best = min(results, key=lambda r: r["avg_t1"])
    print(f"\n  BEST: {best['label']}  avg_t1={best['avg_t1']:.1f}%")


if __name__ == "__main__":
    main()
