"""Spike 01: Pi-harmonic modulation in memory operator.

Theory source: Pi Harmonics in Geometry (internal/out_folder/0029), exp_07 Higgs.
    lambda_Higgs * 4*pi = phi — circular phase space resolves to golden ratio.
    Pi creates "minimum variance in Möbius coherence" and "stabilizes recursive
    symbolic collapse."

Hypothesis: Mass generation should be phase-locked to the E-I oscillation cycle.
    The actualization operator already uses pi/2 modulation. Memory should too —
    mass crystallizes preferentially at specific phases of the E-I cycle, not
    uniformly. This enforces the harmonic structure that the theory demands.

Implementation: Modulate mass_gen by cos²(phase) where phase = atan2(I, E).
    This creates pi-periodic modulation: mass generation peaks when E or I
    dominates (active collapse), suppressed when E ≈ I (balanced, no signal).

Variants tested:
    A) cos²(atan2(I,E)) — full pi-periodic modulation
    B) (1 + cos(2*atan2(I,E))) / 2 — equivalent but explicit period-2
    C) cos²(atan2(I,E)) * phi/(phi+1) — scaled by SEC duty cycle
"""

import math
import torch
from harness import (
    default_pipeline, default_config, run_and_score,
    print_result, print_comparison, PHI
)
from src.v3.operators.memory import MemoryOperator

_EPS = 1e-12
_PHI = (1 + math.sqrt(5)) / 2


class PiHarmonicMemoryOperator(MemoryOperator):
    """Memory with pi-harmonic modulation of mass generation."""

    def __init__(self, mode="cos2"):
        super().__init__()
        self.mode = mode

    @torch.no_grad()
    def __call__(self, state, config, bus=None):
        # Compute pi-harmonic modulation from E-I phase
        phase = torch.atan2(state.I, state.E + _EPS)

        if self.mode == "cos2":
            # cos²(phase): peaks at E-dominant (0) and I-dominant (pi), zero at balance
            pi_mod = torch.cos(phase).pow(2)
        elif self.mode == "duty":
            # Same but scaled by SEC duty cycle phi/(phi+1)
            pi_mod = torch.cos(phase).pow(2) * _PHI / (_PHI + 1.0)
        else:
            pi_mod = torch.ones_like(phase)

        # Run normal memory operator
        result = super().__call__(state, config, bus)

        # Apply modulation to the mass change only (not the whole state)
        # We need to modulate dM, so we compute what the operator did and scale it
        dM = result.M - state.M
        dM_modulated = dM * pi_mod

        # Recompute PAC-conserving drain
        M_new = state.M + dM_modulated
        M_new = torch.clamp(M_new, min=0.0)
        dM_actual = M_new - state.M
        net_pac_drain = dM_actual * 0.5
        E_new = state.E - net_pac_drain
        I_new = state.I - net_pac_drain

        metrics = dict(result.metrics)
        metrics["pi_mod_mean"] = pi_mod.mean().item()
        metrics["pi_mod_std"] = pi_mod.std().item()

        return result.replace(E=E_new, I=I_new, M=M_new, metrics=metrics)


def make_pipeline_with(memory_op):
    """Replace MemoryOperator in standard pipeline."""
    from src.v3.operators.protocol import Pipeline
    from src.v3.operators.rbf import RBFOperator
    from src.v3.operators.qbe import QBEOperator
    from src.v3.operators.actualization import ActualizationOperator
    from src.v3.operators.phi_cascade import PhiCascadeOperator
    from src.v3.operators.gravity import GravitationalCollapseOperator
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
        memory_op, PhiCascadeOperator(),
        GravitationalCollapseOperator(),
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
    print("  SPIKE 01: Pi-Harmonic Modulation in Memory Operator")
    print(f"  Device: {device} | Grid: 128x64 | {ticks} ticks")
    print("=" * 80)

    results = []

    # Baseline
    print("\n  Running baseline...", flush=True)
    r = run_and_score("baseline", default_pipeline(), config, ticks=ticks)
    results.append(r)
    print_result(r)

    # Variant A: cos²(phase)
    print("\n  Running cos²(phase) modulation...", flush=True)
    pipeline_a = make_pipeline_with(PiHarmonicMemoryOperator(mode="cos2"))
    r = run_and_score("pi_cos2", pipeline_a, config, ticks=ticks)
    results.append(r)
    print_result(r)

    # Variant C: cos²(phase) * duty cycle
    print("\n  Running cos²(phase) * duty_cycle...", flush=True)
    pipeline_c = make_pipeline_with(PiHarmonicMemoryOperator(mode="duty"))
    r = run_and_score("pi_cos2_duty", pipeline_c, config, ticks=ticks)
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
